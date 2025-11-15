from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pythainlp.tokenize import word_tokenize
from jiwer import wer
import torch.distributed as dist
import torch

import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

import pandas as pd
import numpy as np
from tqdm import tqdm

import json
import re
import argparse


ROOT_DIR = "/project/lt200304-dipmt/paweekorn"
with open(f"{ROOT_DIR}/data/prompt/base_prompt.txt", "r") as f:
    instruction = f.read()

embed_model = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=f"{ROOT_DIR}/models/retriever/{embed_model}",)
vectorstore = FAISS.load_local(f"{ROOT_DIR}/vector/{embed_model}", embeddings, allow_dangerous_deserialization=True)
gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, vectorstore.index)
vectorstore.index = gpu_index


def get_relevant_docs(query, k=3):
    query_embedding = embeddings.embed_query(query)
    docs = vectorstore.similarity_search_by_vector(query_embedding, k=k)
    
    relevant = ""
    for i, doc in enumerate(docs):
        relevant += f'''English: {doc.page_content}
Thai: {doc.metadata['thai']}\n
'''
    rag_result = "\n## Retrieved References:\n" + relevant + "**Note:** If the retrieved references contain identical English terms with different Thai translations (ambiguity), you must use your expert judgment to select the most appropriate and contextually accurate Thai translation for the current input.\n"
    return rag_result


def formatting_prompt(df, tokenizer, isRAG):
    batch = []
    for _, row in df.iterrows():
        prompt = instruction.format(
            WIPO=row['WIPO'],
            RAG_DOC=get_relevant_docs(row['ENG'], 3) if isRAG else "", 
            ENGLISH=row['ENG']
            )
        chat = [{"role": "user", "content": prompt}]
        chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        batch.append(chat)
    
    return batch


def data_prep(dataset):
    df = pd.read_csv(dataset)

    with open(f'{ROOT_DIR}/data/wipo/WIPO.json', 'r') as f:
        wipo_data = json.load(f)

    wipo_data = {int(k): v for k, v in wipo_data.items()}
    df['WIPO'] = df['NAME'].map(wipo_data)
    
    return df


def inference(queries, model, lora_request):
    decoding_params = SamplingParams(
        temperature=0.2,
        max_tokens=4096,
        skip_special_tokens=True,
        repetition_penalty=1.15
    )

    results = model.generate(queries, decoding_params, lora_request=lora_request)
    response = [r.outputs[0].text for r in results]
    
    return response


def extract_json(text):
    text = text[text.rfind("{"):]
    pattern = r'''{\s*[\'\"]thai_translation[\'\"]:\s*[\'\"].*?[\'\"]\s*}'''
    matches = re.findall(pattern, text, re.DOTALL)

    if matches:
        try:
            loaded = json.loads(matches[0])
            return loaded['thai_translation']
        except json.JSONDecodeError as e:
            return np.nan
    else:
        return np.nan
    
    
def compute_score(df):    
    wer_result, bleu = [], []
    chencherry = SmoothingFunction().method1
    for _, row in df.iterrows():
        wer_result.append(wer(row['THA'], row['PRED_cleaned']))

        ref = word_tokenize(row['THA'], engine='attacut')
        hyp = word_tokenize(row['PRED_cleaned'], engine='attacut')
        
        bleu.append(sentence_bleu([ref], hyp, smoothing_function=chencherry))
        
    wer_avg = np.mean(wer_result);  bleu_avg = np.mean(bleu)

    print(f"\nAverage WER:", np.round(wer_avg, 4))
    print(f"Average BLEU:", np.round(bleu_avg, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, default="/project/lt200304-dipmt/paweekorn/data/test_set.xlsx")
    ap.add_argument("--model_dir", required=True, help="model for fine-tuning")
    ap.add_argument("--adapter_dir", required=False, help="fine-tuned adapter (optional)", default=None)
    ap.add_argument("--quantization", required=False, help="quantization", default="bitsandbytes")
    ap.add_argument("--is_rag", required=False, default=False)
    ap.add_argument("--save_dir", required=False, help="save directory", default=None)
    args = ap.parse_args()
    
    # setup dataset
    print("\nSetup chat template\n")
    test_df = data_prep(args.dataset)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    test_set = formatting_prompt(test_df, tokenizer, (args.is_rag.lower()=="true"))
    
    # init model
    lora_request = LoRARequest("lora_adapter", 1, args.adapter_dir) if args.adapter_dir else None
        
    print(f"\nInitialize model: {args.model_dir.split("/")[-1]}\n")
    model = LLM(
        model=args.model_dir,
        quantization=args.quantization,
        max_model_len=4096,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=True,
        gpu_memory_utilization=0.5,
        enforce_eager=False,
        enable_lora=True,
        max_lora_rank=64,
    )
    
    # Inference part
    infer_results = inference(test_set, model, lora_request)
    print("\nInference Done!")
    
    # Destroy the default process group
    if dist.is_initialized():
        dist.destroy_process_group()
    
    # save result and print metrics
    test_df['PRED'] = infer_results
    test_df['PRED_cleaned'] = test_df['PRED'].apply(extract_json).fillna("")
    
    if args.save_dir is not None:
        test_df[['PRED', 'PRED_cleaned']].to_csv(args.save_dir, index=False)
        print(f"Completely save result at {args.save_dir}")
    compute_score(test_df)
    
if __name__ == "__main__":
    main()
