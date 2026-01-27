from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import argparse

import pandas as pd
import json
import sys
import os

import faiss
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

ROOT_DIR = "/project/lt200304-dipmt/paweekorn"
sys.path.append(os.path.join(ROOT_DIR, "script"))
from utils.retrieval import retrieval_setup, formatting_prompt
from utils.llm import init_model, train_model

with open(f"{ROOT_DIR}/data/prompt/base_en2th.txt", "r") as f:
    instruction = f.read()
    
# ===================
# Retrieval setup
# ===================
retriever = "all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=f"{ROOT_DIR}/models/retriever/{retriever}",)
vectorstore = FAISS.load_local(f"{ROOT_DIR}/vector/en2th/{retriever}", embeddings, allow_dangerous_deserialization=True)
gpu_index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, vectorstore.index)
vectorstore.index = gpu_index

retrieval_setup(embeddings, vectorstore, _cur=None, _instruction=instruction, _k=3)  # init function

# ===================
# Dataset
# ===================
def data_prep(dataset):
    df = pd.read_csv(dataset)

    with open('/project/lt200304-dipmt/paweekorn/data/wipo/WIPO.json', 'r') as f:
        wipo_data = json.load(f)
        wipo_data = {int(k): v for k, v in wipo_data.items()}

    df['WIPO'] = df['NAME'].map(wipo_data)
    df.drop_duplicates(subset=['ENG', 'THA'], inplace=True)
    return df
    
# ===================
# main pipeline
# ===================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dataset", required=True, default="/project/lt200304-dipmt/paweekorn/data/train_40k.csv")
    ap.add_argument("--test_dataset", required=True, default="/project/lt200304-dipmt/paweekorn/data/test_set.csv")
    ap.add_argument("--model_dir", required=True, help="model for fine-tuning")
    ap.add_argument("--load_in_4bit", type=bool, help="quantization", default=True)
    ap.add_argument("--rank", type=int, required=False, help="lora rank", default=16)
    ap.add_argument("--target_modules", type=lambda s: s.split(','), required=False, default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    args = ap.parse_args()

    # prepare data
    train_df = data_prep(args.train_dataset)
    test_df = data_prep(args.test_dataset)

    # model setup
    MODEL_ID = args.model_dir.split('/')[-1]
    print(f"\nInitialize model: {MODEL_ID}\n")
    model, tokenizer = init_model(model_path=args.model_dir, load_in_4bit=args.load_in_4bit, 
                                  rank=args.rank, target_modules=args.target_modules)
    tokenizer = get_chat_template(tokenizer, chat_template = "mistral") if "mistral" in args.model_dir else tokenizer

    print("Formatting Dataset")
    train_set = formatting_prompt(train_df, tokenizer, skip_first=False, finetuning=True)
    test_set = formatting_prompt(test_df, tokenizer, skip_first=False, finetuning=True)
    
    # training loop
    train_model(model, tokenizer, train_set, test_set, 
                output_dir=f"{ROOT_DIR}/models/adapter/{MODEL_ID}", epochs=2)

if __name__ == "__main__":
    main()
    