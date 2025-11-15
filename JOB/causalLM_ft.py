import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import glob

import argparse
import torch
import json

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

ROOT_DIR = "/project/lt200304-dipmt/paweekorn"
embeddings = HuggingFaceEmbeddings(model_name=f"{ROOT_DIR}/models/retriever/all-MiniLM-L6-v2",)
vectorstore = FAISS.load_local(f"{ROOT_DIR}/vector/all-MiniLM-L6-v2", embeddings, allow_dangerous_deserialization=True)
with open(f"{ROOT_DIR}/data/prompt/base_prompt.txt", "r") as f: # system prompt
    instruction = f.read()


def data_prep(dataset):
    df = pd.read_csv(dataset)

    with open('/project/lt200304-dipmt/paweekorn/data/wipo/WIPO.json', 'r') as f:
        wipo_data = json.load(f)
        wipo_data = {int(k): v for k, v in wipo_data.items()}

    df['WIPO'] = df['NAME'].map(wipo_data)
    df.drop_duplicates(subset=['ENG', 'THA'], inplace=True)
    return df


def get_relevant_docs(query, k=4):
    query_embedding = embeddings.embed_query(query)
    docs = vectorstore.similarity_search_by_vector(query_embedding, k=k)
    
    relevant = ""
    for i, doc in enumerate(docs[1:]):
        relevant += f'''English: {doc.page_content}
Thai: {doc.metadata['thai']}\n
'''
    rag_result = f'''\n**Retrieved References:**\n{relevant} 
    **Note:** If the retrieved references contain identical English terms with different Thai translations (ambiguity), you must use your expert judgment to select the most appropriate and contextually accurate Thai translation for the current input.'''
    return rag_result


def init_model(model_dir, load_in_4bit, rank, target_modules):
    print("\nInitialize model:", model_dir.split('/')[-1], "\n")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=4096,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
        device_map="auto",
    )
    
    # lora setup
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.2,
        target_modules=target_modules,
        bias="none",
        random_state=3407,
    )
    
    return model, tokenizer


def formatting_prompt(df, tokenizer):    
    batch = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = [
            {"role": "user", "content": instruction.format(
                WIPO=row['WIPO'],
                RAG_DOC=get_relevant_docs(row['ENG'], k=4), 
                ENGLISH=row["ENG"])},
            {"role": "assistant", "content": row["THA"]}
        ]
        message = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        batch.append({'text': message})

    return Dataset.from_list(batch).shuffle(seed=42)


def training_loop(model, tokenizer, train_set, test_set, MODEL_ID):
    args = SFTConfig(
        output_dir=f"{ROOT_DIR}/models/adapter/{MODEL_ID}",
        overwrite_output_dir=True,
        logging_dir=f"{ROOT_DIR}/models/adapter/{MODEL_ID}/logs",
        dataset_text_field = "text",
        per_device_train_batch_size = 32,
        per_device_eval_batch_size = 32,
        gradient_accumulation_steps = 2,
        warmup_ratio = 0.03,
        num_train_epochs = 2,
        learning_rate = 2e-5,
        logging_steps = 100,
        eval_steps = 100,
        eval_strategy = "steps",
        save_steps = 210,
        save_strategy="steps",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        report_to = "none",
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_set,
        eval_dataset = test_set,
        args = args,
        gradient_checkpoint=True,
    )

    train_stats = trainer.train()
    

def plot_history(MODEL_ID):
    stats = glob.glob(f"{ROOT_DIR}/models/adapter/{MODEL_ID}/checkpoint*")
    sorted_stat = sorted(stats, key=lambda x: int(x.split('-')[-1]))
    with open(f"{sorted_stat[-1]}/trainer_state.json") as f:
        stats = json.load(f)

    stat_df = pd.DataFrame(stats['log_history'])
    stat_df = stat_df.groupby("epoch", as_index=False).agg(lambda x: x.dropna().iloc[0] if x.dropna().size else None)

    # Plot
    plt.figure(figsize=(5, 3))
    plt.plot(stat_df["epoch"], stat_df["loss"], label="Train Loss")
    plt.plot(stat_df["epoch"],  stat_df["eval_loss"], label="Eval Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Evaluation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{ROOT_DIR}/models/adapter/{MODEL_ID}/trainer_state.png")
        
    
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dataset", required=True, default="/project/lt200304-dipmt/paweekorn/data/train_40k.csv")
    ap.add_argument("--test_dataset", required=True, default="/project/lt200304-dipmt/paweekorn/data/test_set.csv")
    ap.add_argument("--model_dir", required=True, help="model for fine-tuning")
    ap.add_argument("--load_in_4bit", type=bool, help="quantization", default=True)
    ap.add_argument("--rank", type=int, required=False, help="lora rank", default=16)
    ap.add_argument("--target_modules", type=lambda s: s.split(','), required=False, default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    args = ap.parse_args()
    
    MODEL_ID = args.model_dir.split('/')[-1]
    
    # setup
    train_df = data_prep(args.train_dataset)
    test_df = data_prep(args.test_dataset)
    model, tokenizer = init_model(args.model_dir, args.load_in_4bit, args.rank, args.target_modules)
    tokenizer = get_chat_template(tokenizer, chat_template = "mistral") if "mistral" in args.model_dir else tokenizer
    
    # formatting dataset
    train_set = formatting_prompt(train_df, tokenizer)
    test_set = formatting_prompt(test_df, tokenizer)
    
    # model training
    training_loop(model, tokenizer, train_set, test_set, MODEL_ID)
    plot_history(MODEL_ID)

if __name__ == "__main__":
    main()
    