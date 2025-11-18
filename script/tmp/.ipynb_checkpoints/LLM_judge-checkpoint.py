from vllm import LLM, SamplingParams
import torch

import pandas as pd
import numpy as np

from tqdm import tqdm
import json

# Global variable
ROOT_DIR = "/project/lt200304-dipmt/paweekorn"

template = """You are an expert in English and Thai translation. Your task is to select the most suitable Thai translation for a given English source text. 
The translations are duplicated and may have subtle differences in meaning, tone, or formality.
Concern about the alignment of punctuation too.

You will be provided with:
1. An English source text.
2. A list of duplicated Thai translation candidates.

Your output must be a single JSON object with the following keys:
- `english_source`: The original English text.
- `selected_thai_translation`: The one Thai translation from the list that is most accurate, natural-sounding, and contextually appropriate.

---
### Input

**English Source:**
{ENGLISH}

**Thai Translation Candidates:**
{THAI}
"""

def data_prep(data_path):
    # data prep
    unique_goods = pd.read_csv(data_path)
    
    with open(f'{ROOT_DIR}/data/WIPO.json', 'r') as f:
        wipo_data = json.load(f)
        wipo_data = {int(k): v for k, v in wipo_data.items()}
        
    unique_goods["WIPO"] = unique_goods["NAME"].map(wipo_data)
    
    group_df = unique_goods.groupby("ENG").agg({"THA": list})
    group_df['LEN'] = group_df["THA"].apply(lambda x: len(x))
    duplicated_df = group_df[ group_df['LEN'] != 1 ].reset_index()
    

def formatting_prompt(df):
    def prepare_duplicate(thai_list):
        output = ""
        for i, text in enumerate(thai_list):
            output += f"{i+1}. {text}"
            output += "\n" if i != len(thai_list)-1 else ""
        return output

    text_set = []
    for _, row in tqdm(df.iterrows(), total=len(duplicated_df)):
        prompt = template.format(
            ENGLISH=row["ENG"], 
            THAI=prepare_duplicate(row["THA"])
            )
        text_set.append(prompt)
    return text_set


def inference(model_dir, text_set):
    model = LLM(
        model=model_dir,
        quantization="bitsandbytes",
        max_model_len=4096,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
    )

    decoding_params = SamplingParams(temperature=0.2,
                                 max_tokens=256,
                                 skip_special_tokens=True,
                                 repetition_penalty=1.15)

    results = model.generate(text_set, decoding_params)
    return results
    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=False, help="dataset dir", default=f"{ROOT_DIR}/data/unique_goods.csv")
    ap.add_argument("--model_dir", required=True, help="model for judging")
    args = ap.parse_args()
    
    # setup
    df = data_prep(args.dataset)
    text_set = formatting_prompt(df)

    # inference
    results = inference(args.model_dir)
    response = [r.outputs[0].text for r in results]

    # extract answer
    df['ANS'] = response
    model_id = model_dir.split('/')[-1]
    df.to_csv(f'./{model_id}.csv', index=False)

if __name__ == "__main__":
    main()