from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm
import argparse
import torch

# Global variable
ROOT_DIR = "/project/lt200304-dipmt/paweekorn"

with open(f"{ROOT_DIR}/data/prompt/condidate_selected.txt", 'r') as f:
    template = f.read()

def data_prep(data_path):
    # data prep
    unique_goods = pd.read_csv(data_path)
    
    group_df = unique_goods.groupby("ENG").agg({"THA": list, "NAME": "unique"})
    group_df['LEN'] = group_df["THA"].apply(lambda x: len(x))
    duplicated_df = group_df[ group_df['LEN'] != 1 ].reset_index()
    
    return duplicated_df
    
    
def prepare_duplicate(thai_list):
        output = ""
        for i, text in enumerate(thai_list):
            output += f"{i+1}. {text}"
            output += "\n" if i != len(thai_list)-1 else ""
        return output

def formatting_prompt(df):
    text_set = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = template.format(
            ENGLISH=row["ENG"], 
            THAI=prepare_duplicate(row['THA'])
            )
        text_set.append(prompt)
    return text_set


def inference(model_dir, quant, text_set):
    quantization = None if quant == "None" else quant
    model = LLM(
        model=model_dir,
        quantization=quantization,
        max_model_len=12000,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
        enforce_eager=False,
    )

    decoding_params = SamplingParams(temperature=0.2,
                                 max_tokens=512,
                                 skip_special_tokens=True,
                                 repetition_penalty=1.15)

    results = model.generate(text_set, decoding_params)
    return results
    

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=False, help="dataset dir", default=f"{ROOT_DIR}/data/unique_goods.csv")
    ap.add_argument("--model_dir", required=True, help="model for judging")
    ap.add_argument("--quantization", required=True, help="quants method")
    args = ap.parse_args()
    
    # setup
    df = data_prep(args.dataset)
    text_set = formatting_prompt(df)

    # inference
    results = inference(args.model_dir, args.quantization, text_set)
    response = [r.outputs[0].text for r in results]

    # extract answer
    df['ANS'] = response
    model_id = args.model_dir.split('/')[-1]
    df['ANS'].to_csv(f'./{model_id}.csv', index=False)

if __name__ == "__main__":
    main()