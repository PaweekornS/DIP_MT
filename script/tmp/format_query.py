from vllm import LLM, SamplingParams
import pandas as pd
from tqdm import tqdm
import argparse
import torch

# Global variable
ROOT_DIR = "/project/lt200304-dipmt/paweekorn"

with open(f"{ROOT_DIR}/data/prompt/format.txt", 'r') as f:
    template = f.read()

def formatting_prompt(df):
    text_set = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prompt = template.format(
            ENGLISH=row["ENG"], 
            THAI=row['THA']
            )
        text_set.append(prompt)
    return text_set


def inference(model_dir, quant, text_set):
    quantization = None if quant == "None" else quant
    model = LLM(
        model=model_dir,
        quantization=quantization,
        max_model_len=16000,
        tensor_parallel_size=torch.cuda.device_count(),
        enable_prefix_caching=True,
        gpu_memory_utilization=0.5,
        enforce_eager=False,
    )

    decoding_params = SamplingParams(temperature=0.2,
                                 max_tokens=16000,
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
    df = pd.read_csv(args.dataset)
    text_set = formatting_prompt(df)

    results = inference(args.model_dir, args.quantization, text_set)
    response = [r.outputs[0].text for r in results]
    df['FORMAT'] = response
    
    model_id = args.model_dir.split('/')[-1]
    df.to_csv(f'./{model_id}_format.csv', index=False)

if __name__ == "__main__":
    main()