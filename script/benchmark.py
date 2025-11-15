from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pythainlp.tokenize import word_tokenize
from jiwer import cer

import pandas as pd
import numpy as np
from tqdm import tqdm
import glob


chencherry = SmoothingFunction()
def evaluate(reference: str, hypothesis: str) -> float:
    ref_tokens = word_tokenize(reference, engine="attacut")
    hyp_tokens = word_tokenize(hypothesis, engine="attacut")

    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=chencherry.method1)
    cer_score = cer(reference, hypothesis)

    return cer_score, bleu


def benchmark(fname, test_df):
    df = pd.read_csv(fname).fillna("")
    df['THA'] = test_df['THA'].tolist()
    
    per_file_metrics = {"cer": [], "bleu": []}
    for _, row in df.iterrows():
        wer, cer, bleu = evaluate(row['THA'], row['PRED_cleaned'])
        
        per_file_metrics["cer"].append(cer)
        per_file_metrics["bleu"].append(bleu)
    
    # Average per file
    per_file_metrics = {k: round(np.mean(v), 4) for k, v in per_file_metrics.items()}
    return per_file_metrics
        

# Example usage:
BASE_DIR = '/project/lt200304-dipmt/paweekorn'
test_df = pd.read_csv(f'{BASE_DIR}/data/DS01/test_v1.csv')
files = np.sort(glob.glob(f'{BASE_DIR}/data/infer-result/en2th/*.csv'))

result = []
for file in tqdm(files):
    fname = file.split('/')[-1].replace('.csv', '')
    name, method = fname.split('_')
    
    metrics = benchmark(file, test_df)
    metrics['fname'] = name;  metrics['type'] = method;
    result.append(metrics)

result_df = pd.DataFrame(result)

result_df.to_csv('/project/lt200304-dipmt/paweekorn/benchmark_final.csv', index=False, encoding='utf-8-sig')
