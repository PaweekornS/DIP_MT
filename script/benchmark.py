import pandas as pd
import numpy as np

from tqdm import tqdm
import glob
import os

from jiwer import cer, wer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pythainlp.tokenize import word_tokenize


def evaluate(reference: str, hypothesis: str) -> float:
    chencherry = SmoothingFunction()

    ref_tokens = [word_tokenize(reference, engine="newmm")] # Needs to be a list of lists
    hyp_tokens = word_tokenize(hypothesis, engine="newmm")

    bleu = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=chencherry.method1)
    meteor = meteor_score(ref_tokens, hyp_tokens)

    wer_score = wer(reference, hypothesis)
    cer_score = cer(reference, hypothesis)

    return wer_score, cer_score, bleu, meteor


def benchmark(fname, test_df):
    df = pd.read_csv(fname).fillna("")
    df['THA'] = test_df['THA'].tolist()
    
    per_file_metrics = {"wer": [], "cer": [], "bleu": [], "meteor": []}
    for _, row in df.iterrows():
        wer, cer, bleu, meteor = evaluate(row['THA'], row['PRED_cleaned'])
        
        per_file_metrics["wer"].append(wer)
        per_file_metrics["cer"].append(cer)
        per_file_metrics["bleu"].append(bleu)
        per_file_metrics["meteor"].append(meteor)
    
    # Average per file
    per_file_metrics = {k: round(np.mean(v), 4) for k, v in per_file_metrics.items()}

    return per_file_metrics
        

# Example usage:
test_df = pd.read_excel('/project/lt200304-dipmt/paweekorn/data/test_set.xlsx', index_col='ID')
files = np.sort(glob.glob('/project/lt200304-dipmt/paweekorn/results/*.csv'))

result = []
for file in tqdm(files):
    fname = file.split('/')[-1].replace('.csv', '')
    name, method = fname.split('_')
    
    metrics = benchmark(file, test_df)
    metrics['fname'] = name;  metrics['type'] = method;
    result.append(metrics)

result_df = pd.DataFrame(result)
result_df.to_csv('/project/lt200304-dipmt/paweekorn/benchmark.csv', index=False)
