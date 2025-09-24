import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (AutoTokenizer, M2M100ForConditionalGeneration,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, 
                          GenerationConfig, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model
import torch

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pythainlp.tokenize import word_tokenize


ROOT_DIR = "/project/lt200304-dipmt/paweekorn"
MODEL_ID = f"{ROOT_DIR}/models/base/nllb-3.3b"
SRC_LANG = "eng_Latn"
TGT_LANG = "tha_Thai"

# ================
# Data Prep
# ================

train_df = pd.read_csv(f'{ROOT_DIR}/data/train_40k.csv')
test_df = pd.read_excel(f'{ROOT_DIR}/data/test_set.xlsx')

train_df.drop_duplicates(subset=['ENG', 'THA'], inplace=True)

# ================
# Init Model
# ================
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16"
    )

model = M2M100ForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    use_cache=False,
    device_map=0
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
tokenizer.src_lang = SRC_LANG
tokenizer.tgt_lang = TGT_LANG

lora_cfg = LoraConfig(
    r=32, lora_alpha=32, lora_dropout=0.2,
    bias="none", target_modules=["q_proj","k_proj","v_proj","o_proj",
                                 "gate_proj","up_proj","down_proj"],
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_cfg)
model.gradient_checkpointing_enable()
model.print_trainable_parameters()

# ================
# Format Dataset
# ================
def preprocess_fn(examples):
    model_inputs = tokenizer(examples["src"], text_target=examples["tgt"] ,max_length=512, truncation=True)
    return model_inputs


def prepare_data(df):
    prompt_set = []
    for _, row in df.iterrows():
            prompt_set.append({'src': row['ENG'], 'tgt': row['THA']})
            
    prompt_set = Dataset.from_list(prompt_set)
    prompt_set = prompt_set.map(preprocess_fn, batched=True, remove_columns=['src', 'tgt'])

    return prompt_set

train_set = prepare_data(train_df)
test_set = prepare_data(test_df)
print("Formatting dataset done!")

# ================
# Eval Function
# ================
chencherry = SmoothingFunction()
def bleu_score(ref, hyp):
    hyp_tok = word_tokenize(hyp, engine="attacut")
    ref_tok = word_tokenize(ref, engine="attacut")
        
    # Sentence-level BLEU with smoothing; returns 0..1
    bleu = sentence_bleu(
        [ref_tok], hyp_tok,
        smoothing_function=chencherry.method3
    )
    return bleu 
    
def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds

    # Replace ignored label id (-100) with pad for decoding
    pad_id = tokenizer.pad_token_id
    labels = [[(tid if tid != -100 else pad_id) for tid in seq] for seq in labels]

    # Decode to text
    pred_texts  = tokenizer.batch_decode(preds,   skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels,  skip_special_tokens=True)
    
    scores = [bleu_score(ref, hyp) for ref, hyp in zip(label_texts, pred_texts)]
    bleu_avg = float(np.mean(scores)) if scores else 0.0
    return {"bleu_nltk_attacut": bleu_avg}

# ================
# Training Setup
# ================
def get_lang_ids(tokenizer, tgt_lang):
    # NLLB tokenizer exposes lang_code_to_id
    forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)
    if forced_bos is None and hasattr(tokenizer, "lang_code_to_id"):
        forced_bos = tokenizer.lang_code_to_id[tgt_lang]
    return forced_bos


forced_bos_token_id = get_lang_ids(tokenizer, TGT_LANG)
model.config.forced_bos_token_id = forced_bos_token_id
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest")

gen_config = GenerationConfig(
    max_length=512,          # Maximum length of the generated text
    do_sample=True,          # Whether to use sampling (e.g., top-p, top-k)
    temperature=0.2,         # Controls randomness
    num_beams=3              # Use beam search for more deterministic output
)

# ================
# Seq2Seq Training
# ================
print("\nStart Seq2Seq training\n")
args = Seq2SeqTrainingArguments(
    output_dir=f"./nllb_checkpoints2",
    logging_dir=f"./nllb_logs",
    per_device_train_batch_size=16,         # higher with QLoRA
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    optim="adamw_torch",
    learning_rate=2e-4,                    # LoRA uses larger LR
    lr_scheduler_type="cosine",
    num_train_epochs=5,
    weight_decay=0.05,
    warmup_ratio=0.03,
    logging_steps=200,
    eval_strategy="steps",
    eval_steps=200,
    save_steps=500,
    save_total_limit=2,
    predict_with_generate=True,
    generation_config=gen_config,
    # deepspeed=f"./ds_config.json",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},  # avoids reentrant edge cases
    run_name="nllb-en2th",
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=train_set,
    eval_dataset=test_set,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=lambda p: compute_metrics(p, tokenizer),
)

trainer_stats = trainer.train()
