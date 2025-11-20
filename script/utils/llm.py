from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json
import os

# =====================
# Initialize LLM
# =====================
def init_model(model_path: str, max_seq_length: int = 4096, load_in_4bit: bool = True, full_finetuning: bool = False,
              rank: int = 16, target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        load_in_8bit = False,
        full_finetuning = False,
        device_map="balanced",
    )
    if not full_finetuning:
        model = FastLanguageModel.get_peft_model(
            model, 
            r = rank,           
            lora_alpha = rank,  
            lora_dropout = 0,
            target_modules = target_modules,
            bias = "none",
            random_state = 3407,
        )
    return (model, tokenizer)


# =====================
# Plot loss
# =====================
def plot_history(output_dir):
    """visualize loss history after done fine-tuning"""
    stats = glob.glob(f"{output_dir}/checkpoint*")
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
    plt.savefig(f"{output_dir}/trainer_state.png")
    

# =====================
# Fine-tuning
# =====================
def train_model(model, tokenizer, train_set, test_set,
                output_dir, logging_steps=100, save_steps=250,
                batch=32, epochs=1, lr=2e-4, lr_scheduler="cosine"):    
    args = SFTConfig(
        output_dir=output_dir,
        logging_dir=os.path.join(output_dir, 'logs'),
        overwrite_output_dir=True,
        dataset_text_field = "text",
        per_device_train_batch_size = batch,
        per_device_eval_batch_size= batch,
        gradient_accumulation_steps = 2,
        warmup_ratio = 0.03,
        num_train_epochs = epochs,
        learning_rate = lr,
        logging_steps = logging_steps,
        eval_steps = logging_steps,
        eval_strategy = "steps",
        save_steps = save_steps,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = lr_scheduler,
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
    plot_history(output_dir)
    return train_stats
    