import os
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "llm_training_dataset_100.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
MODEL_DIR = os.path.join(BASE_DIR, "llm_trained_model")
LOG_DIR = os.path.join(BASE_DIR, "logs")

# Step 1: Load JSON file safely with pandas (handles mixed response types)
df = pd.read_json(DATASET_PATH)

# ✅ Normalize response column: convert dicts/lists to JSON strings
df["response"] = df["response"].apply(lambda x: json.dumps(x, ensure_ascii=False))

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Step 2: Choose a base model (small for demo)
model_name = "distilgpt2"  # small, fast GPT-like model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# GPT2 doesn’t have a pad token — fix this
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Step 3: Preprocess data
def format_example(example):
    # Instruction + Input combined
    if example.get("input"):
        prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nResponse:"
    else:
        prompt = f"Instruction: {example['instruction']}\nResponse:"
    return {"prompt": prompt, "label": example["response"]}

formatted = dataset.map(format_example)

# Tokenize
def tokenize(batch):
    return tokenizer(
        batch["prompt"],
        text_target=batch["label"],
        truncation=True,
        padding="max_length",
        max_length=256
    )

tokenized = formatted.map(tokenize, batched=True, remove_columns=formatted.column_names)

# Step 4: Load model
model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))  # update embeddings if new tokens added

# Step 5: Training setup (compatible with older transformers)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    num_train_epochs=1,   # keep small for demo
    weight_decay=0.01,
    logging_dir=LOG_DIR,
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
)

# Step 6: Train model
trainer.train()

# Step 7: Save fine-tuned model
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print(f"✅ Training complete. Model saved in {MODEL_DIR}")
