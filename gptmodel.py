import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
import wandb
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForCausalLM.from_pretrained(model_name)
dataset = load_dataset("json", data_files="custom_data.jsonl", split="train")
def tokenize_function(examples):
    combined_texts = [f"{question}\n{answer}" for question, answer in zip(examples["question"], examples["answer"])]
    tokenized = tokenizer(combined_texts, truncation=True, max_length=512, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized
tokenized_dataset = dataset.map(tokenize_function, batched=True)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset
)
trainer.train()
model_save_path = "/content/drive/My Drive/gpt2_apple_disease_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model saved to {model_save_path}")
