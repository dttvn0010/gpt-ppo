import torch
from datasets import load_dataset
from transformers import Trainer, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments

training_config = {
    "num_train_epochs": 2,
    "output_dir": "./distilbert-tuned",
    "per_device_eval_batch_size": 16,
    "per_device_train_batch_size": 16,
    "evaluation_strategy": "epoch"
}

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
data_collator = DataCollatorWithPadding(tokenizer)

def tokenize(examples):
    outputs = tokenizer(examples['text'], truncation=True)
    return outputs

ds = load_dataset("hh-rlhf")
tokenized_ds = ds.map(tokenize, batched=True)
train_dataset = tokenized_ds["train"].shuffle()
test_dataset = tokenized_ds["test"]

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=TrainingArguments(**training_config),
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator
)

trainer.train()
