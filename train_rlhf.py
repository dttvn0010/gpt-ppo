import torch
from datasets import load_dataset
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import pipeline
from tqdm import tqdm

dataset = load_dataset("hh-rlhf", split="train")
MAX_SEQ_LEN = 512
config = PPOConfig(
    model_name="gpt2",
    learning_rate=2.0e-5,
    batch_size=8,
    mini_batch_size=8
)

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token
reward_model = pipeline("text-classification", model="./distilbert-tuned", truncation=True, max_length=MAX_SEQ_LEN)


def tokenize(sample):
    text = sample['text']
    items = text.split('\n\nAssistant:')
    query = items[0].replace('\n\nHuman:','').strip()
    sample["input_ids"] = tokenizer.encode(query, truncation=True, max_length=MAX_SEQ_LEN)
    del sample['text']
    del sample['label']
    return sample


dataset = dataset.map(tokenize)
dataset.set_format(type="torch")
data_collator = DataCollatorWithPadding(tokenizer)

ppo_trainer = PPOTrainer(
    model=model,
    config=config,
    dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": MAX_SEQ_LEN
}

for epoch in tqdm(range(10), "epoch: "):
    for batch in tqdm(ppo_trainer.dataloader):
        print(batch['input_ids'].shape)
        input_ids = list(batch["input_ids"])
        attention_mask = list(batch['attention_mask'])

        batch['query'] = [tokenizer.decode(x).replace(tokenizer.eos_token, '') for x in input_ids]
        response_tensors = ppo_trainer.generate(input_ids, **generation_kwargs)
        batch['response'] = [tokenizer.decode(r.squeeze()).replace(tokenizer.eos_token, '') for r in response_tensors]

        texts = [f'\n\nHuman:{q}\n\nAssistant:{r}' for q, r in zip(batch['query'], batch['response'])]
        pipe_outputs = reward_model(texts)
        rewards = [torch.tensor(output["score"]) for output in pipe_outputs]

        stats = ppo_trainer.step(input_ids, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)


ppo_trainer.save_pretrained("gpt-2-tuned")