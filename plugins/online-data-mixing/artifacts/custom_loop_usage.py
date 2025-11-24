# Run commmand
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file fms-acceleration/scripts/benchmarks/accelerate.yaml
# --num_processes=2 --main_process_port=29511 custom_loop_usage.py

# Standard
import json
import os

# Third Party
from accelerate import Accelerator, DataLoaderConfiguration
from datasets import load_dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import torch
from functools import partial

# First Party
from fms_acceleration_odm import OnlineMixingDataset
from fms_acceleration_odm.odm.reward import Reward

model_name = "ibm-granite/granite-4.0-h-1b"
output_dir = "./odm_custom_use"
max_steps = 125
batch_size = 4
log_file = os.path.join(output_dir, "loss.jsonl")

# odm related
step_idx = 0
update_interval = 1  # every step

# model
model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# dataset related
# If you have a single dataset, you can declare it with a single key, pair.
# ODM will auto categorize the dataset into psuedo categories
# If you have multiple categories of dataset, you can declare it with multiple key, pair, eg:
# dataset_dict = {
#     "alpaca": load_dataset("tatsu-lab/alpaca", split="train[:1%]"),
#     "oasst": load_dataset("hakurei/open-instruct-v1", split="train[:1%]"),
# }

dataset_dict = {
    "alpaca_train": load_dataset("tatsu-lab/alpaca", split="train[90%:]")
}
eval_dict = {
    "alpaca_val": load_dataset("tatsu-lab/alpaca", split="train[:1%]")
}


def format_example(example):
    if "instruction" in example:
        prompt = f"Instruction: {example['instruction']}\nInput: {example.get('input','')}\nOutput: {example['output']}"
    elif "text" in example:
        prompt = example["text"]
    return {"text": prompt}


for name in dataset_dict:
    dataset_dict[name] = dataset_dict[name].map(format_example)

for name in eval_dict:
    eval_dict[name] = eval_dict[name].map(format_example)

dataset_dict = DatasetDict(dataset_dict)    #type: ignore
eval_dict = DatasetDict(eval_dict)          #type: ignore

def collate_fn(batch, tokenizer):
    msgs = [b.pop("text") for b in batch]

    return tokenizer(
        msgs,
        truncation=True,
        padding="max_length",
        max_length=1024,
        return_tensors="pt"
    )

collator_dict = {
    name: partial(collate_fn, tokenizer=tokenizer)
    for name in dataset_dict
}

eval_collator_dict = {
    name: partial(collate_fn, tokenizer=tokenizer)
    for name in eval_dict
}

# dataset preparation
dataset = OnlineMixingDataset(
    dataset_dict=dataset_dict,
    collators_dict=collator_dict,
    eval_dataset_dict=eval_dict,
    eval_collators_dict=eval_collator_dict,
    output_dir=output_dir,
    reward_type=Reward.TRAIN_LOSS,
    sampling_interval=batch_size,
    auto_categorize_config={"input_column": "text"}
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=None)

# distributed setup
dataloader_config = DataLoaderConfiguration(split_batches=True, dispatch_batches=True)
accelerator = Accelerator(dataloader_config=dataloader_config)
model, dataloader = accelerator.prepare(model, dataloader)

# training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


# Trainer state
class State:
    log_history: list = []


state = State()


# custom training loop
model.train()
for step, batch in enumerate(
    tqdm(dataloader, disable=not accelerator.is_local_main_process)
):
    step_idx += 1
    outputs = model(**batch)
    loss = outputs.loss
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
    loss = accelerator.gather(loss).mean()
    if step_idx % 1 == 0:
        if torch.isnan(loss):
            loss = torch.tensor([10])  # nan -> very high loss
        if accelerator.is_main_process:
            print(f"Step {step_idx} ||| Loss: {loss.item():.4f}")
            with open(log_file, "a") as f:
                f.write(json.dumps({"loss": loss.item(), "step": step_idx}) + "\n")
        state.log_history.append({"loss": loss.item(), "step": step_idx})
    if step_idx % update_interval == 0:
        with torch.no_grad():
            model.eval()
            dataloader.dataset.update_sampling_weights(model, accelerator, state)
            model.train()
    if step_idx > max_steps:
        break

print("Training completed!")
