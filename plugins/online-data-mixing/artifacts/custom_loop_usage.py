# Standard
import json
import os

# Third Party
from accelerate import Accelerator, DataLoaderConfiguration
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
import torch

# First Party
from fms_acceleration_odm import OnlineMixingDataset

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir = "./odm_custom_use"
max_steps = 200
batch_size = 12
log_file = os.path.join(output_dir, "loss.jsonl")

# model
model = AutoModelForCausalLM.from_pretrained(model_name)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# dataset related
def tokenize_fn(examples):
    return tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=128
    )


dataset_dict = {
    "bookcorpus": load_dataset("rojagtap/bookcorpus", split="train[:1%]"),
    "wikitext": load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]"),
}

# tokenization
dataset_dict["bookcorpus"] = dataset_dict["bookcorpus"].map(
    tokenize_fn, batched=True, remove_columns=dataset_dict["bookcorpus"].column_names
)
dataset_dict["wikitext"] = dataset_dict["wikitext"].map(
    tokenize_fn, batched=True, remove_columns=dataset_dict["wikitext"].column_names
)

collator_dict = {
    "bookcorpus": DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    "wikitext": DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
}

# odm related
update_interval = 1000  # every step
dataset = OnlineMixingDataset(
    dataset_dict=dataset_dict,
    collators_dict=collator_dict,
    eval_dataset_dict={},
    eval_collators_dict={},
    output_dir=output_dir,
    reward_type="train_loss",
    sampling_interval=batch_size,
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=None)

# distributed setup
dataloader_config = DataLoaderConfiguration(split_batches=True, dispatch_batches=True)
accelerator = Accelerator(split_batches=True, dataloader_config=dataloader_config)
model, dataloader = accelerator.prepare(model, dataloader)

# training setup
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()

step_idx = 0


class State:
    log_history: list = []


state = State()
# custom training loop
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

print("training completed!")


# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file /workspace/fms-acceleration/scripts/benchmarks/accelerate.yaml --num_processes=2 --main_process_port=29511 custom_loop_usage.py
