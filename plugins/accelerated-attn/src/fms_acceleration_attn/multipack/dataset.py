
# Third Party
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch

class TokenDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.data = dataset
        self.lengths = np.array(
            self.data.map(lambda x: {"len": len(x["input_ids"])}, num_proc=16)["len"]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[int(idx)]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        labels = torch.tensor(item["labels"], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def get_lengths(self):
        return self.lengths