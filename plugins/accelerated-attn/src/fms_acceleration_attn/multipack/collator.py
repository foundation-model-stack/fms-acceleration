import numpy as np
import torch

# for padding free HF injection 
# - all data put into a single batch with a concat sequence
# - have provisions for per_token_loss, which will return the 
#   number of loss tokens
def build_hugginface_padding_free_collator(
    MAX_BATCH_LEN: int, 
    per_token_loss: bool = False,
):

    def collate_fn(batch):
        lens = np.array([len(item["input_ids"]) for item in batch])

        cumsum_lens = np.cumsum(lens)
        valid_up_to = int((cumsum_lens < MAX_BATCH_LEN).sum())

        batch = batch[:valid_up_to]
        position_ids = []
        for idx in range(len(batch)):
            position_ids += list(range(len(batch[idx]['input_ids'])))
            batch[idx]['labels'][0] = -100
        position_ids = torch.tensor(position_ids, dtype=torch.long).unsqueeze(0)
        input_ids = torch.cat([x['input_ids'] for x in batch]).unsqueeze(0)
        labels = torch.cat([x['labels'] for x in batch]).unsqueeze(0)
        data = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
        }
        
        if per_token_loss:

            num_loss_counted_tokens = sum(
                [(x["labels"] != -100).sum().item() for x in batch]
            )

            data["num_loss_counted_tokens"] = num_loss_counted_tokens

        return data

    return collate_fn