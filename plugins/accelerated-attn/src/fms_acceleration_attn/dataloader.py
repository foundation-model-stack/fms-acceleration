import numpy as np
import torch
from types import MethodType
from torch.utils.data import DataLoader
from accelerate import Accelerator

def build_hugginface_padding_free_collator(
    MAX_BATCH_LEN: int, 
    per_token_loss: bool = False,
):

    def collate_fn(self, batch):
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

        num_loss_counted_tokens = sum(
            [(x["labels"] != -100).sum().item() for x in batch]
        )

        data = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            # "num_loss_counted_tokens": num_loss_counted_tokens,
        }
        if per_token_loss:
            data["num_loss_counted_tokens"] = num_loss_counted_tokens

        return data

    return collate_fn

def get_multipack_dataloader(
    data_path: str,
    num_bins: int,
    effective_batch_size: int = 3840,
    max_batch_len: int = 60000,
    pad_token_id: int = 0, # NOTE: get from somewhere
    is_padding: bool = False,
    num_workers: int = 8,
):
    # NOTE: this is a bit cheating at the moment
    # tokenizer will be ignored
    from instructlab.training.token_dataset import setup_dataset
    dataset = setup_dataset(data_path)

    # first we need to get the special dataset for it
    # maybe this will bypass the original dataset
    # how to handle validation datasets?

    # guarded import
    from instructlab.training.multipack_sampler import find_packing_max_batch_len_and_grad_accum
    from instructlab.training.token_dataset import setup_dataloader

    packing_max_batch_len, grad_accum = find_packing_max_batch_len_and_grad_accum(
        num_gpus=num_bins,
        avg_sample_len=dataset.get_lengths().mean(),
        effective_batch_size=effective_batch_size,
        max_batch_len_per_gpu=max_batch_len,
        is_padding=is_padding,
        dataset=dataset,
        pad_id=pad_token_id,
        seed=42,
    )

    train_loader = setup_dataloader(
        dataset,
        pad_token_id,
        num_workers=num_workers,
        is_granite=not is_padding,
        max_batch_len=max_batch_len,
        packing_max_batch_len=packing_max_batch_len,
        seed=42,
    )

    return train_loader, grad_accum


def patch_multipack_dataloader(
    accelerator: Accelerator, 
    dataloader: DataLoader,
    format: str = 'huggingface',
    per_token_loss: bool = True,
    max_batch_len: int = 60000,
):

    if format == 'huggingface':
        collate_fn = build_hugginface_padding_free_collator(
            per_token_loss=per_token_loss,
            MAX_BATCH_LEN=max_batch_len
        )
    else:
        raise NotImplementedError

    # change the collator
    dataloader.collate_fn = MethodType(collate_fn, dataloader)

    _old_prepare = accelerator.prepare
    def prepare(self, *args, device_placement=None):
        if len(args) == 1 and isinstance(args[0], DataLoader):
            return dataloader

        return _old_prepare(*args, device_placement=device_placement)

    # FIXME: move this somewhere
    accelerator.even_batches = False
    accelerator.prepare = MethodType(prepare, accelerator)
