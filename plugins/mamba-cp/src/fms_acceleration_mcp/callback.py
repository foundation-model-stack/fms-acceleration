# fms-hf-tuning patch
# Standard
from logging import getLogger
import os

# Third Party
from transformers import TrainerCallback
import torch

logger = getLogger(__name__)


class DataloaderSavingCallback(TrainerCallback):
    def __init__(self, accelerator):
        super().__init__()
        self.accelerator = accelerator

    def on_save(self, args, state, control, **kwargs):
        if not self.accelerator.is_main_process:
            return
        # Third Party
        # pylint: disable=import-outside-toplevel
        from torchdata.stateful_dataloader import StatefulDataLoader

        checkpoint_path = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        # It is assumed that one of the datasets would be stateful
        # if stateful then it would be training dataset
        for i, _ in enumerate(self.accelerator._dataloaders):
            if isinstance(
                self.accelerator._dataloaders[i].base_dataloader, StatefulDataLoader
            ):
                torch.save(
                    self.accelerator._dataloaders[i].state_dict(),
                    os.path.join(checkpoint_path, "odm_dl_state_dict.bin"),
                )
                break
