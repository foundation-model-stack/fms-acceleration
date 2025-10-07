# fms-hf-tuning patch
# Standard
from logging import getLogger
import os
import torch
from transformers import TrainerCallback

logger = getLogger(__name__)

class DataloaderSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(
            args.output_dir, f"checkpoint-{state.global_step}"
        )
        print(kwargs["train_dataloader"])
        # save the dataloader
        logger.info("dataloader is saved")
        torch.save(
            kwargs["train_dataloader"].state_dict(),
            os.path.join(checkpoint_path, "odm_dl_state_dict.bin"),
        )
