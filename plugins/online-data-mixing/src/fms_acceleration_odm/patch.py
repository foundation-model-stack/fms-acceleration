# fms-hf-tuning patch
# Standard
from logging import getLogger
import os

logger = getLogger(__name__)

def patch_hf_trainer_evaluate():
    from transformers import Trainer
    # Third Party
    # pylint: disable=import-outside-toplevel
    from fms_acceleration.model_patcher import patch_target_module

    Trainer._evaluate = _evaluate
    Trainer._get_dataloader = _get_dataloader
    Trainer.get_train_dataloader = get_train_dataloader
    patch_target_module("transformers.trainer.Trainer", Trainer)
    patch_target_module("transformers.trainer.skip_first_batches", skip_first_batches)


def _evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
    # Standard
    # pylint: disable=import-outside-toplevel
    import time

    # Third Party
    # pylint: disable=import-outside-toplevel
    import torch
    from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

    metrics = None
    if (
        self.model.ta_eval_steps
        and self.state.global_step % self.model.ta_eval_steps == 0
    ):
        metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
        self._report_to_hp_search(trial, self.state.global_step, metrics)

        # Run delayed LR scheduler now that metrics are populated
        if (
            isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            and not skip_scheduler
        ):
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            try:
                self.lr_scheduler.step(metrics[metric_to_check])
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is "
                    f"set to '{metric_to_check}', "
                    f"which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}."
                    f"Please ensure that the `compute_metrics` function returns a "
                    f"dictionary that includes '{metric_to_check}' or "
                    f"consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc

    if self.state.global_step % self.model.ta_update_interval == 0:
        # prepare model
        # code taken from def evaluation_loop from HF
        model = self._wrap_model(self.model, training=False)
        args = self.args
        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                or (
                    self.is_fsdp_enabled
                    and self.accelerator.mixed_precision != "fp8"
                    and not self.args.torch_compile
                )
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model,
            # whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        if hasattr(model, "eval") and callable(model.eval):
            model.eval()
        if hasattr(self.optimizer, "eval") and callable(self.optimizer.eval):
            self.optimizer.eval()
        # Do this before wrapping.
        if args.past_index >= 0:
            self._past = None
        # prepare dataloader
        self.train_dataset.update_sampling_weights(model, self.accelerator, self.state)
        
        # save the dataloader
        if self.control.should_save:
            logger.info("dataloader is saved")
            output_dir = os.path.join(self.args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.accelerator._dataloaders[0].state_dict(), os.path.join(output_dir, "odm_dl_state_dict.bin"))
    return metrics

# code taken from transformers and modified
def _get_dataloader(
    self,
    dataset,
    description,
    batch_size,
    sampler_fn = None,
    is_training = False,
    dataloader_key = None,
):
    """Create a [`~torch.utils.data.DataLoader`] from the given dataset."""
    import torch
    from torch.utils.data import DataLoader
    from torchdata.stateful_dataloader import StatefulDataLoader
    from transformers import is_datasets_available
    from functools import partial
    from transformers.trainer_utils import seed_worker
    if is_datasets_available():
        import datasets
    
    data_collator = self.data_collator
    if is_datasets_available() and isinstance(dataset, datasets.Dataset):
        dataset = self._remove_unused_columns(dataset, description=description)
    else:
        data_collator = self._get_collator_with_removed_columns(self.data_collator, description=description)

    dataloader_params = {
        "batch_size": batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    if not isinstance(dataset, torch.utils.data.IterableDataset):
        if sampler_fn is not None:
            dataloader_params["sampler"] = sampler_fn(dataset)
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        if is_training:
            dataloader_params["worker_init_fn"] = partial(
                seed_worker, num_workers=self.args.dataloader_num_workers, rank=self.args.process_index
            )
    if is_training:
        dataloader = self.accelerator.prepare(StatefulDataLoader(dataset, **dataloader_params))
    else:
        dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

    # Store the prepared dataloader for subsequent evaluations if using persistent workers.
    if dataloader_key is not None and self.args.dataloader_persistent_workers:
        if hasattr(self, "_eval_dataloaders"):
            self._eval_dataloaders[dataloader_key] = dataloader
        else:
            self._eval_dataloaders = {dataloader_key: dataloader}

    return dataloader

def get_train_dataloader(self):
    from transformers.trainer_utils import get_last_checkpoint, seed_worker
    import torch
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    dataloader = self._get_dataloader(
        dataset=self.train_dataset,
        description="Training",
        batch_size=self._train_batch_size,
        sampler_fn=self._get_train_sampler,
        is_training=True,
    )
    resume_from_checkpoint = self.model.resume_from_checkpoint
    if resume_from_checkpoint:
        # code taken from transformers and modified
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")
        self.model.resume_from_checkpoint = resume_from_checkpoint

        # load state to the dataloader
        dataloader_state_dict_name = "odm_dl_state_dict.bin"
        output_dataloader_state_dict_file = os.path.join(
            resume_from_checkpoint, dataloader_state_dict_name
        )
        self.accelerator._dataloaders[0].load_state_dict(
            torch.load(output_dataloader_state_dict_file)
        )

def skip_first_batches(dataloader, num_batches=0):
    return dataloader
