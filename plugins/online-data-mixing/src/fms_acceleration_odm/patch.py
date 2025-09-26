# fms-hf-tuning patch
# Standard
from logging import getLogger

# Third Party
from transformers import Trainer

logger = getLogger(__name__)


def patch_hf_trainer_evaluate():
    # Third Party
    # pylint: disable=import-outside-toplevel
    from fms_acceleration.model_patcher import patch_target_module

    Trainer._evaluate = _evaluate
    patch_target_module("transformers.trainer.Trainer", Trainer)


def _evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
    # Standard
    # pylint: disable=import-outside-toplevel
    import time

    # Third Party
    # pylint: disable=import-outside-toplevel
    import torch

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

    return metrics
