# samples entropy

# Standard
from enum import StrEnum, auto
from logging import getLogger
from typing import Dict
import math

# Third Party
from transformers import PreTrainedModel
import torch
import torch.nn.functional as F

logger = getLogger(__name__)


class Reward(StrEnum):
    ENTROPY = auto()
    ENTROPY3_VARENT1 = auto()
    ENTROPY_LAST_TOKEN = auto()
    TRAIN_LOSS = auto()
    VALIDATION_LOSS = auto()
    GRADNORM = auto()
    LEARNABILITY = auto()
    VELOCITY = auto()
    COMBINED = auto()


# TODO:
# 1. class implementation to store individual reward states
# 2. sub arguments for rewards for user to pass
# 3. use reward state and expected inputs to control
# flow of information (iteration over eval batch vs not doing so)

TRAIN_LOSS_DATA = {"buffer": []}

EVAL_LOSS_DATA = {"buffer": []}

GRADNORM_DATA = {"buffer": []}

VELOCITY_DATA = {"buffer": []}


def compute_reward(
    model: PreTrainedModel,
    batch: Dict[str, torch.Tensor],
    vocab_size: int,
    reward_type: Reward,
    train_loss_history=None,
    eval_loss_history=None,
    gradnorm_history=None,
    last_sampled_category=None,
    total_categories=None,
    current_category=None,
    zero_shot_batch=None,
    few_shot_batch=None,
    train_step=None,
    total_steps=None,
    beta: float = 1.0,
) -> float:
    """
    Compute rewards based on the provided reward_type.
    You should be extending this function for new rewards.

    Supported rewards:

        Entropy related rewards: ENTROPY, ENTROPY3_VARENT1 & ENTROPY_LAST_TOKEN
        Calculates the entropy and variance of entropy of every sequence in the batch.
        For every sequence,
            1. The token level metrics are computed
            2. The metrics are averaged per sequence after applying the attention mask

        Train loss reward: TRAIN_LOSS
        We maintain a buffer over all the categories capturing their train loss when sampled.
        Higher train loss should reward more to choose from that category to bring this loss down.

        Validation loss reward: VALIDATION_LOSS
        Similar to TRAIN_LOSS reward here we use individual category validation loss instead.

        Grad norm reward: GRADNORM
        Similar to TRAIN_LOSS reward here we use overall gradnorm. However, Higher grad norm
        categories should be less priortized.

        Learnability reward: LEARNABILITY
        Compares the model's loss on a zero-shot batch against a few-shot (templated) batch
        for the same category: reward = 1 - loss_few_shot / loss_zero_shot. Higher values mean
        the category benefits more from in-context examples, i.e. the model has more to learn
        from that category.

        Velocity reward: VELOCITY
        Tracks how quickly a category's (eval) loss is dropping between consecutive reward
        computations: reward = 1 - current_loss / previous_loss. Higher values mean the category
        is still improving quickly and should keep being sampled.

        Combined reward: COMBINED
        Exponentially-decayed blend of LEARNABILITY and VELOCITY:
            R(t) = alpha(t) * learnability + (1 - alpha(t)) * velocity
            alpha(t) = exp(-beta * train_step / total_steps)
        Early in training the blend favors LEARNABILITY (which category teaches the model the
        most relative to its current baseline); later it favors VELOCITY (which category is
        still yielding gains in absolute loss).

    Args:
        model (PreTrainedModel): HF Model object
        batch (torch.Tensor): Batch of samples (input_ids, labels, attention_mask)
        vocab_size (int): Maximum vocab size of the model used by ENTROPY rewards
        reward_type (Reward): Type of the reward
        train_loss_history: list of dicts each holding information on the training loss
        eval_loss_history: list of dicts each holding information on the eval loss
        gradnorm_history: list of dicts each holding information on the grad_norm
        last_sampled_category: index of the last sampled category
        total_categories: total number of categories
        current_category: currently being reward computed category
        zero_shot_batch: batch of zero-shot samples, used by LEARNABILITY/COMBINED
        few_shot_batch: batch of few-shot (templated) samples, used by LEARNABILITY/COMBINED
        train_step: current training step, used by COMBINED to compute alpha(t)
        total_steps: total number of training steps, used by COMBINED to compute alpha(t)
        beta: decay hyper-parameter for COMBINED's alpha(t). Defaults to 1.0.
    Returns:
        float
    """
    if reward_type.startswith(Reward.ENTROPY):
        with torch.inference_mode():
            outputs = model(**batch)
            shift_logits = outputs.logits[:, :-1, :]

            log_probs = F.log_softmax(shift_logits, dim=-1)
            probs = torch.exp(log_probs)

            entropy = -torch.sum(probs * log_probs, dim=-1)
            sum_p_log_sq = torch.sum(probs * (log_probs**2), dim=-1)
            varentropy = sum_p_log_sq - (entropy**2)

            entropy_last_token = entropy[:, -1]

            mask = batch["attention_mask"][:, 1:]

            entropy = (entropy * mask).sum(dim=-1) / mask.sum(dim=-1)
            varentropy = (varentropy * mask).sum(dim=-1) / mask.sum(dim=-1)

        max_entropy = torch.log(
            torch.tensor(vocab_size, dtype=entropy.dtype, device=entropy.device)
        )

        entropy = (entropy / max_entropy).clamp(0.0, 1.0)
        varentropy = (varentropy / max_entropy**2).clamp(0.0, 1.0)
        entropy_last_token = (entropy_last_token / max_entropy).clamp(0.0, 1.0)
        if reward_type == Reward.ENTROPY:
            return entropy.sum().item()
        if reward_type == Reward.ENTROPY3_VARENT1:
            return 0.75 * entropy.sum().item() + 0.25 * varentropy.sum().item()
        if reward_type == Reward.ENTROPY_LAST_TOKEN:
            return entropy_last_token.sum().item()
    if reward_type == Reward.TRAIN_LOSS:
        if not train_loss_history:
            raise ValueError("train_loss_history cannot be a empty list or None")
        if not TRAIN_LOSS_DATA["buffer"]:
            TRAIN_LOSS_DATA["buffer"] = [1e-100] * total_categories
        TRAIN_LOSS_DATA["buffer"][last_sampled_category] = train_loss_history[-1][
            "loss"
        ]
        return TRAIN_LOSS_DATA["buffer"][current_category]
    if reward_type == Reward.VALIDATION_LOSS:
        if not eval_loss_history:
            raise ValueError(
                "eval_loss_history cannot be a empty list or None."
                "Make sure you are using eval_strategy and eval_steps"
                "allowing atleast 1 evaluation before reward computation."
            )
        if not EVAL_LOSS_DATA["buffer"]:
            EVAL_LOSS_DATA["buffer"] = [1e-100] * total_categories
        EVAL_LOSS_DATA["buffer"][current_category] = eval_loss_history[-1]["loss"]
        return EVAL_LOSS_DATA["buffer"][current_category]
    if reward_type == Reward.GRADNORM:
        if not gradnorm_history:
            raise ValueError(
                "gradnorm_history cannot be a empty list or None."
                "Make sure grad norm is made available."
            )
        if not GRADNORM_DATA["buffer"]:
            GRADNORM_DATA["buffer"] = [1e-100] * total_categories
        GRADNORM_DATA["buffer"][last_sampled_category] = 1 / (
            gradnorm_history[-1]["grad_norm"] + 0.0001
        )
        return GRADNORM_DATA["buffer"][current_category]
    if reward_type == Reward.LEARNABILITY:
        return _compute_learnability_reward(model, zero_shot_batch, few_shot_batch)
    if reward_type == Reward.VELOCITY:
        return _compute_velocity_reward(model, batch, current_category, total_categories)
    if reward_type == Reward.COMBINED:
        learn_r = _compute_learnability_reward(model, zero_shot_batch, few_shot_batch)
        vel_r = _compute_velocity_reward(model, batch, current_category, total_categories)
        if not total_steps:
            logger.warning(
                "COMBINED reward received an empty total_steps; falling back to "
                "alpha=1.0 (pure LEARNABILITY) for this call."
            )
            alpha = 1.0
        else:
            alpha = math.exp(-beta * train_step / total_steps)
        return alpha * learn_r + (1.0 - alpha) * vel_r
    raise TypeError(f"Reward {reward_type} not supported")


def _compute_learnability_reward(model, zero_shot_batch, few_shot_batch) -> float:
    if zero_shot_batch is None or few_shot_batch is None:
        raise ValueError(
            "zero_shot_batch and few_shot_batch cannot be None for LEARNABILITY/"
            "COMBINED rewards."
        )
    with torch.inference_mode():
        loss_zero_shot = model(**zero_shot_batch).loss.item()
        loss_few_shot = model(**few_shot_batch).loss.item()
    if loss_zero_shot == 0:
        return 0.0
    return 1.0 - loss_few_shot / loss_zero_shot


def _compute_velocity_reward(model, batch, current_category, total_categories) -> float:
    if batch is None:
        raise ValueError("batch cannot be None for VELOCITY/COMBINED rewards.")
    with torch.inference_mode():
        current_loss = model(**batch).loss.item()
    if not VELOCITY_DATA["buffer"]:
        VELOCITY_DATA["buffer"] = [None] * total_categories
    previous_loss = VELOCITY_DATA["buffer"][current_category]
    VELOCITY_DATA["buffer"][current_category] = current_loss
    if not previous_loss:
        return 0.0
    return 1.0 - current_loss / previous_loss
