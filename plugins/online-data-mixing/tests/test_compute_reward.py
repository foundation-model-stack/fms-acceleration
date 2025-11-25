# Copyright The FMS HF Tuning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Third Party
from transformers import AutoModelForCausalLM

# pylint: disable=import-error
import pytest
import torch

# First Party
from fms_acceleration_odm import Reward, compute_reward

PARAMETERS = [
    ("Maykeye/TinyLLama-v0", "ENTROPY", None, None, None, 1, 8, 0, 3),
    ("Maykeye/TinyLLama-v0", "ENTROPY3_VARENT1", None, None, None, 1, 8, 0, 2.5),
    ("Maykeye/TinyLLama-v0", "ENTROPY_LAST_TOKEN", None, None, None, 1, 8, 0, 3),
    (
        "Maykeye/TinyLLama-v0",
        "TRAIN_LOSS",
        [
            [{"loss": 3}],
            [{"loss": 3}, {"loss": 4}],
            [{"loss": 3}, {"loss": 4}, {"loss": 2}],
            [{"loss": 3}, {"loss": 4}, {"loss": 2}, {"loss": 1}],
        ],
        None,
        None,
        [0, 0, 1, 1],
        2,
        None,
        [[3, 1e-100], [4, 1e-100], [4, 2], [4, 1]],
    ),
    (
        "Maykeye/TinyLLama-v0",
        "VALIDATION_LOSS",
        None,
        [
            [[{"loss": 3}], [{"loss": 10}]],
            [[{"loss": 3}], [{"loss": 5}]],
            [[{"loss": 4}], [{"loss": 3}]],
            [[{"loss": 2}], [{"loss": 3}]],
        ],
        None,
        [0, 1, 1, 1],
        2,
        None,
        [[3, 10], [3, 5], [4, 3], [2, 3]],
    ),
    (
        "Maykeye/TinyLLama-v0",
        "GRADNORM",
        None,
        None,
        [
            [{"grad_norm": 3}],
            [{"grad_norm": 3}, {"grad_norm": 4}],
            [{"grad_norm": 3}, {"grad_norm": 4}, {"grad_norm": 2}],
            [{"grad_norm": 3}, {"grad_norm": 4}, {"grad_norm": 2}, {"grad_norm": 1}],
        ],
        [0, 1, 1, 0],
        2,
        None,
        [
            [1 / (3 + 0.0001), 1e-100],
            [1 / (3 + 0.0001), 1 / (4 + 0.0001)],
            [1 / (3 + 0.0001), 1 / (2 + 0.0001)],
            [1 / (1 + 0.0001), 1 / (2 + 0.0001)],
        ],
    ),
]


@pytest.mark.parametrize(
    (
        "model,reward_type,train_loss_history,eval_loss_history,gradnorm_history,"
        "last_sampled_category,total_categories,current_category,reward"
    ),
    PARAMETERS,
)
def test_compute_reward(
    model,
    reward_type,
    train_loss_history,
    eval_loss_history,
    gradnorm_history,
    last_sampled_category,
    total_categories,
    current_category,
    reward,
):
    loaded_model = AutoModelForCausalLM.from_pretrained(model)
    reward_type = Reward[reward_type]
    batch_size = 3
    seq_length = 6
    vocab_size = 50
    input_ids = (
        torch.arange(batch_size * seq_length).reshape(batch_size, seq_length)
        % vocab_size
    )
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
    )
    labels = input_ids
    batch = {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
    if reward_type == Reward.ENTROPY:
        reward = compute_reward(
            model=loaded_model,
            batch=batch,
            vocab_size=vocab_size,
            reward_type=reward_type,
            current_category=current_category,
            total_categories=total_categories,
            last_sampled_category=last_sampled_category,
        )
        assert reward == 3, f"entropy {reward} does not match the expected value."
    if reward_type == Reward.ENTROPY3_VARENT1:
        reward = compute_reward(
            model=loaded_model,
            batch=batch,
            vocab_size=vocab_size,
            reward_type=reward_type,
            current_category=current_category,
            total_categories=total_categories,
            last_sampled_category=last_sampled_category,
        )
        assert reward >= 2.5, f"entropy {reward} does not match the expected value."
    if reward_type == Reward.ENTROPY_LAST_TOKEN:
        reward = compute_reward(
            model=loaded_model,
            batch=batch,
            vocab_size=vocab_size,
            reward_type=reward_type,
            current_category=current_category,
            total_categories=total_categories,
            last_sampled_category=last_sampled_category,
        )
        assert reward == 3, f"entropy {reward} does not match the expected value."
    if reward_type == Reward.TRAIN_LOSS:
        for h, cc, r in zip(
            train_loss_history, range(len(last_sampled_category)), reward
        ):
            returned_reward = [1e-100, 1e-100]
            for c in range(total_categories):
                returned_reward[c] = compute_reward(
                    model=loaded_model,
                    batch=batch,
                    vocab_size=vocab_size,
                    reward_type=reward_type,
                    current_category=c,
                    total_categories=total_categories,
                    last_sampled_category=last_sampled_category[cc],
                    train_loss_history=h,
                )
            assert returned_reward == r, f"expected {r} but got {returned_reward}"
    if reward_type == Reward.GRADNORM:
        for h, cc, r in zip(
            gradnorm_history, range(len(last_sampled_category)), reward
        ):
            returned_reward = [1e-100, 1e-100]
            for c in range(total_categories):
                returned_reward[c] = compute_reward(
                    model=loaded_model,
                    batch=batch,
                    vocab_size=vocab_size,
                    reward_type=reward_type,
                    current_category=c,
                    total_categories=total_categories,
                    last_sampled_category=last_sampled_category[cc],
                    gradnorm_history=h,
                )
            assert returned_reward == r, f"expected {r} but got {returned_reward}"
