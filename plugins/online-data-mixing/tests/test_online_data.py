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
# pylint: disable=import-error
import pytest
import torch

# First Party
from fms_acceleration_odm import OnlineMixingDataset, Reward

PARAMETERS = [
    (
        [1, 100, 2],
        [[1, 100, 1], [1, 200, 1], [1, 100, 1], [1, 1, 1000], [1, 1, 2000]],
        5,
        [1, 1, 1, 2, 2],
        3,
    )
]


@pytest.mark.parametrize(
    "sampling_weights,rewards,batch_size,expected_arm_idx,total_categories",
    PARAMETERS,
)
def test_online_data_mix_learning(
    sampling_weights, rewards, batch_size, expected_arm_idx, total_categories
):
    batch_size = 100
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
    train_data = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
    eval_data = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }
    dataset = OnlineMixingDataset(
        train_data,
        None,
        eval_data,
        None,
        sampling_weights,
        0.1,
        0.3,
        1,
        batch_size,
        output_dir="odm",
        reward_type=Reward.ENTROPY,
    )
    categories_chosen = []
    for reward in rewards:
        dataset._update_weights([batch_size] * total_categories, reward)
        next(dataset)
        categories_chosen.append(dataset.arm_idx)
    # we check if atleast half of the choices match since this is probabilistic
    # and may fail unit tests randomly
    assert sum(x == y for x, y in zip(categories_chosen, expected_arm_idx)) >= (
        len(expected_arm_idx) / 2
    ), "Not even half of the choices were correct"
