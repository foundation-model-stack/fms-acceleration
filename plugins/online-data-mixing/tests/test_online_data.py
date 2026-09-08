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
from datasets import Dataset
from torch.utils.data import IterableDataset
from transformers import AutoModelForCausalLM

# pylint: disable=import-error
import pytest
import torch

# First Party
from fms_acceleration_odm import OnlineMixingDataset, Reward


class SampleDataset(IterableDataset):
    def __init__(self, seq_length, vocab_size):
        self.seq_length = seq_length
        self.vocab_size = vocab_size

    def __len__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        input_ids = torch.rand(self.seq_length)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(self.seq_length),
            "labels": input_ids,
        }


def get_dataset(seq_len, vocab_size):
    return SampleDataset(seq_length=seq_len, vocab_size=vocab_size)


PARAMETERS = [
    (
        {"data_1": 1, "data_2": 100, "data_3": 2},
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

    train_data_dict = {
        "data_1": get_dataset(seq_len=seq_length, vocab_size=vocab_size),
        "data_2": get_dataset(seq_len=seq_length, vocab_size=vocab_size),
        "data_3": get_dataset(seq_len=seq_length, vocab_size=vocab_size),
    }
    collators_dict = {"data_1": None, "data_2": None, "data_3": None}
    dataset = OnlineMixingDataset(
        train_data_dict,
        collators_dict,
        train_data_dict,
        collators_dict,
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


def _hf_dataset(seq_length, vocab_size, num_rows=4):
    return Dataset.from_dict(
        {
            "input_ids": [[i % vocab_size] * seq_length for i in range(num_rows)],
            "attention_mask": [[1] * seq_length for _ in range(num_rows)],
            "labels": [[i % vocab_size] * seq_length for i in range(num_rows)],
        }
    ).with_format("torch")


def test_online_data_learnability_requires_templated_eval_dataset():
    seq_length = 6
    vocab_size = 50
    train_data_dict = {
        "data_1": _hf_dataset(seq_length, vocab_size),
        "data_2": _hf_dataset(seq_length, vocab_size),
    }
    collators_dict = {"data_1": None, "data_2": None}
    with pytest.raises(ValueError):
        OnlineMixingDataset(
            train_data_dict,
            collators_dict,
            train_data_dict,
            collators_dict,
            output_dir="odm",
            reward_type=Reward.LEARNABILITY,
        )


@pytest.mark.parametrize("reward_type", [Reward.LEARNABILITY, Reward.COMBINED])
def test_online_data_update_sampling_weights_with_templated_eval_dataset(reward_type):
    seq_length = 6
    vocab_size = 50
    train_data_dict = {
        "data_1": _hf_dataset(seq_length, vocab_size),
        "data_2": _hf_dataset(seq_length, vocab_size),
    }
    collators_dict = {"data_1": None, "data_2": None}
    templated_data_dict = {
        "data_1": _hf_dataset(seq_length + 4, vocab_size),
        "data_2": _hf_dataset(seq_length + 4, vocab_size),
    }

    dataset = OnlineMixingDataset(
        train_data_dict,
        collators_dict,
        train_data_dict,
        collators_dict,
        output_dir="odm",
        reward_type=reward_type,
        eval_batch_size=2,
        templated_eval_dataset_dict=templated_data_dict,
        templated_eval_collators_dict=collators_dict,
        beta=1.0,
    )

    class DummyState:
        global_step = 0
        max_steps = 10
        log_history = []

    class CPUAccelerator:
        device = torch.device("cpu")

        def prepare(self, x):
            return x

        def reduce(self, x, reduction):  # pylint: disable=unused-argument
            return x

    model = AutoModelForCausalLM.from_pretrained("Maykeye/TinyLLama-v0")
    # update_sampling_weights() moves eval batches to accelerator.device (or
    # torch.device(0), i.e. cuda:0, if no accelerator is given). Pass a
    # single-process CPU stub so this test doesn't require a GPU.
    dataset.update_sampling_weights(model, accelerator=CPUAccelerator(), state=DummyState())

    assert dataset.log["rewards"], "expected rewards to be logged after update"
    counts = list(dataset.log["count"])
    assert all(
        count > 0 for count in counts
    ), "expected every category to accumulate a nonzero count"
