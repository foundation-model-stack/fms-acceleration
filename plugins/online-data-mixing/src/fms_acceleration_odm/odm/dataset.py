# Standard
from logging import getLogger
from typing import List, Optional
import json
import math
import os
import random

# Third Party
from datasets import DatasetDict
from torch.utils.data import DataLoader, IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
import torch

# Local
from .reward import Reward, compute_reward

logger = getLogger(__name__)


# pylint: disable=too-many-instance-attributes
class OnlineMixingDataset(IterableDataset):
    def __init__(
        self,
        dataset_dict: DatasetDict,
        collators_dict: dict,
        eval_dataset_dict: DatasetDict,
        eval_collators_dict: dict,
        sampling_weights: Optional[List[float]] = None,
        gamma: float = 0.1,
        eta: float = 0.3,
        sampling_interval: int = 1,
        eval_batch_size: int = 5,
        output_dir="odm",
        reward_type=Reward.ENTROPY,
    ):
        """Mixes datasets with sampling ratios learnt using
        Multi Armed Bandit (MAB) EXP3 and rewards defined.
        Rewards are defined in the compute_reward() function.

        NOTE: In distributed setting, this dataset should be used to
        sample on the main process and distribute respective batches
        to other worker processes.

        Args:
            dataset_dict (DatasetDict): keys are category names and values are HF datasets.
            collators_dict (dict): collator corresponding to each dataset
            used while constructing torch dataloader.
            eval_dataset_dict (DatasetDict): keys are category names and values are HF
            eval datasets.
            eval_collators_dict (dict): collator corresponding to each dataset
            used while constructing torch dataloader.
            sampling_weights (Optional[List[float]], optional): Initial
            set of sampling weights to start with. Defaults to equal weightage.
            gamma (float, optional): MAB hyperparameter. Defaults to 0.1.
            eta (float, optional): MAB hyperparameter. Defaults to 0.3.
            sampling_interval (int, optional): sample category at every n samples.
            Defaults to every sample.
            eval_batch_size (int, optional): eval batch size. Defaults to 5.
            output_dir (str, optional): output dir to store logs. Defaults to "odm".
            reward_type (_type_, optional): type of reward to use, more details can
            be found in compute_reward function. Defaults to Reward.ENTROPY.
        """
        logger.info(
            """Values set to OnlineMixingDataset
                    dataset_dict:       {dataset_dict}
                    collators_dict:     {collators_dict}
                    eval_dataset_dict:  {eval_dataset_dict}
                    eval_collators_dict:{eval_collators_dict}
                    sampling_weights:   {sampling_weights}
                    gamma:              {gamma}
                    eta:                {eta}
                    sampling_interval:  {sampling_interval}
                    eval_batch_size:    {eval_batch_size}
                    output_dir:         {output_dir}
                    reward_type:        {reward_type}
                    """.format(
                dataset_dict=dataset_dict,
                collators_dict=collators_dict,
                eval_dataset_dict=eval_dataset_dict,
                eval_collators_dict=eval_collators_dict,
                sampling_weights=sampling_weights,
                gamma=gamma,
                eta=eta,
                sampling_interval=sampling_interval,
                eval_batch_size=eval_batch_size,
                output_dir=output_dir,
                reward_type=reward_type,
            )
        )

        # gamma and eta are MAB hyper-parameters
        self.gamma = gamma
        self.eta = eta
        self.sampling_interval = sampling_interval
        self.collators_dict = collators_dict
        self.eval_collators_dict = eval_collators_dict
        self.eval_dataset_dict = eval_dataset_dict
        self.eval_dataset_dict_dl = {}
        # iterators of the dataloaders
        self.train_dataset_dict_dl = {}
        # to reset iterators holding references to the dataloaer
        self.train_dataset_dict_dl_org = {}
        self.dataset_dict = dataset_dict
        # prepare torch dataloaders for each of the dataset.
        for k, _ in self.dataset_dict.items():
            self.train_dataset_dict_dl_org[k] = StatefulDataLoader(
                self.dataset_dict[k],
                1,
                shuffle=False,
                num_workers=0,
                collate_fn=collators_dict[k] if collators_dict else None,
            )
            self.train_dataset_dict_dl[k] = iter(self.train_dataset_dict_dl_org[k])
        self.eval_batch_size = eval_batch_size
        self.category_list = sorted(self.train_dataset_dict_dl.keys())
        self.id2cat = dict(enumerate(self.category_list))
        self.cat2id = {c: i for i, c in enumerate(self.category_list)}
        self.total_categories = len(self.category_list)

        # If not starting weights given, then all arms (categories)
        # are equally important. Weights based on the size of the datasets
        # and other such heuristics should be computed outside and passed
        # through sampling_weights while initializing this class.
        if sampling_weights is None:
            sampling_weights = [1] * self.total_categories
        self.sampling_weights = torch.tensor(sampling_weights, dtype=torch.float64)
        self.sampling_ratio = []
        self._update_sampling_ratio(self.sampling_weights)

        # curr_cat_count is current sample count per category
        self.curr_cat_count = [0] * self.total_categories

        # produced is total samples sampled so far
        self.produced = 0

        # currently active category (arm)
        self.arm_idx = 0

        # should be one of Reward
        self.reward_type = reward_type
        if isinstance(self.reward_type, str):
            self.reward_type = self.reward_type.upper()
            self.reward_type = Reward[self.reward_type]
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.log_file_path = os.path.join(self.output_dir, "odm.jsonl")
        logger.info(
            "Logs for online data mixing to be stored at {log_file_path}".format(
                log_file_path=self.log_file_path
            )
        )
        self.log = {
            "samples_produced_so_far": 0,
            "sampling_interval": self.sampling_interval,
            "total_categories": self.total_categories,
            "current_sampling_weights": self.sampling_weights.tolist(),
            "current_sampling_ratio": self.sampling_ratio,
            "arm_idx": self.arm_idx,
            "category_level_counts_so_far": self.curr_cat_count,
            "rewards": [0] * self.total_categories,
            "count": 0,
            "action": "",  # one of sample or update
        }

    def log_to_file(self, data: dict):
        """helper function to log the state to the file

        Args:
            data (dict): log state updates
        """
        self.log.update(data)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(self.log) + "\n")

    def __iter__(self):
        self.produced = 0
        return self

    def __next__(self):
        if self.produced % self.sampling_interval == 0:
            self.arm_idx = random.choices(
                range(self.total_categories), weights=self.sampling_ratio, k=1
            )[0]
        sample = None
        try:
            sample = next(self.train_dataset_dict_dl[self.id2cat[self.arm_idx]])
        except StopIteration:
            logger.info(
                "{id} dataset exhausted so the iterator is reset.".format(
                    id=self.id2cat[self.arm_idx]
                )
            )
            self.train_dataset_dict_dl[self.id2cat[self.arm_idx]] = iter(
                self.train_dataset_dict_dl_org[self.id2cat[self.arm_idx]]
            )
            sample = next(self.train_dataset_dict_dl[self.id2cat[self.arm_idx]])

        self.curr_cat_count[self.arm_idx] += 1
        self.produced += 1

        # dataloader returns a batch of 1 sample
        # next should return single sample rather a batch
        if isinstance(sample, torch.Tensor):
            # (edge case) when no collators are passed
            sample = {
                "input_ids": sample[0],
                "attention_mask": torch.ones_like(sample[0]),
                "labels": sample[0],
            }
        else:
            sample = {
                "input_ids": sample["input_ids"][0],
                "attention_mask": (
                    sample["attention_mask"][0]
                    if "attention_mask" in sample
                    else torch.ones_like(sample["input_ids"][0])
                ),
                "labels": (
                    sample["labels"][0]
                    if "labels" in sample
                    else sample["input_ids"][0]
                ),
            }

        self.log_to_file(
            {
                "arm_idx": self.arm_idx,
                "samples_produced_so_far": self.produced,
                "category_level_counts_so_far": self.curr_cat_count,
                "action": "sample",
            }
        )
        return sample

    def load_state_dict(self, state_dict):
        print(state_dict)
        torch.set_rng_state(state_dict["rng"])
        train_dataset_dict_dl_sd = state_dict.pop("train_dataset_dict_dl_sd")
        random.setstate(state_dict.pop("random_state"))
        self.__dict__.update(state_dict)
        self.reward_type = Reward[state_dict["reward_type"].upper()]
        for k, _ in train_dataset_dict_dl_sd.items():
            self.train_dataset_dict_dl[k].load_state_dict(train_dataset_dict_dl_sd[k])

    def state_dict(self):
        return {
            "rng": torch.get_rng_state(),
            "gamma": self.gamma,
            "eta": self.eta,
            "sampling_interval": self.sampling_interval,
            "train_dataset_dict_dl_sd": {k: v.state_dict() for k,v in self.train_dataset_dict_dl.items()},
            "eval_batch_size": self.eval_batch_size,
            "category_list": self.category_list,
            "id2cat": self.id2cat,
            "cat2id": self.cat2id,
            "total_categories": self.total_categories,
            "sampling_weights": self.sampling_weights,
            "sampling_ratio": self.sampling_ratio,
            "curr_cat_count": self.curr_cat_count,
            "produced": self.produced,
            "arm_idx": self.arm_idx,
            "reward_type":  self.reward_type.__str__(),
            "random_state": random.getstate()
            }

    def _reset_eval_dataloaders(self):
        """Helper function to reset eval dataloaders since
        they would be exhausted in the previous evaluation loop.
        """
        self.eval_dataset_dict_dl = {}
        for k, _ in self.eval_dataset_dict.items():
            # this can be improved with persistent workers and caching
            # dataloaders and resetting them when needed.
            self.eval_dataset_dict_dl[k] = (
                iter(
                    DataLoader(
                        self.eval_dataset_dict[k],
                        self.eval_batch_size,
                        shuffle=True,
                        num_workers=0,
                        collate_fn=(
                            self.eval_collators_dict[k]
                            if self.eval_collators_dict
                            else None
                        ),
                    )
                )
                if self.eval_dataset_dict[k]
                else None
            )

    def _update_sampling_ratio(self, weights) -> list:
        """Helper function to convert weights to ratio

        Args:
            weights: sampling weights

        Returns:
            list: sampling ratio
        """
        w = weights
        w_sum = w.sum()
        K = len(w)
        base = (1.0 - self.gamma) * (w / w_sum)
        expl = self.gamma / K
        self.sampling_ratio = (base + expl).tolist()
        return self.sampling_ratio

    def _update_weights(self, count, rewards) -> list:
        """Helper function to update MAB weights with rewards

        Args:
            count: size of total number of categories with count of samples per category
            rewards: same size of count with reward of samples per category

        Returns:
            list: sampling ratio
        """

        for arm in range(self.total_categories):
            avg_r = rewards[arm] / count[arm]
            est_r = avg_r / self.sampling_ratio[arm]
            self.sampling_weights[arm] *= math.exp(
                self.eta * est_r / self.total_categories
            )
        return self._update_sampling_ratio(self.sampling_weights)

    def _extract_information_from_state_for_reward(self, state=None, category=None):
        """Helper function to extract exact information that the reward computation
        can consume. This function has to be expanded for new rewards.

        Args:
            state: HF TrainerState object. Defaults to None.

        Returns:
            dict: arguments prepared for compute_reward function
        """
        if state is None:
            return {}
        if self.reward_type.startswith(Reward.ENTROPY):
            return {}
        if self.reward_type == Reward.TRAIN_LOSS:
            return {"train_loss_history": [d for d in state.log_history if "loss" in d]}
        if self.reward_type == Reward.VALIDATION_LOSS:
            assert category is not None
            return {
                "eval_loss_history": [
                    {"loss": d[f"eval_{category}_loss"], **d}
                    for d in state.log_history
                    if f"eval_{category}_loss" in d
                ]
            }
        if self.reward_type == Reward.GRADNORM:
            return {
                "gradnorm_history": [d for d in state.log_history if "grad_norm" in d]
            }
        return {}

    def update_sampling_weights(self, model, accelerator, state):
        """Function to update MAB weights based on the reward type provided
        during the initialization. This function has to be updated if adding
        new reward types and based on their information needs from training loop.

        Args:
            model: HF model object. Conversion of the model (train to inference mode)
            is NOT the responsibility of this function.
            accelerator: Accelerate object, used for distributed operations.
            Should be None of single GPU runs.
            TODO: There is a hard dependency on accelerator which would be relaxed
            in future versions.
            state: HF TrainerState object (other formats will be supported in the future).
            For custom loop, please prepare your state class following TrainerState class.
        """
        rewards = [0] * self.total_categories
        count = [0] * self.total_categories
        eval_dataset_dict = {}
        device = accelerator.device if accelerator else torch.device(0)
        self._reset_eval_dataloaders()
        for c in range(self.total_categories):
            # accelerator takes care of preparing the eval dataloaders for distributed inference.
            if accelerator:
                eval_dataset_dict[self.id2cat[c]] = (
                    accelerator.prepare(self.eval_dataset_dict_dl[self.id2cat[c]])
                    if self.eval_dataset_dict_dl.get(self.id2cat[c], None)
                    else None
                )
            else:
                eval_dataset_dict[self.id2cat[c]] = self.eval_dataset_dict_dl.get(
                    self.id2cat[c], None
                )
        for c in tqdm(
            range(self.total_categories), total=self.total_categories, desc="Categories"
        ):  # for trian loss you dont need to iterate over eval dataset.
            if not eval_dataset_dict[self.id2cat[c]]:
                rc = compute_reward(
                    model=model,
                    batch=None,
                    vocab_size=32000,
                    reward_type=self.reward_type,
                    current_category=c,
                    total_categories=self.total_categories,
                    last_sampled_category=self.arm_idx,
                    **self._extract_information_from_state_for_reward(
                        state, self.id2cat[c]
                    ),
                )
                rewards[c] += rc
                count[c] += 1
            else:
                for batch in tqdm(
                    eval_dataset_dict[self.id2cat[c]],
                    desc="Reward computation over eval dataset",
                ):
                    rc = compute_reward(
                        model=model,
                        batch={k: v.to(device) for k, v in batch.items()},
                        vocab_size=32000,
                        reward_type=self.reward_type,
                        current_category=c,
                        total_categories=self.total_categories,
                        last_sampled_category=self.arm_idx,
                        **self._extract_information_from_state_for_reward(
                            state, self.id2cat[c]
                        ),
                    )
                    rewards[c] += rc
                    count[c] += batch["input_ids"].shape[0]
        rewards = torch.tensor(rewards, device=device)
        count = torch.tensor(count, device=device)
        if accelerator:
            rewards = accelerator.reduce(rewards, reduction="sum")
            count = accelerator.reduce(count, reduction="sum")
        if accelerator.is_main_process:
            self._update_weights(count, rewards)
        self.log_to_file(
            {
                "current_sampling_weights": self.sampling_weights.tolist(),
                "current_sampling_ratio": self.sampling_ratio,
                "rewards": rewards.tolist(),
                "count": count.tolist(),
                "action": "update",
            }
        )
