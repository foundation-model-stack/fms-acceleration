"""Utilities to automatically cluster a dataset into pseudo categories."""

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

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from logging import getLogger
from typing import Any, Dict, List, Optional
import copy
import math

# Third Party
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

logger = getLogger(__name__)

AUTO_CATEGORIZATION_COLUMN_NAME = "auto_categorization_odm_raw_text"


@dataclass
class AutoCategorizeConfig:  # pylint: disable=too-many-instance-attributes
    """Configuration for sentence-embedding based auto-categorization."""

    input_column: str = "text"
    num_categories: Optional[int] = None
    min_categories: int = 2
    max_categories: int = 15
    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    batch_size: int = 64
    cluster_algo: str = "kmeans"
    category_prefix: str = "auto_category"
    # Args for loading model
    model_kwargs: Dict[str, any] = field(
        default_factory=lambda: {
            "device_map": "auto",
            # "attn_implementation": "flash_attention_2",
        }
    )
    # Args for K means
    cluster_kwargs: Dict[str, Any] = field(default_factory=dict)
    # If the `input_column`` provided does not contain str
    # it is assumed that the data is pre-tokenized
    # and the column will first be detokenized using the tokenizer
    # before performing k means
    tokenizer: Optional[Any] = None


class DatasetAutoCategorizer:
    """Clusters a dataset into pseudo categories using embeddings."""

    def __init__(self, config: Optional[AutoCategorizeConfig] = None):
        self.config = copy.deepcopy(config) or AutoCategorizeConfig()

    def __call__(self, dataset: Dataset) -> DatasetDict:
        if isinstance(dataset, torch.utils.data.IterableDataset):
            raise NotImplementedError(
                "Iteratble (or streaming) datasets are not yet supported for auto categorization."
                "Please use a non-iterable dataset."
            )

        if len(dataset) == 0:
            raise ValueError("Cannot auto-categorize an empty dataset")
        if self.config.input_column not in dataset.column_names:
            raise ValueError(
                "Dataset is missing column '{col}'. Provide a input field in "
                "auto_categorize_config['input_column'].".format(
                    col=self.config.input_column
                )
            )

        dataset = self._maybe_detokenize_data(dataset)

        num_categories = self._determine_category_count(len(dataset))
        logger.info(
            "Auto-categorizing %s rows into %s clusters using %s",
            len(dataset),
            num_categories,
            self.config.model_name,
        )
        embeddings = self._compute_embeddings(dataset)
        labels = self._cluster_embeddings(embeddings, num_categories)

        if AUTO_CATEGORIZATION_COLUMN_NAME in dataset.column_names:
            dataset = dataset.remove_columns(AUTO_CATEGORIZATION_COLUMN_NAME)

        return self._build_dataset_dict(dataset, labels)

    def _maybe_detokenize_data(self, dataset: Dataset) -> Dataset:
        existing_field = self.config.input_column

        if isinstance(dataset[existing_field][0], str):
            logger.info("Detokenization not needed, text data already provided")
            return dataset

        assert self.config.tokenizer is not None, (
            "Attempting detokenizing the data on column '{%s}' but the tokenizer is not provided",
            self.config.input_column,
        )
        assert AUTO_CATEGORIZATION_COLUMN_NAME not in dataset.column_names, (
            "Default detokenizing column '{%s}' is already present in the dataset",
            AUTO_CATEGORIZATION_COLUMN_NAME,
        )

        tokenizer = self.config.tokenizer

        dataset = dataset.map(
            lambda x: {
                AUTO_CATEGORIZATION_COLUMN_NAME: tokenizer.batch_decode(
                    x[existing_field]
                )
            },
            batched=True,
            num_proc=12,
        )
        self.config.input_column = AUTO_CATEGORIZATION_COLUMN_NAME

        return dataset

    def _determine_category_count(self, dataset_size: int) -> int:
        if self.config.num_categories is not None:
            desired = self.config.num_categories
        else:
            # heuristic: sqrt scaling with dataset size
            desired = int(math.sqrt(max(dataset_size, 1)))
            desired = max(desired, self.config.min_categories)
            desired = min(desired, self.config.max_categories)

        # clusters cannot exceed dataset size and must be >=1
        desired = max(1, min(dataset_size, desired))
        return desired

    def _compute_embeddings(self, dataset: Dataset) -> np.ndarray:
        model = SentenceTransformer(
            self.config.model_name,
            model_kwargs=self.config.model_kwargs,
            prompts={
                "clustering": "Identify the topic or theme based on the text: ",
            },
            default_prompt_name="clustering",
        )

        vectors = model.encode(
            dataset[self.config.input_column],
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
        )
        return vectors

    def _cluster_embeddings(
        self, embeddings: np.ndarray, num_categories: int
    ) -> np.ndarray:
        if self.config.cluster_algo.lower() != "kmeans":
            raise ValueError(
                "Unsupported clustering algorithm '%s'. Only 'kmeans' is currently supported."
                % self.config.cluster_algo
            )

        try:
            from cuml import KMeans # pylint: disable=import-outside-toplevel
            print("Using GPU accelerated Kmeans")
        except ImportError:
            print("GPU accelerated KMeans is not avaialble. Falling back to CPU based KMeans")
            from sklearn.cluster import KMeans # pylint: disable=import-outside-toplevel

        kwargs = {"n_init": 10}
        kwargs.update(self.config.cluster_kwargs)
        model = KMeans(n_clusters=num_categories, **kwargs)

        logger.info("Starting %s clustering", self.config.cluster_algo)

        return model.fit_predict(embeddings)

    def _build_dataset_dict(self, dataset: Dataset, labels: np.ndarray) -> DatasetDict:
        grouped_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels.tolist()):
            grouped_indices.setdefault(label, []).append(idx)
        categorized = {}
        for label, indices in sorted(grouped_indices.items()):
            name = f"{self.config.category_prefix}_{label}"
            categorized[name] = dataset.select(indices)
        return DatasetDict(categorized)
