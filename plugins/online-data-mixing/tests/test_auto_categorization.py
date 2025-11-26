# Third Party
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
import numpy as np
import pytest
import torch

# First Party
from fms_acceleration_odm import OnlineMixingDataset

np.random.seed(42)
torch.random.manual_seed(42)


class DummySentenceTransformer:
    """Simple sentence embedder used to avoid network calls in tests."""

    def __init__(self, *_, **__):
        pass

    def encode(self, texts, **_):
        vectors = []
        for _ in texts:
            if np.random.uniform() < 0.5:
                vectors.append([0.0, 0.0])
            else:
                vectors.append([10.0, 10.0])
        return np.asarray(vectors, dtype=np.float32)


class DummyIterable(torch.utils.data.IterableDataset):
    def __iter__(self):
        yield {"x": 1}


def _patch_sentence_transformer(monkeypatch):
    monkeypatch.setattr(
        "fms_acceleration_odm.odm.auto_categorizer.SentenceTransformer",
        DummySentenceTransformer,
    )


def test_auto_categorize_single_dataset(monkeypatch):
    _patch_sentence_transformer(monkeypatch)
    dataset = Dataset.from_dict(
        {"text": ["cat", "dog", "wolf", "apple", "pear", "banana"]}
    )
    dataset_dict = DatasetDict({"train": dataset})

    def x():  # noqa: E731 - simple identity collator for test
        return

    collator = x
    odm_dataset = OnlineMixingDataset(
        dataset_dict=dataset_dict,
        collators_dict={"train": collator},
        eval_dataset_dict={},
        eval_collators_dict={},
        auto_categorize_config={
            "input_column": "text",
            "num_categories": 2,
            "category_prefix": "cluster",
            "model_name": "dummy",
        },
    )

    assert len(odm_dataset.dataset_dict) == 2
    assert set(odm_dataset.category_list) == {"cluster_0", "cluster_1"}
    assert set(odm_dataset.collators_dict.keys()) == set(
        odm_dataset.dataset_dict.keys()
    )

    total_rows = sum(len(ds) for ds in odm_dataset.dataset_dict.values())
    assert total_rows == len(dataset)


def test_auto_categorize_requires_input_column(monkeypatch):
    _patch_sentence_transformer(monkeypatch)
    dataset = Dataset.from_dict({"content": ["hello", "world"]})
    dataset_dict = DatasetDict({"train": dataset})

    with pytest.raises(ValueError):
        OnlineMixingDataset(
            dataset_dict=dataset_dict,
            collators_dict={},
            eval_dataset_dict={},
            eval_collators_dict={},
            auto_categorize_config={"input_column": "text", "model_name": "dummy"},
        )


def test_auto_categorize_pretokenized_data_w_tokenizer(monkeypatch):
    _patch_sentence_transformer(monkeypatch)

    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")

    batch_size, seq_len = 16, 50
    dataset = Dataset.from_dict(
        {"input_ids": torch.randint(0, tok.vocab_size, (batch_size, seq_len))}
    )
    dataset_dict = DatasetDict({"train": dataset})

    def x():  # noqa: E731 - simple identity collator for test
        return

    collator = x
    odm_dataset = OnlineMixingDataset(
        dataset_dict=dataset_dict,
        collators_dict={"train": collator},
        eval_dataset_dict={},
        eval_collators_dict={},
        auto_categorize_config={
            "input_column": "input_ids",
            "num_categories": 2,
            "category_prefix": "cluster",
            "model_name": "dummy",
            "tokenizer": tok,
        },
    )

    print(odm_dataset.dataset_dict, len(odm_dataset.dataset_dict))

    assert len(odm_dataset.dataset_dict) == 2
    assert set(odm_dataset.category_list) == {"cluster_0", "cluster_1"}
    assert set(odm_dataset.collators_dict.keys()) == set(
        odm_dataset.dataset_dict.keys()
    )

    total_rows = sum(len(ds) for ds in odm_dataset.dataset_dict.values())
    assert total_rows == len(dataset) == batch_size


def test_auto_categorize_pretokenized_data_wo_tokenizer(monkeypatch):
    _patch_sentence_transformer(monkeypatch)

    batch_size, seq_len = 16, 50
    dataset = Dataset.from_dict(
        {"input_ids": torch.randint(0, 100, (batch_size, seq_len))}
    )
    dataset_dict = DatasetDict({"train": dataset})

    def x():  # noqa: E731 - simple identity collator for test
        return

    collator = x

    with pytest.raises(AssertionError):
        _ = OnlineMixingDataset(
            dataset_dict=dataset_dict,
            collators_dict={"train": collator},
            eval_dataset_dict={},
            eval_collators_dict={},
            auto_categorize_config={
                "input_column": "input_ids",
                "num_categories": 2,
                "category_prefix": "cluster",
                "model_name": "dummy",
            },
        )


def test_iterable_dataset_not_supported_auto_categorize(monkeypatch):

    _patch_sentence_transformer(monkeypatch)

    dataset = DummyIterable()
    dataset_dict = DatasetDict({"train": dataset})

    def x():  # noqa: E731 - simple identity collator for test
        return

    collator = x

    with pytest.raises(NotImplementedError) as err:
        _ = OnlineMixingDataset(
            dataset_dict=dataset_dict,
            collators_dict={"train": collator},
            eval_dataset_dict={},
            eval_collators_dict={},
            auto_categorize_config={
                "input_column": "input_ids",
                "num_categories": 2,
                "category_prefix": "cluster",
                "model_name": "dummy",
            },
        )

    assert "not yet supported" in str(err.value)
