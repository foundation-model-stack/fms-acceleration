# Third Party
import numpy as np
import pytest
from datasets import Dataset, DatasetDict

# First Party
from fms_acceleration_odm import OnlineMixingDataset


class DummySentenceTransformer:
    """Simple sentence embedder used to avoid network calls in tests."""

    def __init__(self, *_, **__):
        pass

    def encode(self, texts, **_):
        vectors = []
        for text in texts:
            if text in {"cat", "dog", "wolf"}:
                vectors.append([0.0, 0.0])
            else:
                vectors.append([10.0, 10.0])
        return np.asarray(vectors, dtype=np.float32)


def _patch_sentence_transformer(monkeypatch):
    monkeypatch.setattr(
        "fms_acceleration_odm.odm.auto_categorizer.SentenceTransformer",
        DummySentenceTransformer,
    )


def test_auto_categorize_single_dataset(monkeypatch):
    _patch_sentence_transformer(monkeypatch)
    dataset = Dataset.from_dict({"text": ["cat", "dog", "wolf", "apple", "pear", "banana"]})
    dataset_dict = DatasetDict({"train": dataset})

    def x(): # noqa: E731 - simple identity collator for test
        return

    collator = x
    odm_dataset = OnlineMixingDataset(
        dataset_dict=dataset_dict,
        collators_dict={"train": collator},
        eval_dataset_dict={},
        eval_collators_dict={},
        auto_categorize_config={
            "text_field": "text",
            "num_categories": 2,
            "category_prefix": "cluster",
            "model_name": "dummy",
        },
    )

    assert len(odm_dataset.dataset_dict) == 2
    assert set(odm_dataset.category_list) == {"train_cluster_0", "train_cluster_1"}
    # Ensure collators were broadcast to the generated categories
    assert set(odm_dataset.collators_dict.keys()) == set(odm_dataset.dataset_dict.keys())

    # Combined rows should match original dataset size
    total_rows = sum(len(ds) for ds in odm_dataset.dataset_dict.values())
    assert total_rows == len(dataset)


def test_auto_categorize_requires_text_field(monkeypatch):
    _patch_sentence_transformer(monkeypatch)
    dataset = Dataset.from_dict({"content": ["hello", "world"]})
    dataset_dict = DatasetDict({"train": dataset})

    with pytest.raises(ValueError):
        OnlineMixingDataset(
            dataset_dict=dataset_dict,
            collators_dict={},
            eval_dataset_dict={},
            eval_collators_dict={},
            auto_categorize_config={"text_field": "text", "model_name": "dummy"},
        )
