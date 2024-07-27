from dataclasses import dataclass
import warnings
from transformers import DefaultDataCollator, default_data_collator

@dataclass
class DataCollatorWithFlattening(DefaultDataCollator):
    """
    Data collator used for padding free approach. Does the following:
    - concatate the entire mini batch into single long sequence [1, total_tokens]
    - no padding will be added, returns `input_ids`, `labels` and `position_ids`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn(
            "Using `DataCollatorWithFlattening` will flatten the entire mini batch "
            "into single long sequence."
            "Make sure your attention computation is able to handle it!"
        )

    def __call__(self, features, return_tensors=None):
        """
        This implementation assumes that only 3 arguments, input_ids, position_ids and labels
        are needed by the model, anything else is dropped by the collator
        """       
        if return_tensors is None:
            return_tensors = self.return_tensors
        is_labels_provided = "labels" in features[0]
        ret = {"input_ids": [], "labels": [], "position_ids": []}
        for feature in features:
            ret["input_ids"] += feature["input_ids"]
            ret["position_ids"] += list(range(len(feature["input_ids"])))
            if is_labels_provided:
                ret["labels"] += [-100] + feature["labels"][1:]
            else:
                ret["labels"] += [-100] + feature["input_ids"][1:]
        return default_data_collator([ret], return_tensors)