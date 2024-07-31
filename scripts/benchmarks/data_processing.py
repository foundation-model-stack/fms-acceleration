from trl import DataCollatorForCompletionOnlyLM
from transformers import PreTrainedTokenizer
from typing import Dict, Callable, List

PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

RESPONSE_TEMPLATE = '### Response:'
DEFAULT_FIELDS = [
    'input_ids', 
    'attention_mask', 
    'labels'
]

# combine functions
# c = combine(a, b) then c(i) = b(a(i))
FUNC = Callable[[Dict], Dict]
def combine_functions(*funcs : FUNC) -> FUNC:
    def _combine(x):
        for f in funcs:
            x = f(x)
        return x

    return _combine

def build_data_formatting_func(
    tokenizer: PreTrainedTokenizer = None,
    formatting: str = 'instruct',
    tokenize: bool = True,
    input_field: str = 'input',
    dataset_text_field: str = 'output',
    features: List = None, 
):
    # FIFO
    funcs = []

    if features is None:
        features = set()

    if formatting == 'instruct':
        funcs.append(
            instruction_formatter(
                input_field=input_field,
                dataset_text_field=dataset_text_field
            )
        )

    if tokenize:
        funcs.append(
            tokenization(
                tokenizer,
                dataset_text_field=dataset_text_field
            )
        )

        if formatting == 'instruct':
            funcs.append(
                instruction_mask_loss(tokenizer)
            )

    if len(funcs) == 0:
        raise ValueError(
            "Unable to build a data formatting recipe"
        )

    return combine_functions(*funcs), {
        'remove_columns': features.union(
            set([input_field, dataset_text_field])
        ).difference(
            set(DEFAULT_FIELDS)
        )
    }

def instruction_formatter(
    input_field: str = "input", 
    dataset_text_field: str = "output"
):
    def format_fn(example: Dict):
        prompt_input, prompt_no_input = (
            PROMPT_DICT["prompt_input"],
            PROMPT_DICT["prompt_no_input"],
        )
        output = (
            prompt_input.format_map(example)
            if example.get(input_field, "") != ""
            else prompt_no_input.format_map(example)
        )
        output = f"{output} {example[dataset_text_field]}"
        return {dataset_text_field: output}

    return format_fn

def tokenization(
    tokenizer: PreTrainedTokenizer, 
    dataset_text_field: str = "output"
):
    def _tokenize(example):
        text_field = example[dataset_text_field] + tokenizer.eos_token
        return tokenizer(text_field)

    return _tokenize

def instruction_mask_loss(
    tokenizer: PreTrainedTokenizer, 
    response_template: str = RESPONSE_TEMPLATE, 
):
    # cheat, use the data collator to mask the loss tokens
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )[2:]
    collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer, ignore_index=-100)

    def collate_example(example):
        # single example
        collated_example = collator([example], return_tensors = "pt")
        # flatten the additional dim
        return {k: v.view(-1) for k,v in collated_example.items()}

    return collate_example
