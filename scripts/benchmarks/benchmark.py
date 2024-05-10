import argparse
import json
import os
import re
import subprocess
import warnings
from copy import copy
from itertools import product
from typing import Callable, Dict, List, Tuple, Any

import datasets
import pandas as pd
import yaml
from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments

"""
This benchmarking script 
    1. Prepares a standard BenchmarkDataset
    2. Prepares a list of experiment arguments from a set of configs
    (TrainDefaultsConfig, TrainScenariosConfig, ExperimentConfig)
    3. Builds a list of experiment objects to run based on the set of experiment arguments
    4. Consolidates the experiment results into a summary
"""

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

COMMAND_PYTHON = "python"
COMMAND_ACCELERATE = "accelerate launch --config_file {accelerate_config_path} --num_processes={num_processes} --main_process_port={process_port}"
FMS_TRAINER = "-m tuning.sft_trainer"
TRUE_FALSE_ARGUMENTS = []

FILE_STDOUT = "stdout"
FILE_STDERR = "stderr"
FILE_SUMMARY_CSV = 'summary.csv'

# regex to capture the start and end of tracebacks
REGEX_START_OF_TRACEBACK = "Traceback\s\(most\srecent\scall\slast\)"
REGEX_END_OF_TRACEBACK = "\w+Error"

# if any of this errors appear in a traceback, then we will ignore the whole traceback
IGNORE_ERROR_PATTERNS = [
    # dont need to surface torch distributed errors
    "torch.distributed.elastic.multiprocessing.errors.ChildFailedError"
]


def get_hf_arguments_with_no_value(dataclass_types):
    """this function will return a map (str, bool) of true/false arguments.
    The boolean indicates that the prescence of the switch indicates that value
    e.g., (fp16, True) means --fp16 means fp16: True, and vice-versa.
    """
    results = {}
    parser = HfArgumentParser(dataclass_types)
    for action in parser._actions:
        if action.__class__.__name__ in ("_StoreTrueAction", "_StoreFalseAction"):
            key = action.option_strings[0]  # just take the first one for now
            results[key] = not action.default
    return results


# populate the true / false arguments map
TRUE_FALSE_ARGUMENTS = get_hf_arguments_with_no_value(dataclass_types=TrainingArguments)


def format_fn(example, input_key: str = "input", output_key: str = "output"):
    prompt_input, prompt_no_input = (
        PROMPT_DICT["prompt_input"],
        PROMPT_DICT["prompt_no_input"],
    )
    output = (
        prompt_input.format_map(example)
        if example.get(input_key, "") != ""
        else prompt_no_input.format_map(example)
    )
    output = f"{output} {example[output_key]}"
    return {output_key: output}


class BenchmarkDataset:
    def __init__(
        self,
        dataset_name: str,
        format_fn: Callable,
        unused_columns: List[str] = ["instruction", "input"],
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset = self.prepare_dataset(format_fn, unused_columns=unused_columns)

    def save_to_path(self, save_path: str):
        self.dataset.to_json(save_path)

    def prepare_dataset(
        self,
        format_fn: Callable = None,
        dataset_split: str = "train",
        unused_columns: List[str] = None,
    ):
        ds = datasets.load_dataset(self.dataset_name)
        if format_fn:
            ds = ds[dataset_split].map(format_fn, remove_columns=unused_columns)
        return ds


def convert_keypairs_to_map(keypairs: List):
    return {key: val for key, val in zip(keypairs[::2], keypairs[1::2])}


class ConfigUtils:
    @staticmethod
    def read_yaml(yaml_filepath: str):
        with open(str(yaml_filepath)) as stream:
            config = yaml.safe_load(stream)
        return config

    @staticmethod
    def convert_keyvalue_arguments_to_list(args_dict: Dict):
        """
        Used to convert a dictionary of args to a list of [--<arg>, <value>, ...]
        """
        argslist = []
        for arg, val in args_dict.items():
            if arg in TRUE_FALSE_ARGUMENTS:
                # if its a true / false argument
                if val is None and TRUE_FALSE_ARGUMENTS.get(arg) != val:
                    argslist.append(f"--{arg}")

                continue

            # otherwise if a regular argument
            if val is None:
                warnings.warn(
                    f"Argument '{arg}' is not a true/false argument andhad a 'None' value ",
                    "and thus will be ignored.",
                )
                continue

            # append the key value pair
            argslist.append(f"--{arg}")
            argslist.append(val)

        return argslist

    @staticmethod
    def build_args_from_products(products: List[Dict], defaults: Dict):
        # products expected to be
        # output: [{config1: 1, config2: 4}, {config1: 1, config2: 5}, ...]
        args = []
        for product in products:
            num_gpus = product.pop("num_gpus")
            effective_batch_size = product.pop("effective_batch_size")
            combined_args = {**product, **defaults}
            argument_list = ConfigUtils.convert_keyvalue_arguments_to_list(
                combined_args
            )
            argument_list.extend(
                [
                    "--per_device_train_batch_size",
                    str(effective_batch_size // num_gpus),
                ]
            )
            args.append((num_gpus, argument_list))
        return args

    @staticmethod
    def cartesian_product_on_dict(variable_matrices: Dict) -> List[Dict]:
        """
        Used to cartesian product a dictionary of set of configurations
        input: { config1: [1,2,3], config2: [4,5,6], ...}
        output: [{config1: 1, config2: 4}, {config1: 1, config2: 5}, ...]
        """
        list_of_products = []
        product_factors = variable_matrices.values()
        for arg_combinations in product(*product_factors):
            list_of_products.append(
                {
                    name: arg
                for name, arg in zip(variable_matrices.keys(), arg_combinations)
                }
            )
        return list_of_products

    @staticmethod
    def convert_args_to_dict(experiment_arguments: List[Any]):
        "this function converts an uneven keypair list, where some keys are missing values"
        argument_dict = {}
        for item in experiment_arguments:
            if "--" in item:
                current_key = item.replace("--", "")
                argument_dict[current_key] = None
            else:
                v = argument_dict[current_key]
                # is value
                if v is None:
                    argument_dict[current_key] = item
                else:
                    # otherwise it was from a list, so make into sequence
                    argument_dict[current_key] = v + ' ' + item

        return argument_dict


class ScenarioMatrix:

    matrix_args = ['model_name_or_path']

    def __init__(self, scenario: Dict, acceleration_config_map: Dict = None) -> None:
        assert "arguments" in scenario.keys(), "Missing `arguments` key in `scenario`"
        for key, val in scenario.items():
            if key == "framework_config":
                # if acceleration_config_map is None, then do not do mapping
                if acceleration_config_map:
                    val = [
                        acceleration_config_map[k]
                        for k in val
                        if k in acceleration_config_map
                    ]
            setattr(self, key, val)

    def get_scenario_matrices_and_defaults(self):
        scenario_defaults = {}
        matrices = {}
        for arg_name, arg_value in self.arguments.items():
            if arg_name in ScenarioMatrix.matrix_args:
                matrices[arg_name] = arg_value
            elif isinstance(arg_value, list):
                scenario_defaults[arg_name] = [x for x in arg_value]
            else:
                scenario_defaults[arg_name] = arg_value
        matrices["acceleration_framework_config_file"] = getattr(
            self, "framework_config", []
        )
        return matrices, scenario_defaults


class Experiment:
    def __init__(
        self,
        num_gpus: int,
        experiment_arg: List,
        save_dir: str,
        tag: str = None,
    ) -> None:
        self.num_gpus = num_gpus
        self.experiment_arg = experiment_arg
        self.result = None
        self.tag = tag

        # directories
        self.save_dir = save_dir
        self.stdout_filename = os.path.join(self.save_dir, FILE_STDOUT)
        self.stderr_filename = os.path.join(self.save_dir, FILE_STDERR)

    def run(self, run_cmd: str, environment_variables: Dict = None):

        # form the command line
        commands = []
        for c in self.experiment_arg:
            if isinstance(c, list):
                commands.extend([str(x) for x in c])
            else:
                commands.append(str(c))
            
        # will save the command line in str
        self.experiment_args_str = commands
        os.makedirs(self.save_dir, exist_ok=True)
        subprocess.run(
            run_cmd.split() + commands, 
            capture_output=False,
            stdout=open(self.stdout_filename, "w"),
            stderr=open(self.stderr_filename, "w"),
            text=True,
            env={**os.environ.copy(), **environment_variables},
        )

    def get_experiment_final_metrics(
        self, final_metrics_keys: List[str] = ["train_loss", "train_runtime"]
    ):
        results = self.get_printlogger_output()
        # TODO: now we make the assumption that the final json printout is
        # the summary stats, but maybe we can put more robust checking
        if len(results) == 0:
            return {}

        # infer the final metric
        results = [x for x in results if all([y in x for y in final_metrics_keys])]
        if len(results) != 1:
            warnings.warn(
                f"Unable to infer the final metrics for experiment '{self.tag}'"
            )
            return {}  # return empty dictionary
        return results[-1]

    def get_printlogger_output(self):
        "method to get all the print logger outputs"
        results = []
        with open(self.stdout_filename, "r") as f:
            for x in f.readlines():
                try:
                    # the printlogger will print dictionary items.
                    # - read it as a json by replacing the single quotes for doubles
                    results.append(json.loads(x.strip().replace("'", '"')))
                except json.JSONDecodeError:
                    pass
        return results

    def maybe_get_experiment_error_traceback(self):
        "Function to extract the relevant error trace from the run, if any."

        results = []
        current_traceback = []
        within_traceback = 0
        with open(self.stderr_filename, "r") as f:
            for line in f.readlines():
                if re.match(REGEX_START_OF_TRACEBACK, line):
                    within_traceback += 1

                if within_traceback > 0:
                    current_traceback.append(line)

                    # reached the end, do not take in any more
                    if re.match(REGEX_END_OF_TRACEBACK, line):
                        within_traceback -= 1
                        current_traceback = "\n".join(current_traceback)
                        if not any(
                            [x in current_traceback for x in IGNORE_ERROR_PATTERNS]
                        ):
                            results.append(current_traceback)
                        current_traceback = []

        return None if len(results) == 0 else results


def prepare_arguments(args):
    defaults = ConfigUtils.read_yaml(args.defaults_config_path)
    defaults["training_data_path"] = args.dataset_save_path
    scenarios = ConfigUtils.read_yaml(args.scenarios_config_path)["scenarios"]
    acceleration_config_map = convert_keypairs_to_map(
        args.acceleration_framework_config_keypairs
    )
    experiment_matrices = {
        "effective_batch_size": args.effective_batch_size_matrix,
        "num_gpus": args.num_gpus_matrix,
        "packing": args.packing_matrix,
        "max_seq_len": args.max_seq_len_matrix,
    }
    for scenario_config in scenarios:
        # if a `run_only_scenarios` list exist, filter out any scenario not in the list
        if (
            args.run_only_scenarios
            and scenario_config["name"] not in args.run_only_scenarios
        ):
            continue
        scenario = ScenarioMatrix(scenario_config, acceleration_config_map)
        scenario_matrices, scenario_constants = (
            scenario.get_scenario_matrices_and_defaults()
        )
        # update defaults with scenario constants
        constants = {**scenario_constants, **defaults}
        # Remove any empty variables and combine matrices to dictionary to cartesian product on
        combined_matrices = {**scenario_matrices, **experiment_matrices}
        combined_matrices = {
            key: val for key, val in combined_matrices.items() if len(val) > 0
        }
        products = ConfigUtils.cartesian_product_on_dict(combined_matrices)
        for num_gpus, experiment_arg in ConfigUtils.build_args_from_products(
            products, constants
        ):
            yield num_gpus, experiment_arg


def generate_list_of_experiments(
    experiment_args: List[Tuple[int, List]],
    output_dir: str = "results",
    hf_products_dir: str = "hf",
) -> List[Experiment]:
    """Construct list of experiments to be run. Takes in default_config and
    any matrices in scenario and experiment_config
    """
    experiments = []
    for _expr_id, (num_gpus, exp_arg) in enumerate(experiment_args):
        experiment_tag = f"exp_{_expr_id}"
        experiment_output_dir = os.path.join(output_dir, experiment_tag)
        expr_arg_w_outputdir = exp_arg + [
            "--output_dir",
            os.path.join(experiment_output_dir, hf_products_dir),
        ]
        _expr = Experiment(
            num_gpus,
            expr_arg_w_outputdir,
            save_dir=experiment_output_dir,
            tag=experiment_tag,
        )
        experiments.append(_expr)
    return experiments


def main(args):

    # 1. Prepares a standard BenchmarkDataset
    # TODO: consider caching the json file
    benchmark_dataset = BenchmarkDataset(args.dataset_name, format_fn)
    benchmark_dataset.save_to_path(args.dataset_save_path)

    # 2. Prepares a list of experiment arguments from a set of configs
    experiment_args = prepare_arguments(args)

    # 3. Builds a list of experiment objects to run based on the set of experiment arguments
    experiment_stats = {}
    experiment: Experiment
    for experiment in tqdm(generate_list_of_experiments(
        experiment_args, output_dir=args.results_output_path
    )):
        if experiment.num_gpus > 1:
            prefix = COMMAND_ACCELERATE.format(
                accelerate_config_path=args.accelerate_config,
                num_processes=experiment.num_gpus,
                process_port=args.process_port,
            )
        else:
            prefix = COMMAND_PYTHON

        device_ids = ",".join([str(i) for i in range(experiment.num_gpus)])
        experiment.run(
            f"{prefix} {FMS_TRAINER}",
            environment_variables={"CUDA_VISIBLE_DEVICES": device_ids},
        )

        # if there is an error we save the error message else we save the final result
        maybe_error_messages = experiment.maybe_get_experiment_error_traceback()
        if maybe_error_messages is None:
            save_result = {
                **ConfigUtils.convert_args_to_dict(experiment.experiment_args_str),
                **experiment.get_experiment_final_metrics(),
            }
            experiment_stats[experiment.tag] = save_result
        else:
            experiment_stats[experiment.tag] = {"error_messages": maybe_error_messages}

    # 4. Consolidates the experiment results into a summary
    df = pd.DataFrame.from_dict(experiment_stats, orient="index")
    df.to_csv(
        os.path.join(args.results_output_path, FILE_SUMMARY_CSV), index=None
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Acceleration Benchmarking",
        description="This script runs a set of benchmarks on the acceleration library",
    )
    parser.add_argument(
        "--effective_batch_size_matrix",
        type=int,
        nargs="+",
        default=[4, 8],
        help="list of batch sizes to benchmark on",
    )
    parser.add_argument(
        "--num_gpus_matrix",
        type=int,
        nargs="+",
        default=[1, 2],
        help="list of gpus to benchmark on",
    )
    parser.add_argument(
        "--packing_matrix",
        type=bool,
        nargs="+",
        default=[True],
        help="True to pack datasets or False to pad dataset",
    )
    parser.add_argument(
        "--max_seq_len_matrix",
        type=int,
        nargs="+",
        default=[4096],
        help="list of gpus to benchmark on",
    )
    parser.add_argument(
        "--acceleration_framework_config_keypairs",
        type=str,
        nargs="+",
        default=[
            "accelerated-peft-autogptq",
            "./sample-configurations/accelerated-peft-autogptq-sample-configuration.yaml",
            "accelerated-peft-autogptq-unsloth",
            "./sample-configurations/accelerated-peft-autogptq-unsloth-sample-configuration.yaml",
        ],
        help="list of (key, file) keypairs",
    )
    parser.add_argument(
        "--run_only_scenarios",
        type=str,
        nargs="+",
        default=None,
        help="scenarios selected",
    )
    parser.add_argument(
        "--scenarios_config_path",
        type=str,
        default="./scenarios.yaml",
        help="path to scenarios config file",
    )
    parser.add_argument(
        "--defaults_config_path",
        type=str,
        default="./defaults.yaml",
        help="path to defaults config file",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="yahma/alpaca-cleaned",
        help="dataset to benchmark on",
    )
    parser.add_argument(
        "--dataset_save_path",
        type=str,
        default="./data/benchmark_data.json",
        help="dataset cache path",
    )
    parser.add_argument(
        "--accelerate_config",
        type=str,
        default="./fsdp_defaults.yaml",
        help="accelerate config file path",
    )
    parser.add_argument(
        "--results_output_path",
        type=str,
        default="./results",
        help="accelerate config file path",
    )
    parser.add_argument(
        "--process_port", type=int, default=29500, help="accelerate process port"
    )
    args = parser.parse_args()
    main(args)
