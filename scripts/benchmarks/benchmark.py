# Standard
from itertools import product
from typing import Any, Callable, Dict, List, Tuple, Union
import argparse
import json
import os
import re
import subprocess
import warnings

# Third Party
from tqdm import tqdm
from transformers import AutoConfig, HfArgumentParser, TrainingArguments
import datasets
import pandas as pd
import torch
import yaml

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
FILE_RESULTS = "results.json"
FILE_SHELL_COMMAND = "command.sh"
FILE_SCRIPT_ARGS = "script.json"
FILE_SUMMARY_CSV = "raw_summary.csv"

DIR_BENCHMARKS = os.path.dirname(os.path.realpath(__file__))
DIR_PREFIX_EXPERIMENT = "exp"
DIR_NAME_RESULTS_DEFAULT = "benchmark_results"
DIR_SAMP_CONFIGS = os.path.join(DIR_BENCHMARKS, "../../sample-configurations")

# read list of sample configurations from contents file
FRAMEWORK_CONFIG_KEYPAIRS = []
with open(os.path.join(DIR_SAMP_CONFIGS, "CONTENTS.yaml")) as f:
    configs = yaml.safe_load(f)["framework_configs"]
    for d in configs:
        FRAMEWORK_CONFIG_KEYPAIRS.append(d["shortname"])
        FRAMEWORK_CONFIG_KEYPAIRS.append(os.path.join(DIR_SAMP_CONFIGS, d["filename"]))

# regex to capture the start and end of tracebacks
REGEX_START_OF_TRACEBACK = "Traceback\s\(most\srecent\scall\slast\)"
REGEX_END_OF_TRACEBACK = "\w+Error"

# if any of this errors appear in a traceback, then we will ignore the whole traceback
IGNORE_ERROR_PATTERNS = [
    # dont need to surface torch distributed errors
    "torch.distributed.elastic.multiprocessing.errors.ChildFailedError"
]

FILE_MEM = "gpu_memory_logs.csv"
GPU_LOG_USED_MEM_COLUMN_NAME = "memory.used [MiB]"
GPU_LOG_METRIC_SUFFIX = " MiB"
GPU_TABLE = "timestamp,name,index,memory.used"
REPORT_GPU_FIELD_NAME = "reserved_gpu_mem"
REPORT_DEVICE_FIELD_NAME = "gpu_device_name"

HF_TRAINER_LOG_GPU_MEMORY_STAGES = ['before', 'init', 'train']
HF_ARG_SKIP_MEMORY_METRIC = "skip_memory_metrics"
RESULT_FIELD_ALLOCATED_GPU_MEM = "alloc_gpu_memory_in_bytes"

def extract_gpu_memory_metrics(output_metrics) -> Dict[str, float]:
    """
    This function extracts the gpu memory metrics from the output of HFTrainer.train()
    when `skip_memory_metrics` is set to `False` in transformers.TrainingArguments
    HFTrainer.train will return memory logs in `train_metrics` for each of the following stages 
    A. absolute memory before trainer initialization
    B. delta of allocated memory of a stage
    C. peaked_delta of a stage (extra mem consumed and freed in between) 
    
    B. & C. are taken before/after the following stages:
    - <init>: `HFTrainer.__init__`
    - <train>: `HFTrainer.train` (only if called)
    - <evaluate>: `HFTrainer.evaluate` (only if called)
    - <predict>: `HFTrainer.predict` (only if called)

    - e.g. Memory for train stage = B_<train> + C_<train>
    - e.g. Memory consumption in training = A + B_<init> + C_<init> + B_<train> + C_<train>

    NOTE:
    - HFTrainer only takes the rank0 memory when in distributed mode
    - HFTrainer uses `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()` underneath
    - https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.Trainer.log_metrics:~:text=inject%20custom%20behavior.-,log_metrics,-%3C

    Returns a dictionary of keys and gpu memory values (in bytes) 
    corresponding to the relevant stage and delta keys for training 
    """
    return {
        key: val
        for key, val in output_metrics.items() if 'gpu' in key and
        any([stage in key for stage in HF_TRAINER_LOG_GPU_MEMORY_STAGES])
    }

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
                    argument_dict[current_key] = v + " " + item

        return argument_dict


class ScenarioMatrix:

    matrix_args = ["model_name_or_path"]

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

    def preload_models(self):
        for model_name in self.arguments["model_name_or_path"]:
            print(f"Scenario '{self.name}' preloading model '{model_name}'")
            # just preload the config
            AutoConfig.from_pretrained(model_name)

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
        if hasattr(self, "framework_config"):
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

        # to be set in run
        self.shell_command = None
        self.experiment_args_str = None
        self.environment = None

        # directories
        self.save_dir = save_dir
        self.stdout_filename = os.path.join(self.save_dir, FILE_STDOUT)
        self.stderr_filename = os.path.join(self.save_dir, FILE_STDERR)
        self.command_filename = os.path.join(self.save_dir, FILE_SHELL_COMMAND)
        self.results_filename = os.path.join(self.save_dir, FILE_RESULTS)
        self.gpu_log_filename = os.path.join(self.save_dir, FILE_MEM)

    def run(
        self,
        run_cmd: str,
        environment_variables: Dict = None,
        log_nvidia_smi: bool = False,
        memory_log_interval_secs: int = 1,
    ):

        # form the command line
        commands = []
        for c in self.experiment_arg:
            if isinstance(c, list):
                commands.extend([str(x) for x in c])
            else:
                commands.append(str(c))

        # will save the command line in str
        self.shell_command = run_cmd.split() + commands
        self.environment = environment_variables
        self.experiment_args_str = commands
        os.makedirs(self.save_dir, exist_ok=True)

        if log_nvidia_smi:
            """
            Opens a parallel process to log the device memory of the main experiment process.
            - Logs memory at intervals to a csv file in `self.save_dir`
            - Terminates at the end of experiment
            - GPU log is read and aggregated when the experiment ends & results are saved in Experiment.write_result,

            NOTE: This feature assumes the following
            1. Experiment is the only process on the gpu devices -
            there are no other processes running on the device in parallel.

            Can log more info from nvidia-smi by expanding GPU_Table argument
            e.g. "timestamp,name,index,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used"
            Use `nvidia-smi --help-query-gpu` for more reference
            """
            nvidia_logging_cmd = [
                "nvidia-smi",
                "--query-gpu",
                GPU_TABLE,
                "--format",
                "csv",
                "--id",
                str(environment_variables["CUDA_VISIBLE_DEVICES"]),
                "--loop",
                str(memory_log_interval_secs),
            ]
            memory_process = subprocess.Popen(
                nvidia_logging_cmd,
                stdout=open(self.gpu_log_filename, "w"),
                text=True,
            )

        subprocess.run(
            self.shell_command,
            capture_output=False,
            stdout=open(self.stdout_filename, "w"),
            stderr=open(self.stderr_filename, "w"),
            text=True,
            env={**os.environ.copy(), **environment_variables},
        )

        if log_nvidia_smi:
            memory_process.terminate()

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

    def get_peak_mem_usage_by_device_id(self):
        """
        This function retrieves the raw measurements of reserved GPU memory per device across the experiment -
        and perform some simple calibration (subtracts values by the first reading) before computing the peak value for each gpu.
        Returns:
            - pd.Series of peak memory usage per device id
            - the device name as string - e.g. "NVIDIA A100-SXM4-80GB"

        Example: For 2 devices with GPU Indices 0,1 - it will return the max measurement value (in MiB) of each device as a Series:

        - pd.Series
        index
        0    52729.0
        1    52783.0
        Name: memory.used [MiB], dtype: float64
        """

        # group the gpu readings into device ids
        gpu_logs = pd.read_csv(self.gpu_log_filename, skipinitialspace=True)
        # assume that all the devices have the same device name
        device_name = gpu_logs.name.iloc[-1]
        grouped_indices = gpu_logs.groupby(by="index")
        # extract out the gpu memory usage as float values
        mem_usage_by_device_id = grouped_indices.apply(
            lambda x: x[GPU_LOG_USED_MEM_COLUMN_NAME]
            .str.replace(GPU_LOG_METRIC_SUFFIX, "")
            .astype(float)
        ).reset_index(level=1, drop=True)

        # Calibrate values by subtracting out the initial values of the GPU readings
        # to ensure no existing memory is counted in addition with the experiment
        initial_values = mem_usage_by_device_id.groupby("index").first()
        mem_usage_by_device_id = mem_usage_by_device_id.sub(initial_values)
        # Return the max memory for each device
        return mem_usage_by_device_id.groupby("index").max(), device_name

    def write_result(self):
        "Function to write a json result file"

        # save some basic args
        save_result = ConfigUtils.convert_args_to_dict(self.experiment_args_str)
        save_result["num_gpus"] = self.num_gpus

        # if a gpu log file exist, process the raw nvidia logs and write to result
        if os.path.isfile(self.gpu_log_filename):
            # Add GPU info and measurements into the result saving
            peak_mem_usage_by_device_id, device_name = self.get_peak_mem_usage_by_device_id()
            save_result[REPORT_DEVICE_FIELD_NAME] = device_name
            # Memory usage is averaged across all devices in the final result
            save_result[REPORT_GPU_FIELD_NAME] = peak_mem_usage_by_device_id.mean()

        # process gpu mem from output metrics and write to result
        if save_result.get(HF_ARG_SKIP_MEMORY_METRIC) == "False":
            save_result[RESULT_FIELD_ALLOCATED_GPU_MEM] = "{:.2f}".format(
                sum(
                    extract_gpu_memory_metrics(
                        self.get_experiment_final_metrics()
                    ).values())
            )
        
        # if there is an error we save the error message else we save the final result
        maybe_error_messages = self.maybe_get_experiment_error_traceback()
        if maybe_error_messages is None:
            other_results = self.get_experiment_final_metrics()
            save_result = {
                **save_result,
                **self.get_experiment_final_metrics(),
            }
        else:
            other_results = {"error_messages": maybe_error_messages}

        # combine the final thing
        save_result = {**save_result, **other_results}

        with open(self.results_filename, "w") as f:
            json.dump(save_result, f, indent=4, sort_keys=True)

    # NOTE: can be improved. Not sure if this really gets parity with
    # subprocess.run
    def write_shell_command(self):

        def _escape(x: str):
            # if there is is whitespace we just escape with single quotes
            # not sure if this is the best thing to do
            return x if not re.search(r"\s", x) else f"'{x}'"

        "Write a shell script to repro the run"
        with open(self.command_filename, "w") as f:
            f.write("#!/bin/bash\n\n")
            for key, val in self.environment.items():
                f.write(f"export {key}={val}\n")
            f.write(" ".join([_escape(x) for x in self.shell_command]))


class DryRunExperiment(Experiment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, run_cmd: str, environment_variables: Dict = None):
        def _dummy(*args, **kwargs):
            pass

        _old = subprocess.run
        subprocess.run = _dummy
        super().run(run_cmd, environment_variables)
        subprocess.run = _old

    def get_experiment_final_metrics(
        self, final_metrics_keys: List[str] = ["train_loss", "train_runtime"]
    ):
        return {}

    def maybe_get_experiment_error_traceback(self):
        return None


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
    experiment_factor = 1
    for k, v in experiment_matrices.items():
        print(f"Experiment has matrix '{k}' of len {len(v)}")
        experiment_factor *= len(v)
    print(f"Experiment matrices will product by factor of '{experiment_factor}'")

    for scenario_config in scenarios:
        _scn_name = scenario_config["name"]
        # if a `run_only_scenarios` list exist, filter out any scenario not in the list
        if args.run_only_scenarios and _scn_name not in args.run_only_scenarios:
            print(f"Skipping scenario '{_scn_name}'")
            continue
        scenario = ScenarioMatrix(scenario_config, acceleration_config_map)
        scenario_matrices, scenario_constants = (
            scenario.get_scenario_matrices_and_defaults()
        )
        scn_factor = 1
        for k, v in scenario_matrices.items():
            print(f"Scenario '{_scn_name}' has matrix '{k}' of len {len(v)}")
            scn_factor *= len(v)

        # update defaults with scenario constants
        constants = {**scenario_constants, **defaults}
        # Remove any empty variables and combine matrices to dictionary to cartesian product on
        combined_matrices = {**scenario_matrices, **experiment_matrices}
        products = ConfigUtils.cartesian_product_on_dict(combined_matrices)
        print(
            f"Scenario '{_scn_name}' will add to the total products by: ----> '{experiment_factor} x {scn_factor}' = '{len(products)}'\n"
        )
        if args.preload_models and len(products) > 0:
            scenario.preload_models()
        for num_gpus, experiment_arg in ConfigUtils.build_args_from_products(
            products, constants
        ):
            yield num_gpus, experiment_arg


def generate_list_of_experiments(
    experiment_args: List[Tuple[int, List]],
    output_dir: str = "results",
    hf_products_dir: str = "hf",
    dry_run: bool = False,
    log_memory_in_trainer: bool = False,
) -> List[Experiment]:
    """Construct list of experiments to be run. Takes in default_config and
    any matrices in scenario and experiment_config
    """
    experiments = []
    for _expr_id, (num_gpus, exp_arg) in enumerate(experiment_args):
        experiment_tag = f"{DIR_PREFIX_EXPERIMENT}_{_expr_id}"
        experiment_output_dir = os.path.join(output_dir, experiment_tag)
        expr_arg_w_outputdir = exp_arg + [
            "--output_dir",
            os.path.join(experiment_output_dir, hf_products_dir),
            "--skip_memory_metrics",
            not log_memory_in_trainer,
        ]
        expr_cls = Experiment if not dry_run else DryRunExperiment
        _expr = expr_cls(
            num_gpus,
            expr_arg_w_outputdir,
            save_dir=experiment_output_dir,
            tag=experiment_tag,
        )
        experiments.append(_expr)
    return experiments


def gather_report(result_dir: Union[str, List[str]], raw: bool = True):

    def _gather(rdir):

        with open(os.path.join(rdir, FILE_SCRIPT_ARGS)) as f:
            script_args = json.load(f)

        # map from config file to tag
        fcm = convert_keypairs_to_map(
            script_args["acceleration_framework_config_keypairs"]
        )
        fcm = {v: k for k, v in fcm.items()}

        experiment_stats = {}
        exper_dirs = [
            x for x in os.listdir(rdir) if x.startswith(DIR_PREFIX_EXPERIMENT)
        ]
        for tag in exper_dirs:
            try:
                with open(os.path.join(rdir, tag, FILE_RESULTS)) as f:
                    tag = tag.replace(DIR_PREFIX_EXPERIMENT + "_", "")
                    tag = int(tag)
                    experiment_stats[tag] = json.load(f)
            except FileNotFoundError:
                pass
        df = pd.DataFrame.from_dict(experiment_stats, orient="index").sort_index()
        try:
            df["framework_config"] = df["acceleration_framework_config_file"].map(
                lambda x: fcm.get(x, "none")
            )
        except KeyError:
            pass

        return df

    if isinstance(result_dir, str):
        df = _gather(result_dir)
    else:
        df = pd.concat([_gather(x) for x in result_dir])

    if raw:
        return df, None

    # certain columns should not be deduped
    def _nunique(series):
        try:
            return pd.Series.nunique(series, dropna=False)
        except:
            # if unique does not work, then return number of non-na
            # elements
            return len(series) - series.isna().sum()

    u = df.apply(_nunique)  # columns that are unique
    return df.loc[:, u != 1], df.iloc[0][u == 1].to_dict()


def compress(df):
    return df.loc[:, df.apply(pd.Series.nunique) != 1]


def main(args):

    # Gathers available gpu device ids that will be used for benchmarking.
    # If "CUDA_VISIBLE_DEVICES" is specified, it will return the specified device ids
    # if no gpu ids are specified, it will default to the enumeration of available ids
    assert torch.cuda.device_count() > 0, "No device detected for memory logging!"
    available_gpus_indices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if available_gpus_indices:
        available_gpus_indices = available_gpus_indices.split(",")
    else:
        available_gpus_indices = [str(i) for i in range(torch.cuda.device_count())]

    if args.dry_run and args.log_nvidia_smi:
        args.log_nvidia_smi = False

    # 1. Prepares a standard BenchmarkDataset
    # TODO: consider caching the json file
    if not args.no_data_processing:
        benchmark_dataset = BenchmarkDataset(args.dataset_name, format_fn)
        benchmark_dataset.save_to_path(args.dataset_save_path)

    # dump out the script arguments
    os.makedirs(args.results_output_path, exist_ok=True)
    with open(os.path.join(args.results_output_path, FILE_SCRIPT_ARGS), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # 2. Prepares a list of experiment arguments from a set of configs
    experiment_args = prepare_arguments(args)

    # 3. Builds a list of experiment objects to run based on the set of experiment arguments
    experiment_stats = {}
    experiment: Experiment
    for experiment in tqdm(
        generate_list_of_experiments(
            experiment_args,
            output_dir=args.results_output_path,
            dry_run=args.dry_run,
            log_memory_in_trainer=args.log_memory_hf,
        )
    ):
        if experiment.num_gpus > 1:
            prefix = COMMAND_ACCELERATE.format(
                accelerate_config_path=args.accelerate_config,
                num_processes=experiment.num_gpus,
                process_port=args.process_port,
            )
        else:
            prefix = COMMAND_PYTHON

        assert experiment.num_gpus <= len(
            available_gpus_indices
        ), "Experiment requires more gpus than is available on the platform."
        """
        Experiment will take only the ids from the available gpu indices, 
        this ensures that whatever GPUs are exposed to benchmark.py are the only 
        devices that each experiment can have access to.
        """
        device_ids = ",".join(available_gpus_indices[: experiment.num_gpus])

        experiment.run(
            f"{prefix} {FMS_TRAINER}",
            environment_variables={"CUDA_VISIBLE_DEVICES": device_ids},
            log_nvidia_smi=args.log_nvidia_smi,
        )

        # write results and store pointers to files
        experiment.write_result()
        experiment.write_shell_command()
        experiment_stats[experiment.tag] = experiment.results_filename

    # 4. Consolidates the experiment results into a summary
    for tag, path in experiment_stats.items():
        with open(path) as f:
            experiment_stats[tag] = json.load(f)
    df = pd.DataFrame.from_dict(experiment_stats, orient="index")
    df.to_csv(os.path.join(args.results_output_path, FILE_SUMMARY_CSV), index=None)

    # TO CREATE THE checked in CSV FILE DO
    # df, constant = gather_report(..., raw=False)
    # try:
    #     errors = df.error_messages
    #     df = df.loc[df.error_messages.isna()]
    # except:
    #     pass
    # df = df.reset_index()
    # df.drop('output_dir', axis=1).reindex(sorted(df.columns), axis=1).to_csv(
    #     'results.csv',
    #     index=False
    # )


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
        default=FRAMEWORK_CONFIG_KEYPAIRS,
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
        default=f"{DIR_BENCHMARKS}/scenarios.yaml",
        help="path to scenarios config file",
    )
    parser.add_argument(
        "--defaults_config_path",
        type=str,
        default=f"{DIR_BENCHMARKS}/defaults.yaml",
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
        default=f"{DIR_BENCHMARKS}/data/cache.json",
        help="dataset cache path",
    )
    parser.add_argument(
        "--accelerate_config",
        type=str,
        default=f"{DIR_BENCHMARKS}/accelerate.yaml",
        help="accelerate config file path",
    )
    parser.add_argument(
        "--results_output_path",
        type=str,
        default=DIR_NAME_RESULTS_DEFAULT,
        help="accelerate config file path",
    )
    parser.add_argument(
        "--process_port", type=int, default=29500, help="accelerate process port"
    )
    parser.add_argument(
        "--no_data_processing",
        action="store_true",
        help="skip the json data prep (useful for re-runs)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="perform a dry run only. Useful for debuging benchmark scenarios.",
    )
    parser.add_argument(
        "--preload_models",
        action="store_true",
        help="ensures 'model_name_or_paths 'specified in scenarios.yaml work. "
        "Useful to check model paths specified correctly before lengthly benchmark runs.",
    )
    parser.add_argument(
        "--log_nvidia_smi",
        action="store_true",
        help="Use `nvidia-smi` API to log reserved memory of benchmarks",
    )

    parser.add_argument(
        "--log_memory_hf",
        action="store_true",
        help="Uses memory logging from HF Trainer Arguments API to log gpu memory, for distributed runs only rank 0 is measured",
    )

    args = parser.parse_args()
    main(args)
