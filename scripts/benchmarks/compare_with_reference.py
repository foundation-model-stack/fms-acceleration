import argparse
import pandas as pd
from copy import copy
import matplotlib.pyplot as plt
import os
from numpy import linalg

# default columns to compare
DEFAULT_PLOT_COLUMNS = ["mem_torch_mem_alloc_in_bytes", "mem_peak_torch_mem_alloc_in_bytes", "train_loss", "train_tokens_per_second"]     
# Used as combined identifier of experiment
DEFAULT_INDICES = ["framework_config", "peft_method", "model_name_or_path", "num_gpus", "per_device_train_batch_size"]
DEFAULT_OUTLIERS_DF_COLUMN_NAMES = ["scenario", *DEFAULT_INDICES, "reference", "new"]
DEFAULT_REFERENCE_FILEPATH = "scripts/benchmarks/refs/a100_80gb.csv"
BENCHMARK_FILENAME = "benchmarks.csv"

def plot_chart(ax, x, y, title, xlabel, ylabel):
    ax.scatter(x, y, s=10)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axline((0, 0), slope=1)
   
def compare_results(df, ref, plot_columns, num_columns = 2, threshold_ratio=0.1):
    num_plots = len(plot_columns)    
   
    charts = []
    total_outliers = []
    # filter ref to only those rows seen in df
    ref = ref[ref.index.isin(df.index.tolist())]
    for idx in range(num_plots):
        _, ax = plt.subplots(figsize=(8, 8))
        column = plot_columns[idx]
        assert (
            (column in ref.columns)
            and 
            (column in df.columns)
        ), f"Column Name `{column}` not in Dataframe"        

        ref_series = ref[column].fillna(0)
        df_series = df[column].fillna(0)
        # Extract outliers base on some threshold % difference on referance
        ds = abs(df_series-ref_series)/(ref_series+1e-9)    
        outliers = ds.index[ds>threshold_ratio].to_list()        
        plot_chart(
            ax, 
            ref_series, 
            df_series, 
            title=f"Metric: {column}", 
            xlabel="Reference", 
            ylabel="New",
        )        
        charts.append((ax, f"compare-{column}.jpg"))
        total_outliers += [
            [column, *outlier, ref_series[outlier].item(), df_series[outlier].item()]
            for outlier in outliers
        ]
    return total_outliers, charts

def read_df(file_path, indices, plot_columns):
    df = pd.read_csv(file_path)
    df.set_index(indices, inplace=True)
    df = df[plot_columns]
    return df

def main(result_dir, reference_benchmark_filepath, plot_columns):
    ref = read_df(reference_benchmark_filepath, DEFAULT_INDICES, plot_columns)
    df = read_df(os.path.join(result_dir, BENCHMARK_FILENAME), DEFAULT_INDICES, plot_columns)
    total_outliers, charts = compare_results(df, ref, plot_columns, threshold_ratio=.1)
    outliers_df = pd.DataFrame(total_outliers, columns=DEFAULT_OUTLIERS_DF_COLUMN_NAMES)
    outliers_df.to_csv(os.path.join(result_dir, "outliers.csv"), index=None)
    for chart, filename in charts:
        chart.figure.savefig(os.path.join(result_dir, filename))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Acceleration Benchmarking Debug Tool",
        description="This script analyses benchmark outputs against the current reference",
    )
    parser.add_argument(
        "--result_dir",
        default="benchmark_outputs",
        help="benchmark result directory to use for comparison",
    )

    parser.add_argument(
        "--reference_benchmark_filepath",
        default="scripts/benchmarks/refs/a100_80gb.csv",
        help="file path of the csv to compare on",
    )

    parser.add_argument(
        "--plot_columns",
         default=DEFAULT_PLOT_COLUMNS,
         nargs='+'
    )

    args = parser.parse_args()
    main(
        result_dir=args.result_dir,
        reference_benchmark_filepath=args.reference_benchmark_filepath,
        plot_columns=args.plot_columns,
    )
