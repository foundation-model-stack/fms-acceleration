import argparse
import pandas as pd
from copy import copy
import matplotlib.pyplot as plt
import os
from numpy import linalg

PLOT_COLUMNS = ["mem_torch_mem_alloc_in_bytes", "mem_peak_torch_mem_alloc_in_bytes", "train_loss", "train_tokens_per_second"]     
INDICES = ["framework_config", "peft_method", "model_name_or_path", "num_gpus", "per_device_train_batch_size"]
REFERENCE_FILEPATH = "scripts/benchmarks/refs/a100_80gb.csv"
BENCHMARK_FILENAME = "benchmarks.csv"
FIGURE_FILENAME = "comparison.jpg"

def plot_chart(ax, x, y, title, xlabel, ylabel):
    ax.scatter(x, y, s=10)
    ax.plot()
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axline((0, 0), slope=1)

def plot_table(ax, cell_inputs, title, col_widths, col_labels):
    table = ax.table(cellText=cell_inputs, loc="center", colWidths=col_widths, colLabels=col_labels)
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(0)            
    ax.set_title(title, fontsize=10)
    
def compare_results(df, ref, plot_columns, num_columns = 2, threshold_ratio=0.1):
    num_plots = len(plot_columns)    
    rows = num_plots
    fig, axs = plt.subplots(rows, 2, figsize=(20, 28))
    fig.tight_layout(pad=5.0)
    
    # filter ref to only those rows seen in df
    ref = ref[ref.index.isin(df.index.tolist())]
    for idx in range(num_plots):
        column = plot_columns[idx]
        assert (column in ref.columns) and (column in df.columns), f"Column Name `{column}` not in Dataframe"
        ax1 = axs[idx][0]
        ax2 = axs[idx][1]
        ax2.axis('off')
        
        ref_series = ref[column].fillna(0)
        df_series = df[column].fillna(0)
        # Calculate difference of l1 norms as a percentage on reference
        ref_norm = linalg.norm(ref_series, ord=1)
        df_norm = linalg.norm(df_series, ord=1)
        norm_difference = abs(df_norm - ref_norm)/(ref_norm+1e-9)
        # Extract outliers from reference based on % threshold on referance
        ds = abs(df_series-ref_series)/(ref_series+1e-9)    
        outliers = ds.index[ds>threshold_ratio].to_list()
        
        plot_chart(
            ax1, 
            ref_series, 
            df_series, 
            title=f"Metric: {column}", 
            xlabel="Reference", 
            ylabel="New",
        )
        
        cell_inputs = [[outlier, ref_series[outlier], df_series[outlier]] for outlier in outliers] if len(outliers)>0 else [["","",""]]

        plot_table(
            ax2,
            cell_inputs = cell_inputs,
            title=f"Metric: {column} outliers\n\nNorm Difference={norm_difference:.3f}", 
            col_widths=[0.9, 0.2, 0.2], 
            col_labels=["Experiment", "Reference", "New"]
        )                
    return fig

def read_df(file_path, indices, plot_columns):
    df = pd.read_csv(file_path)
    df.set_index(indices, inplace=True)
    df = df[plot_columns]
    return df

def main(result_dir):
    ref = read_df(REFERENCE_FILEPATH, INDICES, PLOT_COLUMNS)
    df = read_df(os.path.join(result_dir, BENCHMARK_FILENAME), INDICES, PLOT_COLUMNS)
    fig = compare_results(df, ref, PLOT_COLUMNS, threshold_ratio=.1)
    plt.savefig(os.path.join(result_dir, FIGURE_FILENAME))    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Acceleration Benchmarking Debug Tool",
        description="This script analyses benchmark outputs against the current reference",
    )
    parser.add_argument(
        "--result_dir",
        default="benchmark_outputs",
        help="benchmark result directory",
    )

    args = parser.parse_args()
    main(
        result_dir=args.result_dir,
    )