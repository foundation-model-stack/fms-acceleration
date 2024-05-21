# Standard
import argparse

# First Party
# import this because of alot of internal contants
from scripts.benchmarks.benchmark import gather_report


def main(*directories: str, output_filename: str = "results.csv"):
    "gather outputs from a list of directories and output to a csv"

    df, constant = gather_report(*directories, raw=False)
    errors = []
    try:
        # remove error messages if any
        errors = df.error_messages
        errors = errors.loc[errors.isna() == False]
        df = df.loc[df.error_messages.isna()]
    except:
        pass
    df = df.reset_index().drop("output_dir", axis=1)
    df.reindex(sorted(df.columns), axis=1).to_csv(output_filename, index=False)
    print("***************** Report Created ******************")
    print(f"Total lines: '{len(df)}'")
    print(f"Number columns included: '{len(df.columns)}'")
    print(f"Number columns excluded: '{len(constant)}'")
    print(f"Excluding number of exceptions caught: '{len(errors)}'")
    print(f"Written report to '{output_filename}'")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="Acceleration Benchmarking Reporting Tool",
        description="This script gathers a set benchmarks to produce a CSV report",
    )
    parser.add_argument(
        "bench_outputs",
        nargs="+",
        help="list of directories from which to gather bench outputs.",
    )
    parser.add_argument(
        "--result_file",
        default="results.csv",
        help="name of final csv report file.",
    )
    args = parser.parse_args()
    main(args.bench_outputs, output_filename=args.result_file)
