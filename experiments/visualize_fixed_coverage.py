
import argparse
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_directory_exists(filepath):
    """Creates the directory for the filepath if it doesn't exist."""
    directory = os.path.dirname(filepath)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def main():
    parser = argparse.ArgumentParser(
        description="Visualize CSV data as a heatmap of CoverageFraction by Algorithm and K."
    )
    parser.add_argument("csv_file", help="Path to the CSV file.")
    parser.add_argument("--metric", default="CoverageFraction",
                        help="CSV column to use as the heatmap values (default: CoverageFraction).")
    parser.add_argument("--xcol", default="K",
                        help="CSV column for the x-axis (default: K).")
    parser.add_argument("--ycol", default="Algorithm",
                        help="CSV column for the y-axis (default: Algorithm).")
    parser.add_argument("--save", type=str,
                        help="Optional path (including filename) to save the heatmap image (e.g., heatmap.png).")
    parser.add_argument(
        "--k", type=float,             
        help=(
            "The single K value you want to visualise. "
            "If omitted, the script exits without plotting."
        )
    )
    args = parser.parse_args()

    
    df = pd.read_csv(args.csv_file)
    
    df[args.xcol] = pd.to_numeric(df[args.xcol], errors="coerce")
    df = df.dropna(subset=[args.xcol])

    
    if args.k is not None:
        df = df[df[args.xcol] >= args.k]
        if df.empty:
            print(f"No rows found with {args.xcol} ≥ {args.k}. Exiting.", file=sys.stderr)
            sys.exit(0)
    
    
    pivot_df = df.pivot_table(index=args.ycol, columns=args.xcol, values=args.metric, aggfunc="mean")
    
    
    pivot_df = pivot_df.sort_index(axis=1)

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="YlGnBu",annot_kws={"fontsize": 8} )
    
    plt.xlabel(args.xcol)
    plt.ylabel(args.ycol)
    plt.tight_layout()
    
    if args.save:
        ensure_directory_exists(args.save)
        plt.savefig(args.save)
        print(f"Heatmap saved as {args.save}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
