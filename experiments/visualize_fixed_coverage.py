#!/usr/bin/env python3
import argparse
import os
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
    args = parser.parse_args()

    # Load the CSV file
    df = pd.read_csv(args.csv_file)
    
    # Pivot the DataFrame to have 'Algorithm' as rows and 'K' as columns with values from 'CoverageFraction'
    pivot_df = df.pivot_table(index=args.ycol, columns=args.xcol, values=args.metric, aggfunc="mean")
    
    # Sort the columns (K values) if they are numeric
    pivot_df = pivot_df.sort_index(axis=1)

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_df, annot=True, fmt=".4f", cmap="YlGnBu")
    plt.title(f"Heatmap of {args.metric} by {args.ycol} vs. {args.xcol}")
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
