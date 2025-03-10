#!/usr/bin/env python3

"""
advanced_visualize.py

Usage Example:
  python advanced_visualize.py /path/to/consensus_log.csv \
      --plot_type bar \
      --metric "Average Parent Coverage" \
      --sort_metric "F1" \
      --k_max 9 \
      --top_n 5 \
      --exclude_consensus

What This Script Does:
  1) Loads a coverage CSV file that must have columns like:
       - "Algorithm"
       - "K"
       - coverage/performance metrics (e.g. "Exact Coverage", "Average Parent Coverage", "F1", etc.)
  2) Optionally filters out all but the top N algorithms by some metric at a certain K (use --sort_metric, --k_max, --top_n).
  3) Optionally excludes rows whose "Algorithm" contains "Consensus Parents" (use --exclude_consensus).
  4) Lets you choose one of four plotting approaches with --plot_type:
       "line"    -> standard line plot (many lines can clutter)
       "bar"     -> grouped bar chart (legend outside, rotated x ticks)
       "facet"   -> multiple subplots (one subplot per Algorithm)
       "heatmap" -> color-coded matrix for (Algorithm, K) vs. metric
  5) Displays the plot interactively on screen.

Install Requirements:
  pip install pandas matplotlib seaborn

Example calls:
  1) Basic bar chart of Average Parent Coverage:
       python advanced_visualize.py my_results.csv --plot_type bar --metric "Average Parent Coverage"

  2) Heatmap of F1, skipping "Consensus Parents":
       python advanced_visualize.py my_results.csv --plot_type heatmap --metric "F1" --exclude_consensus

  3) Show only top 5 algorithms by F1 at K=9, in a line plot:
       python advanced_visualize.py my_results.csv --plot_type line --metric "Exact Coverage" \
           --sort_metric "F1" --k_max 9 --top_n 5
"""

import sys
import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------------------------------
# 1. Optional Filtering Helper: top_n_algorithms
# --------------------------------------------------------------------------
def top_n_algorithms(df, sort_metric="F1", k_val=9, n=5):
    """
    Returns a *new* DataFrame containing only the top n algorithms
    by `sort_metric` at K == k_val.

    If the sort_metric isn't in df.columns, skip filtering.
    If there's no row with K==k_val, skip filtering.
    """
    if sort_metric not in df.columns:
        print(f"[WARNING] sort_metric='{sort_metric}' not in columns. Skipping top_n filtering.")
        return df

    df_k = df[df['K'] == k_val].copy()
    if df_k.empty:
        print(f"[WARNING] No rows found where K={k_val}. Skipping top_n filtering.")
        return df

    # Sort descending by sort_metric
    df_k.sort_values(by=sort_metric, ascending=False, inplace=True)

    top_algos = df_k['Algorithm'].head(n).unique()
    print(f"[INFO] top_{n} algos by '{sort_metric}' at K={k_val}:", top_algos)

    return df[df['Algorithm'].isin(top_algos)].copy()


# --------------------------------------------------------------------------
# 2. Plot Approach A: Line Plot
# --------------------------------------------------------------------------
def plot_line(df, metric="Average Parent Coverage", xcol="K"):
    """
    Basic line plot: x=K, y=metric, color/hue=Algorithm.
    This can become cluttered if you have many algorithms or crossing lines.
    """
    df_sorted = df.copy()
    try:
        df_sorted[xcol] = df_sorted[xcol].astype(float)
    except ValueError:
        pass

    df_sorted.sort_values(by=xcol, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_sorted,
        x=xcol, y=metric,
        hue="Algorithm",
        marker="o"
    )
    plt.title(f"Line Plot: {metric} vs. {xcol}")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# 3. Plot Approach B: Grouped Bar Chart (Legend Outside, Rotated X Ticks)
# --------------------------------------------------------------------------
def plot_bar(df, metric="Average Parent Coverage", xcol="K"):
    """
    Grouped bar chart: x=K, y=metric, hue=Algorithm => side-by-side bars.
    - Legend is placed outside on the right.
    - x-axis labels are rotated 45 degrees for clarity.
    - Bars have a bit of edgecolor for distinction.
    """
    df_bar = df.copy()
    try:
        df_bar[xcol] = df_bar[xcol].astype(float)
    except ValueError:
        pass

    df_bar.sort_values(by=xcol, inplace=True)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df_bar,
        x=xcol,
        y=metric,
        hue="Algorithm",
        ci=None,
        edgecolor="black",
        width=0.8  # adjust bar width to reduce clutter
    )
    plt.title(f"Grouped Bar: {metric} by {xcol} and Algorithm")

    # Rotate x-ticks to avoid overlap
    plt.xticks(rotation=45)

    # Place legend outside on the right
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# 4. Plot Approach C: Facet Subplots (one subplot per Algorithm)
# --------------------------------------------------------------------------
def plot_facet(df, metric="Average Parent Coverage", xcol="K"):
    """
    Creates multiple subplots, one per algorithm (col_wrap=4).
    Each subplot: metric vs. K. No crossing lines, each algorithm is separate.
    """
    df_facet = df.copy()
    try:
        df_facet[xcol] = df_facet[xcol].astype(float)
    except ValueError:
        pass

    g = sns.FacetGrid(df_facet, col="Algorithm", col_wrap=4, sharey=False, height=3)
    g.map_dataframe(sns.lineplot, x=xcol, y=metric, marker="o")
    g.set_titles(col_template="{col_name}")
    g.fig.suptitle(f"{metric} vs. {xcol} by Algorithm", y=1.05)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# 5. Plot Approach D: Heatmap
# --------------------------------------------------------------------------
def plot_heatmap(df, metric="Average Parent Coverage", xcol="K"):
    """
    Heatmap: rows=Algorithm, columns=K, color=metric value.
    Very compact. Good for large sets of algorithms/K.
    """
    df_hm = df.copy()
    try:
        df_hm[xcol] = df_hm[xcol].astype(float)
    except ValueError:
        pass

    pivoted = df_hm.pivot_table(index="Algorithm", columns=xcol, values=metric, aggfunc="mean")

    pivoted = pivoted.reindex(sorted(pivoted.columns), axis=1)

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivoted, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(f"Heatmap of {metric} by Algorithm vs. {xcol}")
    plt.ylabel("Algorithm")
    plt.xlabel(xcol)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
# 6. Main Script
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Advanced coverage visualization with multiple plot types."
    )
    parser.add_argument("csv_file", help="Path to the coverage log CSV.")
    parser.add_argument("--plot_type", choices=["line", "bar", "facet", "heatmap"],
                        default="bar",
                        help="Choose one of: line, bar, facet, heatmap.")
    parser.add_argument("--metric", default="Average Parent Coverage",
                        help="Which column to plot on the y-axis.")
    parser.add_argument("--k_col", default="K",
                        help="Which column is used for the x-axis (default: K).")
    parser.add_argument("--sort_metric", default=None,
                        help="If given, we pick top_n algorithms by this metric at K=k_max.")
    parser.add_argument("--k_max", type=float, default=5,
                        help="Which K to use when picking top_n. Must be numeric. If not set, skip top_n.")
    parser.add_argument("--top_n", type=int, default=0,
                        help="If >0, filter to top_n algorithms by --sort_metric at --k_max.")
    parser.add_argument("--exclude_consensus", action="store_true",
                        help="If set, remove rows where Algorithm contains 'Consensus Parents'.")

    args = parser.parse_args()

    # Load CSV
    if not os.path.isfile(args.csv_file):
        print(f"[ERROR] CSV file not found: {args.csv_file}")
        sys.exit(1)

    df = pd.read_csv(args.csv_file)
    print(f"[INFO] Loaded {df.shape[0]} rows from {args.csv_file}")
    print(f"[INFO] Columns: {df.columns.tolist()}")

    # Optionally exclude lines with "Consensus Parents"
    if args.exclude_consensus:
        before_count = df.shape[0]
        df = df[~df["Algorithm"].str.contains("Consensus Parents", na=False)].copy()
        after_count = df.shape[0]
        print(f"[INFO] Excluded consensus lines. Rows from {before_count} to {after_count}.")

    # top_n filtering if requested
    df = top_n_algorithms(df, sort_metric=args.sort_metric, k_val=args.k_max, n=5)

    
    plot_line(df, metric=args.metric, xcol=args.k_col)
    
    plot_bar(df, metric=args.metric, xcol=args.k_col)
    
    plot_facet(df, metric=args.metric, xcol=args.k_col)
   
    plot_heatmap(df, metric=args.metric, xcol=args.k_col)
    

if __name__ == "__main__":
    main()
