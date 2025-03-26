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
    if sort_metric not in df.columns:
        print(f"[WARNING] sort_metric='{sort_metric}' not in columns. Skipping top_n filtering.")
        return df

    df_k = df[df['K'] == k_val].copy()
    if df_k.empty:
        print(f"[WARNING] No rows found where K={k_val}. Skipping top_n filtering.")
        return df

    df_k.sort_values(by=sort_metric, ascending=False, inplace=True)
    top_algos = df_k['Algorithm'].head(n).unique()
    print(f"[INFO] top_{n} algos by '{sort_metric}' at K={k_val}:", top_algos)

    return df[df['Algorithm'].isin(top_algos)].copy()


# --------------------------------------------------------------------------
# 2. Plot Approach A: Line Plot
# --------------------------------------------------------------------------
def plot_line(df, metric="Average Parent Coverage", xcol="K", save=None):
    df_sorted = df.copy()
    try:
        df_sorted[xcol] = df_sorted[xcol].astype(float)
    except ValueError:
        pass

    df_sorted.sort_values(by=xcol, inplace=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_sorted, x=xcol, y=metric, hue="Algorithm", marker="o")
    plt.title(f"Line Plot: {metric} vs. {xcol}")
    plt.tight_layout()
    
    if save:
        plt.savefig(save.replace('.csv','_line.png'))
        print(f"Plot saved as {save}")
    else:
        plt.show()


# --------------------------------------------------------------------------
# 3. Plot Approach B: Grouped Bar Chart
# --------------------------------------------------------------------------
def plot_bar(df, metric="Average Parent Coverage", xcol="K", save=None):
    df_bar = df.copy()
    try:
        df_bar[xcol] = df_bar[xcol].astype(float)
    except ValueError:
        pass

    df_bar.sort_values(by=xcol, inplace=True)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df_bar, x=xcol, y=metric, hue="Algorithm", ci=None, edgecolor="black", width=0.8)
    plt.title(f"Grouped Bar: {metric} by {xcol} and Algorithm")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    plt.tight_layout()

    if save:
        plt.savefig(save.replace('.csv','_bar.png'))
        print(f"Plot saved as {save}")
    else:
        plt.show()


# --------------------------------------------------------------------------
# 4. Plot Approach C: Facet Subplots
# --------------------------------------------------------------------------
def plot_facet(df, metric="Average Parent Coverage", xcol="K", save=None):
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

    if save:
        plt.savefig(save.replace('.csv','_facet.png'))
        print(f"Plot saved as {save}")
    else:
        plt.show()


# --------------------------------------------------------------------------
# 5. Plot Approach D: Heatmap
# --------------------------------------------------------------------------
def plot_heatmap(df, metric="Average Parent Coverage", xcol="K",
                 heatmap_figsize=(30,24), save=None):
    """
    Creates a pivot table (Algorithm vs. K) of <metric> and displays
    as a heatmap, with a default figure size of (30,24).
    """
    df_hm = df.copy()
    try:
        df_hm[xcol] = df_hm[xcol].astype(float)
    except ValueError:
        pass

    pivoted = df_hm.pivot_table(index="Algorithm", columns=xcol, values=metric, aggfunc="mean")
    # Sort columns by ascending K
    pivoted = pivoted.reindex(sorted(pivoted.columns), axis=1)

    plt.figure(figsize=heatmap_figsize)
    sns.heatmap(pivoted, annot=True, fmt=".4f", cmap="YlGnBu",   annot_kws={"fontsize": 14} )
    plt.title(f"Heatmap of {metric} by Algorithm vs. {xcol}")
    plt.ylabel("Algorithm")
    plt.xlabel(xcol)
    plt.tight_layout()

    if save:
        outpath = save.replace('.csv','_heatmap.png')
        plt.savefig(outpath)
        print(f"[INFO] Heatmap plot saved as {outpath}")
    else:
        plt.show()


# --------------------------------------------------------------------------
# 6. Main Script
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Advanced coverage visualization with multiple plot types.")
    parser.add_argument("csv_file", help="Path to the coverage log CSV.")
    parser.add_argument("--plot_type", choices=["line", "bar", "facet", "heatmap"], default="bar", help="Choose one of: line, bar, facet, heatmap.")
    parser.add_argument("--metric", default="Average Parent Coverage", help="Which column to plot on the y-axis.")
    parser.add_argument("--k_col", default="K", help="Which column is used for the x-axis (default: K).")
    parser.add_argument("--sort_metric", default=None, help="If given, we pick top_n algorithms by this metric at K=k_max.")
    parser.add_argument("--k_max", type=float, default=5, help="Which K to use when picking top_n. Must be numeric. If not set, skip top_n.")
    parser.add_argument("--top_n", type=int, default=0, help="If >0, filter to top_n algorithms by --sort_metric at --k_max.")
    parser.add_argument("--exclude_consensus", action="store_true", help="If set, remove rows where Algorithm contains 'Consensus Parents'.")
    parser.add_argument("--save", type=str, help="Path to save the plot (e.g., 'plot.png').")

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

   

    plot_line(df, metric=args.metric, xcol=args.k_col, save=args.csv_file)

    plot_bar(df, metric=args.metric, xcol=args.k_col, save=args.csv_file)

    plot_facet(df, metric=args.metric, xcol=args.k_col, save=args.csv_file)

    plot_heatmap(df, metric=args.metric, xcol=args.k_col, save=args.csv_file)


if __name__ == "__main__":
    main()
