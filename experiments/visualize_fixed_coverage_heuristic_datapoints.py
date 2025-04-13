import os
import pandas as pd
import matplotlib.pyplot as plt

# Create an output directory (if it doesn't exist)
os.makedirs("plots", exist_ok=True)

# Paths to the CSV files by sample size
files = {
    50:   "/home/gulce/Downloads/thesis/data/coverage/enginefuel/enginefuel_coverage_results_50.csv",
    100:  "/home/gulce/Downloads/thesis/data/coverage/enginefuel/enginefuel_coverage_results_100.csv",
    200:  "/home/gulce/Downloads/thesis/data/coverage/enginefuel/enginefuel_coverage_results_200.csv",
    1000:  "/home/gulce/Downloads/thesis/data/coverage/enginefuel/enginefuel_coverage_results_1000.csv",
    10000: "/home/gulce/Downloads/thesis/data/coverage/enginefuel/enginefuel_coverage_results_10000.csv"
}

# Read each file into a dictionary keyed by sample size
data_by_size = {}
for sample_size, path in files.items():
    data_by_size[sample_size] = pd.read_csv(path)

# Collect all unique algorithm names across all files
all_algs = set()
for sample_size, df in data_by_size.items():
    all_algs.update(df["Algorithm"].unique())
all_algs = sorted(all_algs)  # sort them for consistency

# Create one figure per algorithm, then save it
for alg in all_algs:
    plt.figure()  # new figure for this algorithm
    
    # Plot coverage vs K for each data size
    for sample_size, df in data_by_size.items():
        # Filter rows for this algorithm
        subset = df[df["Algorithm"] == alg]
        
        # Group by K in case multiple runs exist, then sort by K
        coverage_by_k = subset.groupby("K")["CoverageFraction"].mean().sort_index()
        
        # Plot with a marker so we can see each point
        plt.plot(coverage_by_k.index, coverage_by_k.values, marker='o', label=f"N={sample_size}")
    
    # Configure legend, labels, title
    plt.legend()
    plt.xlabel("Max Parent-Set Size (K)")
    plt.ylabel("Coverage Fraction")
    plt.title(f"Coverage vs K: {alg}")
    plt.ylim(0, 1.05)  # coverage fraction is between 0 and 1
    
    # Save plot to the "plots" directory (e.g., "plots/greedy_coverage_plot.png")
    plt.savefig(f"/home/gulce/Downloads/thesis/data/coverage/enginefuel/datapointsplots/{alg}_coverage_plot.png")
    
    # Close the figure to free memory (avoids piling up many open figures)
    plt.close()
