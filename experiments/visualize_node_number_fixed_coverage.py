import os
import pandas as pd
import matplotlib.pyplot as plt

files = {
    "Asia (8 Nodes)": "/home/gulce/Downloads/thesis/data/coverage/asia/asia_coverage_results_10000.csv",
    # "Sachs (11 Nodes)": "/home/gulce/Downloads/thesis/data/coverage/sachs/sachs_coverage_results_10000.csv",
    "Credit (12 Nodes)": "/home/gulce/Downloads/thesis/data/coverage/credit/credit_coverage_results_10000.csv",
    "EngineFuel (9 Nodes)": "/home/gulce/Downloads/thesis/data/coverage/enginefuel/enginefuel_coverage_results_10000.csv"
}

datasets = {}
for dataset_name, path in files.items():
    datasets[dataset_name] = pd.read_csv(path)

all_algorithms = sorted(
    set().union(*(df["Algorithm"].unique() for df in datasets.values()))
)

output_folder = "/home/gulce/Downloads/thesis/data/coverage/nodenumber/10000"
os.makedirs(output_folder, exist_ok=True)

for alg in all_algorithms:
    plt.figure(figsize=(8, 5))
    
    for dataset_name, df in datasets.items():
        subset = df[df["Algorithm"] == alg]
        
        # Group coverage by the "K" in the CSV...
        coverage_by_k = subset.groupby("K")["CoverageFraction"].mean().sort_index()
        
        # But we know "K" means "K+1" in reality, so shift x-values by +1
        real_k = coverage_by_k.index   # <--- shift by 1
        coverage_values = coverage_by_k.values
        
        plt.plot(real_k, coverage_values, marker='o', label=dataset_name)
    
    plt.title(f"Coverage vs. (K+1): {alg}\n(Exact Parents, 10000 Data Points)")
    plt.xlabel("Exact Parent-Set Size = (CSV K) + 1")
    plt.ylabel("Coverage Fraction")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(output_folder, f"{alg}_coverage_shifted.png")
    plt.savefig(save_path)
    plt.close()
