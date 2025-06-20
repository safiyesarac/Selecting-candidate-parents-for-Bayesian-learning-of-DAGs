import os
import pandas as pd
import matplotlib.pyplot as plt

files = {
    "Asia (8 Nodes)": "/home/gulce/Downloads/thesis/data/coverage/asia/asia_coverage_results_1000.csv",
     "Sachs (11 Nodes)": "/home/gulce/Downloads/thesis/data/coverage/sachs/sachs_coverage_results_1000.csv",
    "Credit (12 Nodes)": "/home/gulce/Downloads/thesis/data/coverage/credit/credit_coverage_results_1000.csv",
    "EngineFuel (9 Nodes)": "/home/gulce/Downloads/thesis/data/coverage/enginefuel/enginefuel_coverage_results_1000.csv"
}

datasets = {}
for dataset_name, path in files.items():
    datasets[dataset_name] = pd.read_csv(path)

all_algorithms = sorted(
    set().union(*(df["Algorithm"].unique() for df in datasets.values()))
)

output_folder = "/home/gulce/Downloads/thesis/data/coverage/nodenumber/1000"
os.makedirs(output_folder, exist_ok=True)

for alg in all_algorithms:
    plt.figure(figsize=(8, 5))
    
    for dataset_name, df in datasets.items():
        subset = df[df["Algorithm"] == alg]
        
        
        coverage_by_k = subset.groupby("K")["CoverageFraction"].mean().sort_index()
        
        
        real_k = coverage_by_k.index   
        coverage_values = coverage_by_k.values
        
        plt.plot(real_k, coverage_values, marker='o', label=dataset_name)
    
    plt.title(f"Coverage vs. K: {alg}\n(1000 Data Points)")
    plt.xlabel("Parent-Set Size")
    plt.ylabel("Coverage Fraction")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    save_path = os.path.join(output_folder, f"{alg}_coverage_shifted.png")
    plt.savefig(save_path)
    plt.close()
