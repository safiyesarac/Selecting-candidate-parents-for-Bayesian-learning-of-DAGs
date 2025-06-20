import os
import pandas as pd
import matplotlib.pyplot as plt


os.makedirs("plots", exist_ok=True)


files = {
    
    
    200:  "/home/gulce/Downloads/thesis/data/coverage/survey/survey_coverage_results_200.csv",
    1000:  "/home/gulce/Downloads/thesis/data/coverage/survey/survey_coverage_results_1000.csv",
    10000: "/home/gulce/Downloads/thesis/data/coverage/survey/survey_coverage_results_10000.csv"
}


data_by_size = {}
for sample_size, path in files.items():
    data_by_size[sample_size] = pd.read_csv(path)


all_algs = set()
for sample_size, df in data_by_size.items():
    all_algs.update(df["Algorithm"].unique())
all_algs = sorted(all_algs)  


for alg in all_algs:
    plt.figure()  
    
    
    for sample_size, df in data_by_size.items():
        
        subset = df[df["Algorithm"] == alg]
        
        
        coverage_by_k = subset.groupby("K")["CoverageFraction"].mean().sort_index()
        
        
        plt.plot(coverage_by_k.index, coverage_by_k.values, marker='o', label=f"N={sample_size}")
    
    
    plt.legend()
    plt.xlabel("Max Parent-Set Size (K)")
    plt.ylabel("Coverage Fraction")
    plt.title(f"Coverage vs K: {alg}")
    plt.ylim(0, 1.05)  
    
    
    plt.savefig(f"/home/gulce/Downloads/thesis/data/coverage/survey/datapointsplots/{alg}_coverage_plot.png")
    
    
    plt.close()
