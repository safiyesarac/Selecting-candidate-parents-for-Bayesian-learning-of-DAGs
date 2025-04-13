import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
csv_path = "/home/gulce/Downloads/thesis/data/fair/coverage_log_nVars=10_maxInDeg=4_alpha=1.0_smallData=5000_refData=50000_means.csv"
df = pd.read_csv(csv_path)
import matplotlib.colors as mcolors
# Pivot data for coverage_prior and coverage_gobnilp
pivot_prior = df.pivot(index='method', columns='K', values='coverage_prior')
pivot_gobnilp = df.pivot(index='method', columns='K', values='coverage_gobnilp')

# Generate filenames by replacing .csv with _prior.png and _gobnilp.png
base_filename = csv_path.replace('.csv', '')
output_prior_path = f"{base_filename}_prior.png"
output_gobnilp_path = f"{base_filename}_gobnilp.png"
# Here, we assume coverage ranges from 0 to 1. Adjust if your min/max coverage differ.
norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
# Heatmap for Coverage w.r.t. Prior DAG
plt.figure(figsize=(16, 12))
sns.heatmap(pivot_prior, annot=True, fmt=".3f", cmap="YlGnBu", annot_kws={"fontsize": 12})
plt.title("Coverage w.r.t. Prior DAG", fontsize=18)
plt.xlabel("K", fontsize=16)
plt.ylabel("Algorithm", fontsize=16)
plt.tight_layout()
plt.savefig(output_prior_path, dpi=300)
plt.show()

# Heatmap for Coverage w.r.t. Gobnilp DAG
plt.figure(figsize=(16, 12))
sns.heatmap(pivot_gobnilp, annot=True, fmt=".3f", cmap="YlGnBu", annot_kws={"fontsize": 12})
plt.title("Coverage w.r.t. Gobnilp DAG", fontsize=18)
plt.xlabel("K", fontsize=16)
plt.ylabel("Algorithm", fontsize=16)
plt.tight_layout()
plt.savefig(output_gobnilp_path, dpi=300)
plt.show()
