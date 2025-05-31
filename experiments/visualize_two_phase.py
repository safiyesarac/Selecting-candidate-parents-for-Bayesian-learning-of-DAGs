import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
csv_path = "/home/gulce/Downloads/thesis/data/fair/coverage_log_nVars=10_maxInDeg=4_alpha=1.0_smallData=5000_refData=50000_means.csv"
df = pd.read_csv(csv_path)
import matplotlib.colors as mcolors
# Pivot data for coverage_prior and coverage_gobnilp
pivot_prior = df.pivot(index='method', columns='K', values='coverage_prior')


# Generate filenames by replacing .csv with _prior.png and _gobnilp.png
base_filename = csv_path.replace('.csv', '')
output_prior_path = f"{base_filename}_prior_2f.png"

# Here, we assume coverage ranges from 0 to 1. Adjust if your min/max coverage differ.
norm = mcolors.Normalize(vmin=0.5, vmax=1.0)
# Heatmap for Coverage w.r.t. Prior DAG
ax = sns.heatmap(
        pivot_prior,
        annot=True, fmt=".2f",
        cmap="YlGnBu",
        annot_kws={"fontsize": 6}
)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)   # ← shrink algorithm names
plt.title("Arc Coverage w.r.t. G*", fontsize=12)
plt.xlabel("K", fontsize=12)
plt.ylabel("Algorithm", fontsize=12)
plt.tight_layout()
plt.savefig(output_prior_path, dpi=300)
plt.show()

