
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Paths relative to the repository root
results = Path("task3/results")
csv_path = results / "fanout_full_epoch" / "fanout_full_epoch_summary.csv"
memory_path = results / "memory_comparison.png"
output_path = results / "scalability_comparison.png"

# Read your actual experiment results
df = pd.read_csv(csv_path)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

# Left: Existing memory comparison
axes[0].imshow(mpimg.imread(memory_path))
axes[0].axis("off")
axes[0].set_title("A. Full-batch vs. Mini-batch Memory",
                  fontsize=11, fontweight="bold")

# Right: New fanout training time comparison
axes[1].bar(
    df["fanout"].astype(str),
    df["mean_training_s_per_epoch"],
    color="#4472C4"
)

axes[1].set_title("B. GraphSAGE Fanout Scaling",
                  fontsize=11, fontweight="bold")
axes[1].set_xlabel("Neighbor fanout")
axes[1].set_ylabel("Training time per epoch (s)")
axes[1].set_ylim(0, df["mean_training_s_per_epoch"].max() * 1.2)

for i, value in enumerate(df["mean_training_s_per_epoch"]):
    axes[1].text(i, value + 8, f"{value:.1f}s",
                 ha="center", fontsize=9)

axes[1].grid(axis="y", alpha=0.2)
axes[1].set_axisbelow(True)

fig.tight_layout()
fig.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output_path}")
