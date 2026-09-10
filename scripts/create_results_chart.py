from pathlib import Path

import matplotlib.pyplot as plt


# Frozen test-set WMAPE values from the final model comparison.
models = [
    "Seasonal Naive",
    "LightGBM Poisson",
    "Two-Stage LightGBM",
]

wmape = [
    89.82,
    73.92,
    70.22,
]

# Ensure the output directory exists before saving the chart.
output_dir = Path("assets")
output_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(models, wmape)

ax.set_title("Final Test WMAPE Comparison")
ax.set_ylabel("WMAPE (%)")
ax.set_ylim(0, 100)

# Label each bar so the values are visible without reading the axis precisely.
for bar, value in zip(bars, wmape):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{value:.2f}%",
        ha="center",
        va="bottom",
    )

plt.xticks(rotation=15)
plt.tight_layout()

fig.savefig(
    output_dir / "model_wmape_comparison.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)