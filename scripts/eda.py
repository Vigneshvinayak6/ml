"""Generate exploratory figures for the red wine dataset."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from red_wine_quality.data import load_data

out = Path("reports/figures")
out.mkdir(parents=True, exist_ok=True)
df = load_data()
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.histplot(df["quality"], discrete=True)
plt.title("Red wine quality distribution")
plt.tight_layout()
plt.savefig(out / "quality_distribution.png", dpi=150)
plt.close()
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), cmap="vlag", center=0)
plt.title("Feature correlation matrix")
plt.tight_layout()
plt.savefig(out / "correlations.png", dpi=150)
plt.close()
