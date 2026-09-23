"""Generate exploratory figures for the red wine dataset."""
from pathlib import Path
import sys

# Make ``python scripts/eda.py`` work without requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from red_wine_quality.data import load_data
from red_wine_quality.model import train
from red_wine_quality.preprocessing import split_features_target

out = Path("reports/figures")
out.mkdir(parents=True, exist_ok=True)
df = load_data()
X, y = split_features_target(df)
sns.set_theme(style="whitegrid")
for column in df.columns:
    plt.figure(figsize=(7, 4))
    sns.histplot(df[column], kde=True)
    plt.title(f"Distribution: {column}")
    plt.tight_layout()
    plt.savefig(out / f"hist_{column.replace(' ', '_')}.png", dpi=130)
    plt.close()
    plt.figure(figsize=(7, 3))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot: {column}")
    plt.tight_layout()
    plt.savefig(out / f"box_{column.replace(' ', '_')}.png", dpi=130)
    plt.close()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), cmap="vlag", center=0)
plt.title("Feature correlation matrix")
plt.tight_layout()
plt.savefig(out / "correlations.png", dpi=150)
plt.close()
for feature in ["alcohol", "volatile acidity", "sulphates"]:
    plt.figure(figsize=(7, 4))
    sns.regplot(data=df, x=feature, y="quality", scatter_kws={"alpha": 0.25}, line_kws={"color": "red"})
    plt.title(f"{feature.title()} vs quality")
    plt.tight_layout()
    plt.savefig(out / f"{feature.replace(' ', '_')}_vs_quality.png", dpi=150)
    plt.close()

model, _ = train(tune=True)
rf = model.best_estimator_.named_steps["model"] if hasattr(model, "best_estimator_") else model.named_steps["model"]
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
importance.plot.barh(figsize=(8, 5), title="Random Forest feature importance")
plt.tight_layout()
plt.savefig(out / "feature_importance.png", dpi=150)
plt.close()
predicted = model.predict(X)
plt.figure(figsize=(6, 6))
plt.scatter(y, predicted, alpha=0.3)
plt.xlabel("Observed quality")
plt.ylabel("Predicted quality")
plt.title("Evaluation: observed vs predicted")
plt.tight_layout()
plt.savefig(out / "evaluation_observed_vs_predicted.png", dpi=150)
plt.close()
