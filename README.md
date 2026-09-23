# Red Wine Quality ML

An end-to-end, reproducible machine-learning project for predicting the sensory
quality score (0–10) of Portuguese red wine.

## Data

The loader uses the **UCI Machine Learning Repository Red Wine Quality
dataset** (Cortez et al., 2009), downloaded from
`https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/`.
If that endpoint is unavailable it tries the equivalent public CSV mirror at
`selva86/datasets` on GitHub. The data is semicolon-delimited and is downloaded
to `data/winequality-red.csv` (ignored by git).

## Quick start

```bash
python -m pip install -r requirements.txt
python train.py                 # tuned random forest + holdout and 5-fold CV
python scripts/eda.py           # reports/figures/*.png
streamlit run app.py
pytest -q
```

Use `python train.py --no-tune` for a fast Ridge baseline. The trained
pipeline is persisted to `models/wine_quality_regressor.joblib` (also ignored).
The primary task is regression; the numeric score can be rounded/clipped by a
consumer when a discrete classification interpretation is required.

## Design

`red_wine_quality/data.py` handles download and schema validation;
`preprocessing.py` builds a leakage-safe imputation/scaling pipeline; and
`model.py` provides tuning, evaluation, cross-validation, persistence, and
prediction APIs. `scripts/eda.py` generates a target histogram and correlation
heatmap. The Streamlit app exposes all 11 physicochemical inputs.

## Results

Run `python train.py` locally to generate metrics for your environment. Metrics
are intentionally not hard-coded here: they depend on the downloaded dataset,
scikit-learn version, and split. The command prints holdout MAE/RMSE/R² and
cross-validation scores.

## Citation and license

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).
Modeling wine preferences by data mining from physicochemical properties.
*Decision Support Systems*, 47(4), 547–553. Dataset:
https://archive.ics.uci.edu/dataset/186/wine+quality

Code is released under the MIT License.
