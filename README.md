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
python -m src.train              # tuned random forest + holdout and 5-fold CV
python -m src.evaluate           # evaluate models/red_wine_quality_model.joblib
python -m src.predict            # prediction smoke test; --input features.csv is supported
python scripts/eda.py           # reports/figures/*.png
streamlit run app.py
pytest -q
```

Use `python -m src.train --no-tune` for a fast Ridge baseline. The trained
pipeline is persisted to `models/red_wine_quality_model.joblib` (ignored).
The primary task is regression; an optional Random Forest classifier is
available via `red_wine_quality.model.train_classifier`. Training includes
holdout evaluation, five-fold CV, and modest Random Forest hyperparameter
tuning.

## Design

The required `src/` modules provide stable CLI/API entry points:
`data_loader.py`, `preprocessing.py`, `train.py`, `evaluate.py`, and
`predict.py`. Reusable implementation lives in `red_wine_quality/`.
`notebooks/red_wine_quality_analysis.ipynb` documents interactive analysis;
`scripts/eda.py` generates distributions, boxplots, requested feature
relationships, a correlation heatmap, feature importance, and evaluation
plots. The
Streamlit app exposes all 11 physicochemical inputs.

## Results

The final executed run (`python -m src.train`, 2026-09-23) produced:

| Model / split | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Tuned RF holdout | 0.4195 | 0.3216 | 0.5671 | 0.5016 |
| 5-fold Ridge CV | 0.5070 | 0.4287 | 0.6536 | 0.3425 |
| LinearRegression CV | 0.5070 | 0.4288 | 0.6536 | 0.3424 |
| RandomForest CV | 0.4144 | 0.3296 | 0.5737 | 0.4938 |
| GradientBoosting CV | 0.4755 | 0.3819 | 0.6175 | 0.4136 |

The selected tuned RF parameters were `n_estimators=250`, `max_depth=None`,
and `min_samples_leaf=1`. Optional classifier holdout metrics were accuracy
0.6813, macro precision 0.4392, macro recall 0.3933, and macro F1 0.4095;
the confusion matrix is returned by `train_classifier`.

Limitations: this is a small, imbalanced sensory-label dataset; scores are
not laboratory measurements, causal claims, or purchasing advice. The random
split does not represent temporal or producer-level generalization, and rare
quality classes make classification difficult. Results vary with dependency
versions and source availability.

## Citation and license

Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009).
Modeling wine preferences by data mining from physicochemical properties.
*Decision Support Systems*, 47(4), 547–553. Dataset:
https://archive.ics.uci.edu/dataset/186/wine+quality

Code is released under the MIT License.
