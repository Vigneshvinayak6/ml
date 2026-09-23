"""Streamlit interface for interactive wine-quality prediction."""
import pandas as pd
import streamlit as st
from red_wine_quality.model import load_model, predict

st.set_page_config(page_title="Red Wine Quality", page_icon="🍷")
st.title("🍷 Red Wine Quality Predictor")
st.caption("Predictions use a persisted scikit-learn regression pipeline.")

@st.cache_resource
def get_model():
    return load_model("models/wine_quality_regressor.joblib")

try:
    model = get_model()
except FileNotFoundError:
    st.error("Model not found. Run `python train.py` first.")
    st.stop()

defaults = {
    "fixed acidity": 7.4, "volatile acidity": 0.70, "citric acid": 0.00,
    "residual sugar": 1.9, "chlorides": 0.076, "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0, "density": 0.9978, "pH": 3.51,
    "sulphates": 0.56, "alcohol": 9.4,
}
values = {}
cols = st.columns(2)
for index, (name, default) in enumerate(defaults.items()):
    values[name] = cols[index % 2].number_input(name, value=float(default), format="%.4f")
if st.button("Predict quality", type="primary"):
    result = predict(model, pd.DataFrame([values]))[0]
    st.metric("Predicted quality score", f"{result:.2f} / 10")
