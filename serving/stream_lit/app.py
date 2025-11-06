from pathlib import Path
import pandas as pd
from joblib import load
import streamlit as st

# Paths (mismo criterio que en FastAPI)
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]  # .../mlops-class/
DATA_PATH = PROJECT_ROOT / "data/processed/ar_properties_2022.csv"
MODEL_PATH = PROJECT_ROOT / "models/price_mpl/models/price_mlp_v1.joblib"


# Carga
@st.cache_data
def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_resource
def load_model(path: Path):
    return load(path)


df = load_df(DATA_PATH)
model = load_model(MODEL_PATH)

NUM_COL = "surface_covered"
CAT_COLS = ["l3", "property_type"]
EXPECTED_FEATURES = [NUM_COL] + CAT_COLS

neighborhoods = sorted(map(str, df["l3"].dropna().unique()))
property_types = sorted(map(str, df["property_type"].dropna().unique()))
default_surface = float(df[NUM_COL].dropna().median()) if NUM_COL in df else 50.0

# UI
st.set_page_config(page_title="Estimador de precios", page_icon="🏠", layout="centered")
st.title("🏠 Estimador de precio de propiedades")

col_a, col_b = st.columns(2)
with col_a:
    selected_type = st.selectbox("Tipo de propiedad", options=property_types)
with col_b:
    selected_nei = st.selectbox("Barrio (l3)", options=neighborhoods)

surface = st.number_input(
    "Superficie cubierta (m²)", min_value=1.0, step=1.0, value=round(default_surface, 1)
)

if st.button("Estimar precio"):
    X = pd.DataFrame(
        [{NUM_COL: surface, "l3": selected_nei, "property_type": selected_type}],
        columns=EXPECTED_FEATURES,
    )

    pred = float(model.predict(X)[0])
    st.success(f"Precio estimado: $ {pred:,.0f}".replace(",", "."))
    with st.expander("Ver entrada utilizada"):
        st.json(
            {
                "surface_covered": surface,
                "l3": selected_nei,
                "property_type": selected_type,
            }
        )

st.caption(f"Modelo: {MODEL_PATH}")
