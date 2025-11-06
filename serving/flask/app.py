import os
from typing import List

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from joblib import load

# =========================
# CONFIGURACIÓN (constantes)
# =========================
DATA_PATH = "data/processed/ar_properties_2022.csv"
MODEL_PATH = "models/price_mpl/models/price_mlp_v1.joblib"

NUM_COL = "surface_covered"
CAT_COLS = ["l3", "property_type"]  # orden esperado por el pipeline entrenado
EXPECTED_FEATURES = [NUM_COL] + CAT_COLS

# =========================
# APP
# =========================
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-key")  # para flash messages

# Carga única al iniciar el proceso
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"No se encontró el modelo en {MODEL_PATH}")
MODEL = load(MODEL_PATH)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"No se encontró el dataset en {DATA_PATH}")
DF = pd.read_csv(DATA_PATH)


# Valores únicos para los selects (ordenados, sin nulos)
def unique_sorted(series: pd.Series) -> List[str]:
    return sorted([str(x) for x in series.dropna().unique()])


PROPERTY_TYPES = unique_sorted(DF["property_type"])
NEIGHBORHOODS = unique_sorted(DF["l3"])


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_path": MODEL_PATH})


@app.get("/")
def index():
    return render_template(
        "index.html",
        property_types=PROPERTY_TYPES,
        neighborhoods=NEIGHBORHOODS,
        default_surface=(
            round(float(DF[NUM_COL].dropna().median()), 1) if NUM_COL in DF else 50.0
        ),
    )


@app.post("/predict")
def predict():
    try:
        property_type = request.form.get("property_type", "").strip()
        neighborhood = request.form.get("neighborhood", "").strip()
        surface_str = request.form.get("surface_covered", "").strip()

        errors = []
        if property_type == "" or property_type not in PROPERTY_TYPES:
            errors.append("Seleccioná un tipo de propiedad válido.")
        if neighborhood == "" or neighborhood not in NEIGHBORHOODS:
            errors.append("Seleccioná un barrio (l3) válido.")
        try:
            surface = float(surface_str.replace(",", "."))
            if surface <= 0:
                errors.append("La superficie debe ser un número positivo.")
        except Exception:
            errors.append("Ingresá una superficie válida (número).")

        if errors:
            for e in errors:
                flash(e, "error")
            return redirect(url_for("index"))

        # Construir DataFrame con el esquema esperado por el pipeline
        X = pd.DataFrame(
            [{NUM_COL: surface, "l3": neighborhood, "property_type": property_type}],
            columns=EXPECTED_FEATURES,
        )

        y_hat = float(MODEL.predict(X)[0])

        # Formato simple para ARS; podés ajustar a tu preferencia
        def fmt_currency(x: float) -> str:
            return f"$ {x:,.0f}".replace(",", ".")  # miles con punto

        return render_template(
            "result.html",
            pred=y_hat,
            pred_fmt=fmt_currency(y_hat),
            surface=surface,
            property_type=property_type,
            neighborhood=neighborhood,
        )

    except Exception as ex:
        flash(f"Error al predecir: {ex}", "error")
        return redirect(url_for("index"))


if __name__ == "__main__":
    # Permite: python serving/flask_app.py
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=True)
