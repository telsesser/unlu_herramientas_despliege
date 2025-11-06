# src/train_regression.py
import os
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    mean_squared_error,
)


# CONFIGURACIÓN (constantes)

RANDOM_STATE = 22

DATA_PATH = "data/processed/ar_properties_2022.csv"  # dataset limpio
TARGET_COL = "price"

NUM_FEATURES = ["surface_covered"]
CAT_FEATURES = ["l3", "property_type"]  # l3 = barrio

TEST_SIZE = 0.25

# Hiperparámetros del MLP
MLP_PARAMS = dict(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    alpha=1e-3,
    learning_rate_init=1e-3,
    max_iter=500,
    random_state=RANDOM_STATE,
    verbose=True,
    early_stopping=True,
)

THIS_FILE = Path(__file__).resolve()

PROJECT_ROOT = THIS_FILE.parent

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "price_mlp_v1.joblib"

REPORT_DIR = PROJECT_ROOT / "reports" / "metrics"
REPORT_PATH = REPORT_DIR / "price_mlp_v1.json"


def ensure_dirs():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el dataset en {path}")
    df = pd.read_csv(path)
    return df


def build_pipeline():
    """
    Pipeline = Preprocesamiento (escala numéricas + OHE categóricas) + MLPRegressor.
    Guardar el pipeline completo permite reusar el preprocesamiento en despliegue.
    """
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore", sparse_output=False
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_FEATURES),
            ("cat", categorical_transformer, CAT_FEATURES),
        ]
    )

    mlp = MLPRegressor(**MLP_PARAMS)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("mlp", mlp)])

    return pipeline


def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"MSE": mse, "MAE": mae, "RMSE": rmse, "R2": r2}


def main():
    ensure_dirs()

    # Cargar datos
    print("Cargando dataset")
    df = load_dataset(DATA_PATH)

    # Seleccionar X, y
    expected_cols = NUM_FEATURES + CAT_FEATURES + [TARGET_COL]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")

    X = df[NUM_FEATURES + CAT_FEATURES]
    y = df[TARGET_COL]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Pipeline
    pipe = build_pipeline()

    print("Entrenando modelo")
    # Entrenar
    pipe.fit(X_train, y_train)

    # Evaluar
    y_pred = pipe.predict(X_test)
    metrics = evaluate(y_test, y_pred)

    # Exportar modelo
    dump(pipe, MODEL_PATH)

    # Guardar métricas + metadatos
    report = {
        "model_path": str(MODEL_PATH),
        "dataset_path": DATA_PATH,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "random_state": RANDOM_STATE,
        "features": {"numerical": NUM_FEATURES, "categorical": CAT_FEATURES},
        "mlp_params": MLP_PARAMS,
        "split": {"test_size": TEST_SIZE},
        "metrics": metrics,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Entrenamiento finalizado")
    print(f"Modelo guardado en: {MODEL_PATH}")
    print(
        f"Métricas (test): MSE={metrics['MSE']:.2f} | MAE={metrics['MAE']:.2f} | RMSE={metrics['RMSE']:.2f} | R2={metrics['R2']:.4f}"
    )
    print(f"Reporte métricas: {REPORT_PATH}")


if __name__ == "__main__":
    main()
