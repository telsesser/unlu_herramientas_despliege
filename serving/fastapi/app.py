import os
from datetime import datetime
from typing import List, Literal, Optional
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, ConfigDict
from joblib import load

# uvicorn serving.fastapi.app:app --host 0.0.0.0 --port 8000 --reload


# CONFIG
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
TEMPLATES_DIR = THIS_FILE.with_name("templates")

DATA_PATH = PROJECT_ROOT / "data/processed/ar_properties_2022.csv"
MODEL_PATH = PROJECT_ROOT / "models/price_mpl/models/price_mlp_v1.joblib"

NUM_COL = "surface_covered"
CAT_L3 = "l3"
CAT_PROP = "property_type"
EXPECTED_FEATURES = [NUM_COL, CAT_L3, CAT_PROP]

MODEL = load(MODEL_PATH)
DF = pd.read_csv(DATA_PATH)


def unique_sorted(series: pd.Series) -> List[str]:
    return sorted([str(x) for x in series.dropna().unique()])


PROPERTY_TYPES: List[str] = unique_sorted(DF[CAT_PROP])
NEIGHBORHOODS: List[str] = unique_sorted(DF[CAT_L3])


# SCHEMAS
class PredictIn(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "surface_covered": 55.0,
                    "l3": "Recoleta",
                    "property_type": "Departamento",
                }
            ]
        }
    )
    surface_covered: float = Field(..., gt=0, description="Superficie cubierta (m²)")
    l3: str = Field(..., description="Barrio (nivel l3)")
    property_type: str = Field(..., description="Tipo de propiedad")


class PredictOut(BaseModel):
    price: float = Field(..., description="Precio estimado")
    currency: str = Field(default="ARS")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )


class OptionsOut(BaseModel):
    property_types: List[str]
    neighborhoods: List[str]


# APP
app = FastAPI(
    title="Estimador de precios de propiedades (FastAPI)",
    version="1.1.0",
    description="""
API para estimar precio de propiedades.
""",
)


@app.get(
    "/options", response_model=OptionsOut, summary="Devuelve los valores permitidos"
)
def options():
    return OptionsOut(property_types=PROPERTY_TYPES, neighborhoods=NEIGHBORHOODS)


@app.post(
    "/predict", response_model=PredictOut, summary="Estima el precio de una propiedad"
)
def predict(inp: PredictIn):
    if inp.l3 not in NEIGHBORHOODS:
        raise HTTPException(
            status_code=400, detail=f"Barrio inválido. Consultar en /options"
        )

    if inp.property_type not in PROPERTY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de propiedad inválida. Consultar en /options",
        )
    # Construir DataFrame con el orden esperado por el pipeline
    X = pd.DataFrame(
        [{NUM_COL: inp.surface_covered, CAT_L3: inp.l3, CAT_PROP: inp.property_type}],
        columns=EXPECTED_FEATURES,
    )

    try:
        y_hat = float(MODEL.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")

    return PredictOut(price=y_hat)


@app.get(
    "/predict",
    response_model=PredictOut,
    summary="Estima el precio de una propiedad (GET demo)",
)
def predict_get(
    surface_covered: float = Query(..., gt=0, description="Superficie cubierta (m²)"),
    l3: str = Query(..., description="Barrio (nivel l3)"),
    property_type: str = Query(..., description="Tipo de propiedad"),
):
    if l3 not in NEIGHBORHOODS:
        raise HTTPException(
            status_code=400, detail=f"Barrio inválido. Consultar en /options"
        )

    if property_type not in PROPERTY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de propiedad inválido. Consultar en /options",
        )

    X = pd.DataFrame(
        [{NUM_COL: surface_covered, CAT_L3: l3, CAT_PROP: property_type}],
        columns=EXPECTED_FEATURES,
    )

    try:
        y_hat = float(MODEL.predict(X)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en predicción: {e}")

    return PredictOut(price=y_hat)


del DF
