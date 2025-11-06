# MLOps – Demo de Despliegue (Flask • FastAPI • Streamlit)

Repositorio docente para una práctica de **Ciencia de Datos / MLOps**. Objetivo: llevar un modelo ya entrenado a **inferencia y uso** mediante API y UI simples, con una estructura ordenada y reproducible.

## Propósito

* Mostrar el flujo mínimo: **entrenamiento → serialización → inferencia → despliegue local**.
* Comparar tres vías de consumo: **Flask** (form), **FastAPI** (API con OpenAPI) y **Streamlit** (UI interactiva).
* Mantener una estructura inspirada en **CRISP-DM** sin entrar en limpieza/EDA.

## Estructura simplificada

```
data/processed/         # CSV limpio
models/                 # modelos
serving/                # Flask, FastAPI, Streamlit
```

## Dataset y modelo

* **Problema:** regresión de precio de propiedades (AR).
* **Features:** `surface_covered`, `l3` (barrio), `property_type`.
* **Modelo:** pipeline con `OneHotEncoder + StandardScaler + MLPRegressor`, serializado con **joblib**.

## Requisitos

```
python >= 3.11
pip install -r requirements.txt
```

### 1) Entrenar y exportar

Genera el pipeline y métricas.

```bash
python models/price_mpl/train_price_mlp.py
# crea ./models/price_mlp_v1.joblib y ./reports/metrics/price_mlp_v1.json
```

### 2) Desplegar para inferencia

**Flask (formulario web)**

```bash
python serving/flask/app.py 
```

**FastAPI (API y /docs)**

```bash
uvicorn serving.fastapi.app:app --host 0.0.0.0 --port 8000 --reload
```

**Streamlit (UI)**

```bash
streamlit run serving/stream_lit/app.py 

```


