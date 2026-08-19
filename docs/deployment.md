# Deployment Guide

## Local Development

```bash
streamlit run app.py
# Dashboard available at http://localhost:8501
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
```

```bash
# Build and run
docker build -t delhivery-ni .
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/outputs:/app/outputs \
  delhivery-ni
```

---

## Streamlit Community Cloud

1. Push repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repo.
4. Set **Main file path** to `app.py`.
5. Add secrets if needed (none required for this project).
6. Click **Deploy**.

> **Note:** Models and outputs (`.pkl`, large `.csv`) are excluded via `.gitignore`.
> For cloud deployment, either commit lightweight versions or load from cloud storage.

---

## Predictor API as a REST Service

The `ETAPredictor` can be wrapped in FastAPI for production serving:

```python
# serve.py
from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.predictor import ETAPredictor

app = FastAPI(title="Delhivery ETA Predictor")
predictor = ETAPredictor.load("models/", "outputs/")

class PredictRequest(BaseModel):
    osrm_time: float
    osrm_distance: float
    actual_distance: float
    route_type: str
    hour: int
    day_of_week: int
    month: int
    source_center: str
    destination_center: str

@app.post("/predict")
def predict(req: PredictRequest):
    result = predictor.predict(**req.dict())
    return result.to_dict()
```

```bash
pip install fastapi uvicorn
uvicorn serve:app --host 0.0.0.0 --port 8000
```

---

## Environment Variables

No environment variables are required for local or dashboard deployment.

For production API deployments, consider:

```bash
export MODEL_DIR=/app/models
export REPORTS_DIR=/app/outputs
export LOG_LEVEL=INFO
```

---

## Model Artefact Management

Trained models are stored as pickle files in `models/`:

| File | Size | Contents |
|------|------|---------|
| `improved_models.pkl` | ~15 MB | HistGBM FTL, Carting, global models + blend weights |
| `trained_models.pkl` | ~8 MB | Baseline RF models |
| `graphs.pkl` | ~2 MB | NetworkX graph objects |
| `feature_store.pkl` | ~170 MB | Full train/test feature arrays (excluded from git) |

For production, store large artefacts in cloud object storage (S3, GCS) and load on startup.

---

## Performance Characteristics

| Operation | Time | Hardware |
|-----------|------|---------|
| Full pipeline (all 7 steps) | ~8–12 min | 4-core CPU, 16 GB RAM |
| HistGBM training (per model) | ~30–60 s | 4-core CPU |
| Batch prediction (40K rows) | ~0.5 s | Single core |
| Single prediction (API call) | ~5 ms | Single core |
| Dashboard startup | ~3 s | Any |
