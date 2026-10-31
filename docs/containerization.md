# BankGuard — Containerization & Deployment Guide (Task 7.3)

This guide documents the containerization architecture, build commands, and execution modes for the BankGuard Fraud Intelligence Platform.

---

## 1. Container Architecture

The BankGuard container image is built from `python:3.10-slim` using multi-purpose entrypoint routing:

- **Base Image:** `python:3.10-slim` (minimal security surface, small footprint).
- **Runtime Dependency:** `libgomp1` installed for LightGBM OpenMP multithreaded tree evaluation.
- **Security:** Operates as a non-root system user (`bankguard`, UID 1000).
- **Layer Caching:** Python dependencies in `requirements.txt` and `pyproject.toml` are cached before application source code.
- **Data Policy:** Raw Kaggle dataset files are strictly excluded via `.dockerignore`; pre-trained model weights (`artifacts/improved_model.joblib`) and preprocessors are packaged self-contained.
- **Healthcheck:** Standard HTTP probe at `/health` using Python's standard library `urllib` (no external `curl` dependency required).

---

## 2. Quickstart with Docker Compose (Recommended)

To launch both the **FastAPI Inference Service** (port 8000) and the **Berry FinTech Dashboard** (port 8501) with automated health dependency orchestration:

```bash
# Build and start services in detached mode
docker compose up -d --build

# View real-time logs
docker compose logs -f

# Stop and remove containers
docker compose down
```

Once running:
- **FastAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **FastAPI Health Status:** [http://localhost:8000/health](http://localhost:8000/health)
- **Monitoring Telemetry:** [http://localhost:8000/monitoring/summary](http://localhost:8000/monitoring/summary)
- **Streamlit Dashboard:** [http://localhost:8501](http://localhost:8501)

---

## 3. Standalone Docker Commands

### Step A: Build the Image

```bash
docker build -t bankguard:latest .
```

### Step B: Run the FastAPI Serving Service (Port 8000)

```bash
docker run -d \
  --name bankguard-api \
  -p 8000:8000 \
  --restart unless-stopped \
  bankguard:latest api
```

Verify service readiness:
```bash
# Check container health status
docker inspect --format='{{json .State.Health.Status}}' bankguard-api

# Probe health endpoint
curl http://localhost:8000/health
```

Score a sample transaction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "TransactionID": 3459510,
    "TransactionAmt": 280.0,
    "ProductCD": "W",
    "card1": 13926,
    "card6": "credit"
  }'
```

### Step C: Run the Streamlit Dashboard (Port 8501)

```bash
docker run -d \
  --name bankguard-dashboard \
  -p 8501:8501 \
  --restart unless-stopped \
  bankguard:latest dashboard
```

Open your browser to `http://localhost:8501`.

### Step D: Run Both Services in a Single Container

```bash
docker run -d \
  --name bankguard-all \
  -p 8000:8000 \
  -p 8501:8501 \
  bankguard:latest both
```

---

## 4. Mounting Custom Model Artifacts & Data

To run inference against external or newly trained model weights without rebuilding the Docker image, mount a host directory into `/app/artifacts`:

```bash
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/my_custom_artifacts:/app/artifacts \
  bankguard:latest api
```

---

## 5. Verification Checklist

| Verification Check | Target Standard | Verification Method |
| :--- | :--- | :--- |
| **Base Image** | `python:3.10-slim` | Checked in `Dockerfile` line 7 |
| **Security User** | Non-root `bankguard` (UID 1000) | `USER bankguard` verified |
| **Data Separation** | Raw Kaggle files excluded | `.dockerignore` excludes `data/raw/*` |
| **Inference Engine** | OpenMP `libgomp1` present | Tested with LightGBM runtime |
| **Health Probe** | Automated `/health` check | `HEALTHCHECK` passes within 15s |
| **Port Exposure** | 8000 (API) & 8501 (Dashboard) | `EXPOSE 8000 8501` verified |

