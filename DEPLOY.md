# Deploying Smart Campus Water Demand

Your app is a **Flask** service. Production uses **Gunicorn** (`Procfile`). The **model is trained during the build** from `Modified_Campus_Water_Full_Feature_Set.csv` (same as locally), so you do not have to commit `model_artifact.joblib`.

## Option A — Render (free tier)

1. Push this folder to **GitHub** (include `Modified_Campus_Water_Full_Feature_Set.csv`, `water_ml/`, `app.py`, `index.html`, `requirements.txt`, `Procfile`).
2. In [Render](https://render.com): **New → Web Service** → connect the repo.
3. Settings:
   - **Runtime:** Python 3.12
   - **Build command:** `pip install -r requirements.txt && python train_and_save_model.py`
   - **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120`
4. Deploy. Open the service URL (HTTPS). The dashboard calls the API on the **same origin**.

*Or* use **Blueprint**: connect repo and select `render.yaml` if you use Render’s YAML deploy.

## Option B — Railway

1. Push repo to GitHub.
2. [Railway](https://railway.app) → **New Project** → **Deploy from GitHub**.
3. Railway detects `Procfile`; set **Root Directory** to this project if the repo is monorepo.
4. If needed, set **build** to: `pip install -r requirements.txt && python train_and_save_model.py` (Render-style). Some setups run build from Nixpacks and need a `railway.json` or custom build — if the model is missing at runtime, add that train step to the build.
5. Deploy; use the generated public URL.

## Option C — Docker (Fly.io, Azure, AWS ECS, any host)

From the project root:

```bash
docker build -t campus-water .
docker run -p 8080:8080 campus-water
```

Open `http://localhost:8080`. For cloud, map the platform’s HTTP port to container **8080**.

## Local production smoke test

```bash
pip install -r requirements.txt
python train_and_save_model.py
set PORT=5000
gunicorn app:app --bind 0.0.0.0:5000 --workers 2
```

(On PowerShell: `$env:PORT=5000` then the same `gunicorn` line.)

## Notes

- **CORS** is open (`*`) for API experiments; tighten for a real company deployment.
- **Free tiers** may sleep or limit CPU; first request after sleep can be slow.
- Commit **no secrets**; this app needs none for basic deploy.
