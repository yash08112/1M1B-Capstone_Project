# Smart Campus Water — production image (trains model at build time).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py train_and_save_model.py index.html Modified_Campus_Water_Full_Feature_Set.csv ./
COPY water_ml ./water_ml

RUN python train_and_save_model.py

EXPOSE 8080
ENV PORT=8080

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-8080} --workers 2 --timeout 120"]
