FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
RUN pip install --no-cache-dir ultralytics==8.4.180 opencv-python-headless lap==0.5.13 numpy==2.2.6 pandas==3.0.6 \
    openpyxl==3.1.5 fastapi==0.136.1 uvicorn==0.46.0 python-multipart==0.0.27

WORKDIR /app
COPY . /app
ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1 PORT=8108
WORKDIR /app/service
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port $PORT --workers 1"]
