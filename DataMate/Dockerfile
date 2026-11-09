# DataMate - deterministic agent, no model weights, CPU only, fully offline.
# NOTE: check the organisers' Docker guideline for the required base image before the final submission.
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY datamate /app/datamate
COPY predict.sh /app/predict.sh
RUN chmod +x /app/predict.sh && mkdir -p /data /result
ENTRYPOINT ["/app/predict.sh"]
