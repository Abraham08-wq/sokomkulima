FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
ENV PYTHONUNBUFFERED=1 PORT=8000
COPY requirements.txt requirements-gnn.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir -r requirements-gnn.txt
COPY pipeline ./pipeline
COPY model ./model
COPY api ./api
COPY agro ./agro
COPY insure ./insure
COPY service ./service
COPY disease ./disease
COPY tests ./tests
COPY frontend ./frontend
COPY run_all.sh ./
RUN mkdir -p data/raw data/processed outputs && chmod +x run_all.sh
EXPOSE 8000
CMD ["./run_all.sh"]
