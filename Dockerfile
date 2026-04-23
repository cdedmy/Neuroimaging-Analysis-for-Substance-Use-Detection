FROM python:3.11-slim

# System deps for numerical libraries + unzip
RUN apt-get update && apt-get install -y --no-install-recommends \
    unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy ML pipeline
COPY ml_pipeline/requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_pipeline/ /app/ml_pipeline/
COPY classifier_inputs.zip /app/

# Extract data
RUN unzip classifier_inputs.zip -d classifier_data/

WORKDIR /app/ml_pipeline

CMD ["python", "main.py", \
     "--fc-dir", "/app/classifier_data/fc_matrices", \
     "--participants", "/app/classifier_data/participants.tsv", \
     "--tags", "cocaine", \
     "--output-dir", "/app/results", \
     "--cv-splits", "3"]
