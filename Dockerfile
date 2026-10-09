# Container for running ResearchPilot AI on a Nebius AI Cloud VM (or any Docker host).
# NOTE: this Dockerfile has not been build-tested by the author of this package.
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1 DB_PATH=/data/researchpilot.db
VOLUME /data
EXPOSE 8501
# Pass secrets at runtime (docker run --env-file .env); never bake them into the image.
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
