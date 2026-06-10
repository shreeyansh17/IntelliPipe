from fastapi import FastAPI
from datetime import datetime

app = FastAPI(
    title="IntelliPipe API",
    description="Autonomous Data Quality & Anomaly Intelligence Platform",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "project": "IntelliPipe",
        "description": "Autonomous Data Quality & Anomaly Intelligence Platform",
        "version": "1.0.0",
        "github": "https://github.com/shreeyansh17/IntelliPipe",
        "status": "live"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/features")
def features():
    return {
        "features": [
            "Real-time Kafka event streaming",
            "PySpark anomaly detection",
            "Great Expectations DQ validation",
            "Claude AI root-cause analysis",
            "Auto GitHub PR generation",
            "Jira + Slack incident reporting",
            "RAG over dbt documentation",
            "MLflow model tracking"
        ]
    }
