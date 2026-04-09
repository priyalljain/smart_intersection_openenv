"""
FastAPI + OpenEnv server for HuggingFace Spaces deployment
Absolute imports from root level (not from server package)
"""

import os

from fastapi import FastAPI
from openenv.core.env_server import create_app

# ABSOLUTE IMPORTS FROM ROOT (critical for HF Spaces)
from env import TrafficControlEnv
from models import TrafficAction, TrafficObservation

# Create OpenEnv app with POSITIONAL arguments (not keyword arguments)
app = create_app(TrafficControlEnv, TrafficAction, TrafficObservation)

def main():
    """Entry point for running the server"""
    import uvicorn
    
    # HuggingFace Spaces uses port 7860
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    workers = int(os.getenv("WORKERS", "1"))
    
    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        workers=workers,
        reload=False
    )

if __name__ == "__main__":
    main()