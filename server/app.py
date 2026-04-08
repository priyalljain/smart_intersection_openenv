import os
from fastapi import FastAPI
from openenv.core.env_server import create_app

from env import TrafficControlEnv
from models import TrafficAction, TrafficObservation

app = create_app(TrafficControlEnv, TrafficAction, TrafficObservation)

def main():
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 7860))  # HF default
    workers = int(os.getenv("WORKERS", 4))
    uvicorn.run("server.app:app", host=host, port=port, workers=workers, reload=False)

if __name__ == "__main__":
    main()