---
title: Traffic Control Environment Server
sdk: docker
app_port: 7860
base_path: /web
pinned: false
---

# Autonomous Traffic Control Environment for OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-brightgreen)](https://github.com/meta-pytorch/OpenEnv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready reinforcement learning environment that simulates a **4-way intersection** with realistic traffic dynamics, emergency vehicle prioritisation, pedestrians, crashes, flooding, and road closures. Designed for the Meta OpenEnv Hackathon 2026.

The environment is fully compliant with the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) specification – it exposes `reset()`, `step()`, and `state()` methods, uses typed Pydantic models, and can be deployed as a containerised service on Hugging Face Spaces.

---

## Real-World Problem

In many cities, emergency vehicles (ambulances, fire trucks) get stuck in traffic, costing lives. This environment trains AI agents to manage traffic lights adaptively, prioritising emergencies while keeping normal traffic flowing. It also models floods, crashes, pedestrians, and fairness between lanes – all essential for real-world deployment.

---

## Environment Features

- **4-way intersection** – Lanes: North, South, East, West. Left-hand traffic (India).
- **Traffic phases** – `ns` (North-South green) and `ew` (East-West green).
- **Emergency vehicles** – Ambulance, fire truck, police – with preemption over normal phases.
- **Pedestrians** – Normal, elderly, child – with waiting time limits and equity penalties.
- **Crashes** – Random events that block lanes, trigger all-red, and heavily penalise safety.
- **Flooding** – Lanes become impassable (hard task); they automatically clear after 60-120 seconds.
- **Road closures / construction** – Lanes can be blocked; vehicles cannot enter or exit.
- **Fairness enforcement** – Penalty if one direction's queue is more than double the other.
- **Starvation prevention** – Maximum green time (60 seconds) forces a phase switch.
- **Thrashing penalty** – Penalises too-frequent phase changes.
- **Dense reward** – Four sub-scores (safety, emergency, efficiency, equity) weighted and normalised to 0-1.

---

## Tasks (3 difficulty levels)

| Task   | Description | Challenges |
|--------|-------------|------------|
| **easy** | Normal vehicles only | Basic queue management, avoid thrashing, keep all lanes moving. |
| **medium** | Adds emergency vehicles (ambulance, fire) | Must detect and prioritise emergencies while maintaining reasonable queues. |
| **hard** | Adds floods, crashes, and multiple emergencies | Simultaneous events, all-red phases, lane blockages, fairness penalties. |

---

## Action Space

The agent selects one of two phases:
- `ns` – Green for North and South, red for East and West.
- `ew` – Green for East and West, red for North and South.

---

## Observation Space

The observation (Pydantic `TrafficObservation`) includes:
- `time` – current episode time (seconds)
- `phase` – current phase (`ns`, `ew`, or `all_red`)
- `time_in_phase` – seconds since last phase change
- `queues` – dictionary `{lane: count}` for each lane
- `total_queue_length` – sum of all queues
- `vehicles_cleared_this_step` – throughput metric
- `active_emergencies` – number of waiting emergency vehicles
- `waiting_pedestrians` – number of pedestrians waiting to cross
- `flooded_lanes`, `blocked_lanes`, `crashed_this_step` – lane statuses
- `phase_history` – list of recent phase changes (for thrashing detection)
- `safety_score`, `emergency_score`, `efficiency_score`, `equity_score` – sub-scores (0-1000) for debugging

---

## Reward Function

The reward is a weighted combination of four sub-scores, normalised to [0, 1]:

```text
reward = (safety_score*0.30 + emergency_score*0.35 + efficiency_score*0.20 + equity_score*0.15) / 1000
```

| Component | Weight | Penalties (examples) |
|-----------|--------|----------------------|
| **Safety** | 30%    | Crash (-500), min-green violation (-10), thrashing (-20) |
| **Emergency** | 35%    | Emergency wait time: >30s -> -100, >60s -> -300 |
| **Efficiency**| 20%    | Queue length (-15 per vehicle), wasted green (-5 per second) |
| **Equity** | 15%    | Pedestrian wait > max allowed (-50), queue imbalance (up to -200) |

Positive signals are given for clearing vehicles and serving emergencies quickly.

---

## Installation & Local Usage

### Prerequisites
- Python 3.10+
- Docker (optional, for containerised deployment)

### 1. Clone & Setup

```bash
git clone [https://github.com/layirp/traffic-control-env](https://github.com/layirp/traffic-control-env)
cd traffic-control-env
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Run the Baseline Inference Script
This runs the heuristic expert agent (or an LLM if you set `HF_TOKEN` in `.env`). It prints logs in the required `[START]/[STEP]/[END]` format.

```bash
python inference.py
```

### 3. Run the Three Task Graders

```bash
python tests/test_graders.py
```

*Output example:*
```text
Easy:   0.77
Medium: 0.73
Hard:   0.78
```

### 4. Validate OpenEnv Compliance

```bash
openenv validate --verbose
```

### 5. Start the Environment Server Locally (FastAPI)

```bash
uvicorn server.app:app --reload --port 8000
```
Then open `http://localhost:8000/docs` for API documentation.

### 6. Build and Run the Docker Container

```bash
docker build -t traffic-control .
docker run -p 8000:7860 traffic-control
```

*Test the `/reset` endpoint:*
```bash
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d '{}'
```

---

## Deploy to Hugging Face Spaces

### 1. Login to Hugging Face
Use your token with write permissions:

```bash
huggingface-cli login
```
*(If the CLI is not found, use `python -c "from huggingface_hub import login; login()"`)*

### 2. Push the Environment

```bash
openenv push --repo-id your-username/traffic-control-env
```
After deployment, your Space will be live at:
`https://huggingface.co/spaces/your-username/traffic-control-env`

### 3. Test the Live Space

```bash
curl -X POST [https://your-username-traffic-control-env.hf.space/reset](https://your-username-traffic-control-env.hf.space/reset) -H "Content-Type: application/json" -d '{}'
```

### 4. Run the Pre-Submission Validation Script
Provided by the hackathon:

```bash
bash validate-submission.sh [https://your-username-traffic-control-env.hf.space](https://your-username-traffic-control-env.hf.space) .
```

---

## Baseline Scores (Heuristic Expert Agent)

These scores are obtained by running the `HeuristicExpertAgent` (hand-coded domain knowledge) for 100 steps per episode, averaged over 3 episodes.

| Task   | Average Reward (0-1) |
|--------|----------------------|
| **easy** | 0.77                 |
| **medium** | 0.73                 |
| **hard** | 0.78                 |

---

## Project Structure

```text
traffic-control-env/
├── .gitignore
├── README.md
├── LICENSE
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── inference.py
├── client.py
├── env.py
├── models.py
├── simulator.py
├── agents.py
├── server/
│   ├── __init__.py
│   ├── app.py
│   └── requirements.txt
├── tests/
│   ├── __init__.py
│   └── test_graders.py
└── .env (ignored, for local secrets)
```

---

## Acknowledgements

This project would not have been possible without the support and infrastructure provided by:

- **Meta (PyTorch team)** – for creating the OpenEnv framework and sponsoring the hackathon.
- **Hugging Face** – for hosting the Spaces platform, providing free inference APIs, and supporting open-source AI research.
- **Scaler** – for organising the Meta OpenEnv Hackathon and providing mentorship.
- **OpenEnv contributors** – for developing the standardised environment API that makes RL environments interoperable.
- **The open-source community** – for tools like FastAPI, Uvicorn, Pydantic, and Docker that made this environment robust and production-ready.

Special thanks to the hackathon judges and mentors for their valuable feedback and for promoting the development of real-world RL benchmarks.
