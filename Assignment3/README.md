# Assignment 3 — Policy Gradient Methods

This folder contains simple implementations of policy-gradient methods applied to the CartPole-v1 environment.

Files
- `A2C.py` — Advantage Actor-Critic training and plotting
- `AC.py` — Basic Actor-Critic (Monte Carlo returns) implementation
- `reinforce.py` — REINFORCE (vanilla policy gradient)
- `BaselineDataCartPole.csv` — baseline data used for comparison/plotting

Requirements
- See `requirements.txt` for required Python packages and pinned versions.

Quick start

1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run an algorithm (example):

```bash
python3 reinforce.py
python3 AC.py
python3 A2C.py
```

Notes
- These scripts are written for Python 3.10+ and assume a local display for plotting.
- If running on a headless server, save plots to files or use a non-interactive matplotlib backend.
