# ReinforcementLearning

**Quick Start**

1. Ensure you have Python 3.8+ installed.
2. (Optional) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # if you add one
```

3. Run an experiment script (examples below).

**Running experiments**

- To run experiments and compare algorithms, open `Experiment.py` and adjust parameters or run it directly:

```bash
python Experiment.py
```

- Use `Agent.py` and `Environment.py` to create custom agents or environments for further experiments.

**Project structure**

- `Agent.py`: base agent utilities.
- `Environment.py`: environment definition and simulation loops used by experiments.
- `DynamicProgramming.py`: dynamic programming algorithms (policy evaluation/improvement).
- `MonteCarlo_solution.py`: Monte Carlo policy evaluation / control examples.
- `Nstep_solution.py`: N-step learning approaches.
- `Q_learning_solution.py`: Q-learning implementation and experiments.
- `SARSA_solution.py`: SARSA implementation and experiments.
- `Experiment.py`: script to run experiments and produce basic outputs/plots.
- `Helper.py`: helper functions used across modules (plotting, metrics, seeds).
