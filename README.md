# Q-Learning: Tabular and Deep

## Requirements
Install dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure
```
Assignment2/
├── dqn_naive/
│   ├── averaged_runs.py
│   ├── average_dqn_naive_results.csv
│   └── dqn_naive_results_0-4.csv
├── dqn_tn/
│   ├── averaged_runs.py
│   ├── average_dqn_target_network_results.csv
│   └── dqn_target_network_results_0-4.csv
├── dqn_er/
│   ├── averaged_runs.py
│   ├── average_dqn_er_results.csv
│   └── dqn_er_results_0-4.csv
├── dqn_er_tn/
│   ├── averaged_runs.py
│   ├── average_dqn_er_tn_results.csv
│   └── dqn_er_tn_results_0-4.csv
├── experiments-ablation/
│   └── summary.csv
├── DQN_naive.py
├── DQN_target_network.py
├── DQN_experience_replay.py
├── DQN_ER_TN.py
├── main_Naive.py
├── main_TN.py
├── main_ER.py
├── main_ER_TN.py
├── experiments_ablation_study.py
├── plot_ablation_study.py
├── compare_dqn.py
└── BaselineDataCartPole.csv
requirements.txt
README.md
```

## Running Experiments

### 1. Hyperparameter Ablation Study
```bash
python Assignment2/experiments_ablation_study.py
python Assignment2/plot_ablation_study.py
```

### 2. Naive DQN (5 seeds)
```bash
python Assignment2/main_Naive.py
python Assignment2/dqn_naive/averaged_runs.py
```

### 3. Target Network DQN (5 seeds)
```bash
python Assignment2/main_TN.py
python Assignment2/dqn_tn/averaged_runs.py
```

### 4. Experience Replay DQN (5 seeds)
```bash
python Assignment2/main_ER.py
python Assignment2/dqn_er/averaged_runs.py
```

### 5. TN + ER DQN (5 seeds)
```bash
python Assignment2/main_ER_TN.py
python Assignment2/dqn_er_tn/averaged_runs.py
```

### 6. Four-Way Comparison Plot and Metrics
```bash
python Assignment2/compare_dqn.py
```

## Notes
- Each main script automatically runs 5 seeds and saves individual results to the corresponding subfolder
- Run the averaged_runs.py script in each subfolder after the main script to combine seeds
