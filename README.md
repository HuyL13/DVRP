IPPO Multi-Agent Dynamic Vehicle Routing Problem (DVRP)
A multi-agent reinforcement learning framework for Dynamic Vehicle Routing with Time Windows (DVRP-TW) using Independent Proximal Policy Optimization (IPPO).
This repository contains:

A custom DVRP environment with dynamic order arrivals, time windows, and shared penalties.
IPPO agents with action masking, reward shaping, and per-agent logging.
Training script with checkpointing and resume support.
Live visualizer with real-time animated simulation.


Directory Structure
text.
├── dvrp_env.py                 # MultiDVRPEnv implementation
├── agent/
│   ├── IPPOAgent.py            # IPPO agent (policy + value nets)
│   └── Transition.py           # Named tuple for rollout storage
├── logger.py                   # IPPOLogger (CSV + TensorBoard)
├── load_checkpoint.py          # Helper to load policy/value/optimizer
├── train_ippo.py               # Main training loop
├── visualize_ippo.py           # Live animated visualizer
├── checkpoint/                 # Saved models (final, iter_X)
│   └── final/
│       └── agent_0/, agent_1/, ...
├── logs/                       # TensorBoard + CSV logs
└── requirements.txt

Installation
bash: # Clone the repo
git clone https://github.com/yourname/ippo-dvrp.git
cd ippo-dvrp

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows


Training
bash: python train_ippo.py
Resume Training
pythonagents = run_ippo(
    env,
    load_path="checkpoint/iter_750",  # Resume from here
    num_iterations=2000
)

Visualization (Live Animation)
bash: python visualizer.py
In Jupyter Notebook:
python%matplotlib notebook
In .py script: plt.ion() is already enabled.
What You’ll See:

Red square: Depot (restock point)
Colored circles: Agents (labeled 0, 1, 2)
Gold circles: Available orders
Gold stars: Accepted orders
Info panel: Time, success rate, rewards, distance
Smooth movement as agents move step-by-step


Environment Details























































ParameterValueMeaninggrid_size1010×10 gridn_agents33 vehiclesN_max10Max concurrent ordersC10Vehicle capacityT480Episode horizonp_arrival0.25Prob. of new order per steptw60Base time windowα1/3Accept reward fractionpenalty_f50Penalty for missed delivery/expiration
Zones (Order Generation)



































ZoneX-RangeReward RangeProbability0[0–2](6–10)0.11[3–5](2–4)0.42[6–7](2–4)0.43[8–9](6–10)0.1

Evaluation Metrics
At end of episode:
textSuccess Rate           = delivered / total_orders
Avg Distance per Agent = total_dist / M
Objective (lower better) = 0.5 × (10 × (1 - success) + avg_dist / 480)

Example Output (Final Stats)
text======================================================================
                    FINAL EPISODE STATISTICS
======================================================================
Total Orders Generated      : 118
Successful Deliveries       : 92
Success Rate                : 0.780
Total Distance Traveled     : 1842.0
Avg Distance per Agent      : 614.0
Objective (lower better)    : 1.6493
----------------------------------------------------------------------
Per-Agent Breakdown:
  Agent 0: Reward= 245.3 | Dist=  620 | Accept=  82.1 | Deliver= 163.2
  Agent 1: Reward= 238.7 | Dist=  598 | Accept=  79.6 | Deliver= 159.1
  Agent 2: Reward= 241.0 | Dist=  624 | Accept=  80.3 | Deliver= 160.7
======================================================================

Tips for Better Performance

Increase rollout_steps to 480+ for longer episodes.
Tune α (0.2–0.5) to balance exploration vs. delivery focus.
Enable normalization in training (use_normalization=True) if observations vary widely.
Use GPU for faster training.


Contributors

Nguyễn Quang Huy – Core developer


License
MIT License – feel free to use, modify, and distribute.
