
# 🚚 IPPO Multi-Agent Dynamic Vehicle Routing Problem (DVRP)

A **multi-agent reinforcement learning** framework for the **Dynamic Capacitated Vehicle Routing Problem with Time Windows (DVRP-TW)** using **Independent Proximal Policy Optimization (IPPO)**.

---

## ✨ Features

* 🧩 **Custom DVRP environment** with dynamic order arrivals, time windows, and shared penalties.
* 🤖 **IPPO agents** with action masking, reward shaping, and per-agent logging.
* 🏋️ **Training script** with checkpointing and resume support.
* 🎬 **Live visualizer** for real-time animated simulations.

---

## ⚙️ Installation

```bash
git clone https://github.com/yourname/ippo-dvrp.git
cd ippo-dvrp
```

---

## 🚀 Training

```bash
python train_ippo.py
```

---

## ▶️ Resume Training

```python
agents = run_ippo(
    env,
    load_path="checkpoint/iter_750",  # Resume from checkpoint
    num_iterations=2000
)
```

---

## 🎥 Visualization (Live Animation)

```bash
python visualizer.py
```

### In Jupyter Notebook

```python
%matplotlib notebook
```

### In Python Script

`plt.ion()` is already enabled.

---

## 🗺️ What You’ll See

| Symbol             | Meaning                               |
| ------------------ | ------------------------------------- |
| 🟥 Red Square      | Depot (restock point)                 |
| 🔵 Colored Circles | Agents (labeled 0, 1, 2)              |
| 🟡 Gold Circles    | Available orders                      |
| ⭐ Gold Stars       | Accepted orders                       |
| 🧾 Info Panel      | Time, success rate, rewards, distance |

Agents move smoothly step-by-step across the grid.

---

## 📊 Evaluation Metrics

At the end of each episode:

```
Success Rate           = delivered / total_orders
Avg Distance per Agent = total_dist / M
Objective (lower better) = 0.5 × (10 × (1 - success) + avg_dist / 480)
```

---

## 🧠 Tips for Better Performance

* Increase `rollout_steps` to **480+** for longer episodes.
* Tune `α` (0.2–0.5) to balance exploration vs. delivery focus.
* Enable normalization (`use_normalization=True`) if observations vary widely.
* Use **GPU** for faster training.

---

## 👥 Contributors

* **Nguyễn Quang Huy** – Core Developer

---

## 📄 License

Feel free to **use, modify, and distribute** this project under an open license.

---
