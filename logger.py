# -------------------------
# Logger Class
# -------------------------

import os
import random
import math
import time
import csv
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, NamedTuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from dvrp_env import MultiDVRPEnv
# Try to import tensorboard, make it optional for Kaggle
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Logging will use CSV only.")

class IPPOLogger:
    def __init__(self, agent_id: int, log_dir: str = "logs", use_tensorboard: bool = True):
        self.agent_id = agent_id
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE

        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        agent_log_dir = os.path.join(log_dir, f"agent_{agent_id}")
        os.makedirs(agent_log_dir, exist_ok=True)

        # Initialize CSV logger
        self.csv_file = os.path.join(agent_log_dir, "training_log.csv")
        self.csv_handle = open(self.csv_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_handle)
        self.csv_writer.writerow([
            'iteration', 'timesteps', 'avg_reward', 'avg_acp_component',
            'avg_deliver_component', 'avg_penalty', 'avg_cost', 'avg_dist_travel',
            'success_rate', 'avg_total_reward', 'policy_loss', 'value_loss',
            'entropy', 'duration_secs'
        ])

        # Initialize TensorBoard writer
        if self.use_tensorboard:
            tensorboard_dir = os.path.join("runs", "ippo", f"agent_{agent_id}")
            self.writer = SummaryWriter(tensorboard_dir)
        else:
            self.writer = None

        # Logger state
        self.i_so_far = 0
        self.t_so_far = 0
        self.rewards = []
        self.losses = []
        self.delta_t = time.time()

    def log_iteration(self, env: MultiDVRPEnv, policy_loss: float, value_loss: float, entropy: float,
                      learning_rate: float):
        """Log metrics for current iteration"""
        self.i_so_far += 1
        self.t_so_far += env.t

        # Calculate metrics
        avg_reward = env.reward[self.agent_id] / max(1, env.t)
        avg_acp = env.acp_component[self.agent_id] / max(1, env.t)
        avg_deliver = env.deliver_component[self.agent_id] / max(1, env.t)
        avg_penalty = env.penalty[self.agent_id] / max(1, env.t)
        avg_cost = env.avg_cost[self.agent_id] / max(1, env.t)
        avg_dist = env.dist_travel[self.agent_id] / max(1, env.t)
        success_rate = env.successful_deliveries / max(1, env.total_order)
        avg_total_reward = env.total_rewards_per_agent[self.agent_id] / max(1, env.t)

        # Calculate duration
        current_time = time.time()
        duration = current_time - self.delta_t
        self.delta_t = current_time

        # Write to CSV
        self.csv_writer.writerow([
            self.i_so_far, self.t_so_far, round(avg_reward, 5),
            round(avg_acp, 5), round(avg_deliver, 5), round(avg_penalty, 5),
            round(avg_cost, 5), round(avg_dist, 5), round(success_rate, 5),
            round(avg_total_reward, 5), round(policy_loss, 5),
            round(value_loss, 5), round(entropy, 5), round(duration, 2)
        ])
        self.csv_handle.flush()

        # Write to TensorBoard
        if self.writer:
            self.writer.add_scalar("charts/learning_rate", learning_rate, self.i_so_far)
            self.writer.add_scalar("rewards/avg_reward", avg_reward, self.i_so_far)
            self.writer.add_scalar("rewards/avg_total_reward", avg_total_reward, self.i_so_far)
            self.writer.add_scalar("rewards/avg_acp_component", avg_acp, self.i_so_far)
            self.writer.add_scalar("rewards/avg_deliver_component", avg_deliver, self.i_so_far)
            self.writer.add_scalar("penalties/avg_penalty", avg_penalty, self.i_so_far)
            self.writer.add_scalar("costs/avg_cost", avg_cost, self.i_so_far)
            self.writer.add_scalar("metrics/avg_dist_travel", avg_dist, self.i_so_far)
            self.writer.add_scalar("metrics/success_rate", success_rate, self.i_so_far)
            self.writer.add_scalar("losses/policy_loss", policy_loss, self.i_so_far)
            self.writer.add_scalar("losses/value_loss", value_loss, self.i_so_far)
            self.writer.add_scalar("losses/entropy", entropy, self.i_so_far)
            self.writer.add_scalar("charts/timesteps", self.t_so_far, self.i_so_far)

        return {
            'iteration': self.i_so_far,
            'timesteps': self.t_so_far,
            'avg_reward': avg_reward,
            'avg_acp': avg_acp,
            'avg_deliver': avg_deliver,
            'avg_penalty': avg_penalty,
            'avg_cost': avg_cost,
            'avg_dist': avg_dist,
            'success_rate': success_rate,
            'avg_total_reward': avg_total_reward,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'duration': duration
        }

    def print_summary(self, metrics: Dict[str, Any]):
        """Print formatted summary to console"""
        print(flush=True)
        print(f"========== Agent {self.agent_id} - Iteration #{metrics['iteration']} ==========", flush=True)
        print(f"Timesteps So Far: {metrics['timesteps']}", flush=True)
        print(f"Average Reward: {metrics['avg_reward']:.5f}", flush=True)
        print(f"  - Accept Component: {metrics['avg_acp']:.5f}", flush=True)
        print(f"  - Deliver Component: {metrics['avg_deliver']:.5f}", flush=True)
        print(f"Average Penalty: {metrics['avg_penalty']:.5f}", flush=True)
        print(f"Average Cost: {metrics['avg_cost']:.5f}", flush=True)
        print(f"Average Distance: {metrics['avg_dist']:.5f}", flush=True)
        print(f"Success Rate: {metrics['success_rate']:.5f}", flush=True)
        print(f"Policy Loss: {metrics['policy_loss']:.5f}", flush=True)
        print(f"Value Loss: {metrics['value_loss']:.5f}", flush=True)
        print(f"Entropy: {metrics['entropy']:.5f}", flush=True)
        print(f"Iteration Duration: {metrics['duration']:.2f} secs", flush=True)
        print(f"=" * 60, flush=True)
        print(flush=True)

    def close(self):
        """Close logger resources"""
        self.csv_handle.close()
        if self.writer:
            self.writer.close()