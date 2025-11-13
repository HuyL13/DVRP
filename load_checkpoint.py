import os
from typing import Tuple

import torch

from agent.IPPOAgent import IPPOAgent
from agent.PNet import PolicyNet
from agent.VNet import ValueNet

#Load checkpoint to continue training
def load_agent(agent: IPPOAgent,
               load_dir: str,
               device: torch.device) -> Tuple[int, int]:
    """
    Load policy, value and optimizer state from <load_dir>/agent_<id>.
    Returns (iteration, timesteps) that were saved with the model.
    """
    policy_path = os.path.join(load_dir, f"agent_{agent.agent_id}", "policy.pth")
    value_path  = os.path.join(load_dir, f"agent_{agent.agent_id}", "value.pth")
    opt_path    = os.path.join(load_dir, f"agent_{agent.agent_id}", "optimizer.pth")
    meta_path   = os.path.join(load_dir, f"agent_{agent.agent_id}", "meta.pth")

    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    agent.policy.load_state_dict(torch.load(policy_path, map_location=device))
    agent.value.load_state_dict(torch.load(value_path, map_location=device))

    if os.path.exists(opt_path):
        agent.opt.load_state_dict(torch.load(opt_path, map_location=device))

    # meta contains iteration / timesteps that the logger should start from
    iteration = 0
    timesteps = 0
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location=device)
        iteration = meta.get("iteration", 0)
        timesteps = meta.get("timesteps", 0)

    return iteration, timesteps

#load policy for visualizing
def load_ippo_agent(agent_dir: str, input_dim: int, action_dim: int, device: torch.device):
    policy_path = os.path.join(agent_dir, "policy.pth")
    value_path  = os.path.join(agent_dir, "value.pth")

    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy not found: {policy_path}")

    policy = PolicyNet(input_dim, action_dim).to(device)
    value  = ValueNet(input_dim).to(device)

    policy.load_state_dict(torch.load(policy_path, map_location=device))
    value.load_state_dict(torch.load(value_path, map_location=device))

    policy.eval()
    value.eval()

    return policy, value