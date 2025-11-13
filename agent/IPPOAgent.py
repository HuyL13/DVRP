import math
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agent.PNet import PolicyNet
from agent.VNet import ValueNet
from agent.Transition import Transition

def flatten_observation(obs: Dict[str, Any]) -> np.ndarray:
    vec = []
    vec.append(float(obs['t_normalized']))
    vec.append(float(obs['depot_dist']))
    vec.append(float(obs['capacity']))
    vec.append(float(obs['available_count']))
    vec.append(float(obs['my_accepted_dist']))
    vec.append(float(obs['my_accepted_remaining_tw']))
    vec.append(float(obs['dist_travel_norm']))
    for dist, st, rem_tw, rv in zip(obs['order_dists'], obs['order_statuses'],
                                   obs['order_remaining_tws'], obs['order_rewards']):
        vec.append(float(dist))
        vec.append(float(st))
        vec.append(float(rem_tw))
        vec.append(float(rv))
    for dist in obs['other_dists']:
        vec.append(float(dist))
    return np.array(vec, dtype=np.float32)



class IPPOAgent:
    def __init__(self, input_dim: int, action_dim: int, lr: float = 3e-4, clip_eps: float = 0.2, vf_coeff: float = 0.5,
                 ent_coeff: float = 0.01):
        self.policy = PolicyNet(input_dim, action_dim)
        self.value = ValueNet(input_dim)
        self.opt = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)
        self.clip_eps = clip_eps
        self.vf_coeff = vf_coeff
        self.ent_coeff = ent_coeff
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy.to(self.device)
        self.value.to(self.device)

    def act(self, obs_vec: np.ndarray, mask: List[int]):
        obs_t = torch.from_numpy(obs_vec).float().unsqueeze(0).to(self.device)
        logits = self.policy(obs_t)  # shape (1, action_dim)
        # ensure mask has at least one True; if none, force 'stay' (index 2) to be allowed
        mask_arr = np.array(mask, dtype=bool)
        if mask_arr.sum() == 0:
            mask_arr[2] = True
        mask_t = torch.tensor(mask_arr, dtype=torch.bool, device=self.device).unsqueeze(0)  # (1, action_dim)
        large_neg = torch.tensor(-1e9, device=self.device, dtype=logits.dtype)
        logits_masked = torch.where(mask_t, logits, large_neg)
        # numeric safety: if all masked logits are very negative -> softmax could underflow; handle explicitly
        probs = torch.softmax(logits_masked, dim=-1)
        probs_np = probs.detach().cpu().numpy()[0]
        # if numeric issue (sum zero / nan), fallback to uniform over valid actions
        if not np.isfinite(probs_np).all() or probs_np.sum() == 0:
            valid_inds = np.where(mask_arr)[0]
            if len(valid_inds) == 0:
                # as a last resort, pick stay
                chosen = 2
                logp = math.log(1.0)
                value = float(self.value(obs_t).detach().cpu().numpy()[0])
                return int(chosen), float(logp), float(value), np.zeros_like(probs_np)
            uni = np.zeros_like(probs_np)
            uni[valid_inds] = 1.0 / len(valid_inds)
            probs_np = uni
            probs = torch.from_numpy(probs_np).float().unsqueeze(0).to(self.device)
        m = Categorical(probs)
        a = m.sample()
        logp = m.log_prob(a)
        value = self.value(obs_t)
        return int(a.item()), float(logp.item()), float(value.item()), probs.detach().cpu().numpy()[0]

    def compute_loss(self, obs_batch, act_batch, old_logp_batch, ret_batch, adv_batch, mask_batch):
        obs_t = torch.from_numpy(obs_batch).float().to(self.device)
        acts = torch.from_numpy(act_batch).long().to(self.device)
        oldlogp = torch.from_numpy(old_logp_batch).float().to(self.device)
        rets = torch.from_numpy(ret_batch).float().to(self.device)
        advs = torch.from_numpy(adv_batch).float().to(self.device)
        mask_t = torch.from_numpy(mask_batch).to(self.device).bool()  # ensure bool

        # ensure no row of mask_t is all False: set 'stay' (index 2) to True for such rows
        if mask_t.dim() == 2:
            all_false_rows = (~mask_t).all(dim=1)
            if all_false_rows.any():
                mask_t[all_false_rows, 2] = True

        logits = self.policy(obs_t)  # (B, A)
        large_neg = torch.full_like(logits, -1e9)
        logits_masked = torch.where(mask_t, logits, large_neg)
        dist = Categorical(logits=logits_masked)
        logp = dist.log_prob(acts)
        ratio = torch.exp(logp - oldlogp)
        surr1 = ratio * advs
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advs
        policy_loss = -torch.mean(torch.min(surr1, surr2))
        values = self.value(obs_t)
        value_loss = torch.mean((rets - values) ** 2)
        entropy = torch.mean(dist.entropy())
        loss = policy_loss + self.vf_coeff * value_loss - self.ent_coeff * entropy
        return loss, policy_loss.item(), value_loss.item(), entropy.item()

    def update(self, trajectories: List[Transition], mask_list: List[List[int]], epochs: int = 4, batch_size: int = 64,
               gamma: float = 0.99, lam: float = 0.95):
        if not trajectories:
            return
        obs = np.vstack([t.obs for t in trajectories])
        acts = np.array([t.action for t in trajectories], dtype=np.int64)
        oldlogp = np.array([t.logp for t in trajectories], dtype=np.float32)
        rews = np.array([t.reward for t in trajectories], dtype=np.float32)
        dones = np.array([t.done for t in trajectories], dtype=np.float32)
        vals = np.array([t.value for t in trajectories], dtype=np.float32)

        N = len(rews)
        returns = np.zeros_like(rews)
        advs = np.zeros_like(rews)
        lastgaelam = 0.0
        nextval = 0.0
        for t in reversed(range(N)):
            nonterminal = 1.0 - dones[t]
            delta = rews[t] + gamma * nextval * nonterminal - vals[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            advs[t] = lastgaelam
            nextval = vals[t]
        returns = advs + vals
        if advs.std() > 1e-8:
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        mask_arr = np.array(mask_list, dtype=bool)
        # ensure no mask row is all False -> set stay (2) True
        if mask_arr.ndim == 2:
            rows_all_false = (~mask_arr).all(axis=1)
            if rows_all_false.any():
                mask_arr[rows_all_false, 2] = True

        inds = np.arange(N)
        for _ in range(epochs):
            np.random.shuffle(inds)
            for start in range(0, N, batch_size):
                mb = inds[start:start + batch_size]
                loss, pl, vl, ent = self.compute_loss(obs[mb], acts[mb], oldlogp[mb], returns[mb], advs[mb],
                                                      mask_arr[mb])
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), 0.5)
                self.opt.step()


