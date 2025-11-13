# --------------------------------------------------------------
#  IPPO VISUALIZER – LIVE ANIMATION (moving vehicles & orders)
# --------------------------------------------------------------

import os
import time
from typing import List

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
import torch

# --------------------------------------------------------------
#  NOTE:  In a Jupyter notebook put the following line **once**
#        at the top of the cell:
#        %matplotlib notebook
# --------------------------------------------------------------
# In a plain .py script the line below is enough:
plt.ion()                     # <-- interactive mode for scripts

from agent.IPPOAgent import flatten_observation
from dvrp_env import MultiDVRPEnv
from load_checkpoint import load_ippo_agent


# ------------------------------------------------------------------
#  Helper – create a *single* figure that will be updated every step
# ------------------------------------------------------------------
def _init_figure(env: MultiDVRPEnv):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_xlim(-0.5, env.grid_size - 0.5)
    ax.set_ylim(-0.5, env.grid_size - 0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(env.grid_size))
    ax.set_yticks(range(env.grid_size))
    ax.grid(True, linewidth=0.5, alpha=0.5)
    ax.set_title("IPPO Multi-Agent DVRP – Live Simulation", fontsize=14)
    return fig, ax


# ------------------------------------------------------------------
#  Visualiser class – now uses a *persistent* Matplotlib canvas
# ------------------------------------------------------------------
class DVRPVisualizerIPPO:
    def __init__(self, env: MultiDVRPEnv,
                 policies: List[torch.nn.Module],
                 values: List[torch.nn.Module],
                 device: torch.device,
                 speed: float = 0.25,
                 use_normalization: bool = False):
        self.env = env
        self.policies = policies
        self.values = values
        self.device = device
        self.speed = speed
        self.use_normalization = use_normalization

        self.M = env.M
        self.normalisation_buffer = [[] for _ in range(self.M)]

        # ---- persistent Matplotlib objects --------------------------------
        self.fig, self.ax = _init_figure(env)

        # pre-allocate artists that will be updated (vehicles & orders)
        self.vehicle_patches = [Circle((0, 0), 0.35) for _ in range(self.M)]
        self.vehicle_texts   = [self.ax.text(0, 0, "", ha='center', va='center',
                                            color='white', fontsize=11, weight='bold')
                               for _ in range(self.M)]
        for p in self.vehicle_patches:
            self.ax.add_patch(p)

        self.order_scats = self.ax.scatter([], [], s=[], marker='o',
                                          edgecolors=[], facecolors='gold', linewidths=1.5)
        self.order_texts = []                     # created on-the-fly

        # depot (static)
        depot_x, depot_y = env.depot
        self.depot_patch = Rectangle((depot_x-0.4, depot_y-0.4), 0.8, 0.8,
                                     facecolor='red', edgecolor='black')
        self.ax.add_patch(self.depot_patch)
        self.depot_text = self.ax.text(depot_x, depot_y, 'D',
                                       ha='center', va='center',
                                       color='white', fontsize=12, weight='bold')

        # info panel (text object)
        self.info_text = self.ax.text(0.02, 0.98, "", transform=self.ax.transAxes,
                                      va='top', fontsize=9.5,
                                      bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9))

        plt.tight_layout()

    # ------------------------------------------------------------------
    def _draw_frame(self):
        """Update *only* the parts that changed."""
        env = self.env
        M   = self.M

        # ---- 1. Depot (static) -----------------------------------------
        # already added, nothing to do

        # ---- 2. Vehicles ------------------------------------------------
        colors = plt.cm.tab10(np.linspace(0, 1, M))
        for i in range(M):
            x, y = env.driver_locs[i]
            self.vehicle_patches[i].center = (x, y)
            self.vehicle_patches[i].set_color(colors[i])
            self.vehicle_texts[i].set_position((x, y))
            self.vehicle_texts[i].set_text(str(i))

        # ---- 3. Orders --------------------------------------------------
        max_r = max(b for _, b in env.zone_reward_ranges)

        xs, ys, sizes, edge_colors, markers = [], [], [], [], []
        order_texts = []                     # will be re-created each frame

        for order in env.orders:
            if order is None or order.status == 0:
                continue
            x, y = order.loc
            size = 200 + 300 * (order.reward / max_r)

            xs.append(x)
            ys.append(y)
            sizes.append(size)
            edge_colors.append('green' if order.status == 2 else 'black')
            markers.append('*' if order.status == 2 else 'o')

            # reward label slightly above the marker
            txt = self.ax.text(x, y + 0.3, f"{order.reward:.1f}",
                               ha='center', fontsize=8, color='darkred')
            order_texts.append(txt)

        # update scatter (fast path – avoids recreating the artist)
        self.order_scats.set_offsets(np.c_[xs, ys])
        self.order_scats.set_sizes(sizes)
        self.order_scats.set_edgecolors(edge_colors)
        self.order_scats.set_facecolors('gold')
        self.order_scats.set_array(np.zeros(len(xs)))   # dummy for marker style
        # marker style cannot be changed per-point with a single Scatter,
        # so we remove old texts and add new ones
        for txt in self.order_texts:
            txt.remove()
        self.order_texts = order_texts

        # ---- 4. Info panel -----------------------------------------------
        lines = [
            f"Time: {env.t}/{env.T}",
            f"Orders: {env.total_order} | Delivered: {env.successful_deliveries}",
            f"Success: {env.successful_deliveries/max(1,env.total_order):.3f}",
        ]
        for i in range(M):
            lines.append(f"A{i} R: {env.total_rewards_per_agent[i]:.1f} | D: {env.dist_travel[i]:.0f}")

        self.info_text.set_text("\n".join(lines))

        # ---- 5. Refresh --------------------------------------------------
        self.fig.canvas.draw_idle()
        plt.pause(0.001)                 # tiny pause to let GUI breathe

    # ------------------------------------------------------------------
    def step_and_draw(self) -> bool:
        """One environment step → actions → env.step() → redraw."""
        actions = {}
        obs_dict = {i: self.env._get_obs_for_agent(i) for i in range(self.M)}

        for aid in range(self.M):
            vec = flatten_observation(obs_dict[aid])

            # ---- optional on-the-fly normalisation --------------------
            if self.use_normalization and len(self.normalisation_buffer[aid]) > 1:
                arr = np.vstack(self.normalisation_buffer[aid])
                vec = (vec - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-8)
            self.normalisation_buffer[aid].append(vec.copy())
            if len(self.normalisation_buffer[aid]) > 1000:
                self.normalisation_buffer[aid].pop(0)

            # ---- action mask -------------------------------------------
            mask = self.env.get_action_mask(aid)
            mask_arr = np.array(mask, dtype=bool)
            if mask_arr.sum() == 0:
                mask_arr[2] = True                     # stay as fallback

            # ---- policy forward pass -----------------------------------
            with torch.no_grad():
                state_t = torch.from_numpy(vec).float().unsqueeze(0).to(self.device)
                logits = self.policies[aid](state_t)
                large_neg = torch.full_like(logits, -1e9)
                logits_masked = torch.where(torch.tensor(mask_arr, device=self.device),
                                            logits, large_neg)
                probs = torch.softmax(logits_masked, dim=-1)
                action = torch.multinomial(probs, 1).item()

            actions[aid] = action

        # ---- environment step -------------------------------------------
        _, rewards, dones, _ = self.env.step(actions)

        # ---- redraw ----------------------------------------------------
        self._draw_frame()

        return all(dones.values())

    # ------------------------------------------------------------------
    def run(self):
        self.env.reset()
        print("Starting LIVE IPPO visualisation … (Ctrl+C to stop)")
        step = 0
        try:
            while True:
                done = self.step_and_draw()
                step += 1
                time.sleep(self.speed)
                if done:
                    break
        except KeyboardInterrupt:
            print("\nStopped by user.")

        # final static frame
        self._draw_frame()
        print(f"\nEpisode finished after {step} steps.")
        self.print_final_stats()

    # ------------------------------------------------------------------
    def print_final_stats(self):
        env = self.env
        success_rate = env.successful_deliveries / max(1, env.total_order)
        total_dist = sum(env.dist_travel)
        print("\n" + "="*70)
        print(" " * 20 + "FINAL EPISODE STATISTICS")
        print("="*70)
        print(f"Total Orders Generated      : {env.total_order}")
        print(f"Successful Deliveries       : {env.successful_deliveries}")
        print(f"Success Rate                : {success_rate:.3f}")
        print(f"Total Distance Traveled     : {total_dist:.1f}")
        print(f"Avg Distance per Agent      : {total_dist/env.M:.1f}")
        print(f"Objective (lower better)    : {0.5 * (10 * (1 - success_rate) + (total_dist / env.M) / 480):.4f}")
        print("-" * 70)
        print("Per-Agent Breakdown:")
        for i in range(env.M):
            print(f"  Agent {i}: "
                  f"Reward={env.total_rewards_per_agent[i]:6.1f} | "
                  f"Dist={env.dist_travel[i]:5.0f} | "
                  f"Accept={env.acp_component[i]/0.333:.1f} | "
                  f"Deliver={env.deliver_component[i]/0.667:.1f}")
        print("="*70)


# ==============================================================
#  MAIN – load models and start the live animation
# ==============================================================
if __name__ == "__main__":
    # ---- 1. Environment (must match training) -----------------------
    env = MultiDVRPEnv(
        n_agents=3,
        grid_size=10,
        N_max=10,
        C=10,
        T=480,
        seed=42
    )

    # ---- 2. Observation / action dimensions ------------------------
    sample_obs = env.reset()[0]
    obs_vec = flatten_observation(sample_obs)
    input_dim = obs_vec.shape[0]
    action_dim = env.N + 4

    # ---- 3. Load IPPO agents ---------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_root = "checkpoint/final"          # <-- change if you stored elsewhere

    policies, values = [], []
    for i in range(env.M):
        agent_dir = os.path.join(load_root, f"agent_{i}")
        if not os.path.isdir(agent_dir):
            raise FileNotFoundError(f"Missing folder: {agent_dir}")
        print(f"Loading agent {i} ← {agent_dir}")
        pol, val = load_ippo_agent(agent_dir, input_dim, action_dim, device)
        policies.append(pol)
        values.append(val)

    print(f"\nAll {env.M} agents loaded – device: {device}")

    # ---- 4. Run live visualiser ------------------------------------
    viz = DVRPVisualizerIPPO(env, policies, values, device,
                             speed=0.3, use_normalization=False)
    viz.run()