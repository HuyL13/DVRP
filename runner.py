import os
import time
from typing import Optional

import numpy as np
import torch

from agent.IPPOAgent import flatten_observation, IPPOAgent
from agent.Transition import Transition
from dvrp_env import MultiDVRPEnv
from load_checkpoint import load_agent
from logger import IPPOLogger, TENSORBOARD_AVAILABLE


def run_ippo(env: MultiDVRPEnv,
             num_iterations: int = 5000,
             rollout_steps: int = 512,
             epochs: int = 4,
             batch_size: int = 256,
             print_freq: int = 10,
             save_freq: int = 50,
             save_dir: str = "models",
             load_path: Optional[str] = None):      # <<< NEW

    sample_obs = env.reset()[0]
    sample_vec = flatten_observation(sample_obs)
    input_dim = sample_vec.shape[0]
    action_dim = env.N + 4

    # ------------------------------------------------------------------
    # 1. Create agents
    # ------------------------------------------------------------------
    agents = [IPPOAgent(input_dim, action_dim) for _ in range(env.M)]
    for i, a in enumerate(agents):
        a.agent_id = i                     # give each agent an id for loading

    # ------------------------------------------------------------------
    # 2. Create loggers
    # ------------------------------------------------------------------
    loggers = [IPPOLogger(i, use_tensorboard=TENSORBOARD_AVAILABLE) for i in range(env.M)]

    # ------------------------------------------------------------------
    # 3. OPTIONAL: Load checkpoint
    # ------------------------------------------------------------------
    start_iteration = 1
    if load_path is not None:
        print(f"\n=== Loading checkpoint from {load_path} ===")
        for aid, agent in enumerate(agents):
            it, ts = load_agent(agent,
                                load_dir=load_path,
                                device=agent.device)

            # initialise logger counters
            loggers[aid].i_so_far = it
            loggers[aid].t_so_far = ts
            print(f"  Agent {aid}: iteration={it}, timesteps={ts}")

        start_iteration = max(loggers[aid].i_so_far for aid in range(env.M)) + 1
        print(f"Resuming training at iteration {start_iteration}\n")

    # ------------------------------------------------------------------
    # 4. Normal training loop
    # ------------------------------------------------------------------
    print(f"IPPO start: agents={env.M}, obs_dim={input_dim}, action_dim={action_dim}")
    print(f"TensorBoard: {'Enabled' if TENSORBOARD_AVAILABLE else 'Disabled (CSV only)'}")
    print(f"Logging to: logs/")
    print("=" * 80)

    start_time = time.time()
    agent_losses = [{'policy': [], 'value': [], 'entropy': []} for _ in range(env.M)]

    for it in range(start_iteration, start_iteration + num_iterations):
        # ---------- rollout ----------
        trajs = [[] for _ in range(env.M)]
        mask_lists = [[] for _ in range(env.M)]
        obs_vectors = [[] for _ in range(env.M)]

        obs_all = env.reset()
        for step in range(rollout_steps):
            actions = {}
            step_info = {}
            for aid in range(env.M):
                obs = obs_all[aid]
                vec = flatten_observation(obs)
                obs_vectors[aid].append(vec)

                # optional on-the-fly normalisation
                if len(obs_vectors[aid]) > 1:
                    arr = np.vstack(obs_vectors[aid])
                    vec = (vec - arr.mean(axis=0)) / (arr.std(axis=0) + 1e-8)

                mask = env.get_action_mask(aid)
                a, logp, val, _ = agents[aid].act(vec, mask)
                actions[aid] = a
                step_info[aid] = (vec, a, logp, val, mask)

            next_obs, rewards, dones, infos = env.step(actions)

            for aid in range(env.M):
                vec, a, logp, val, mask = step_info[aid]
                trajs[aid].append(Transition(obs=vec,
                                            action=a,
                                            logp=logp,
                                            reward=rewards[aid],
                                            done=float(dones[aid]),
                                            value=val))
                mask_lists[aid].append(mask)

            obs_all = next_obs
            if all(dones.values()):
                break

        # ---------- update ----------
        for aid in range(env.M):
            agents[aid].update(trajs[aid], mask_lists[aid],
                               epochs=epochs, batch_size=batch_size)

            # ----- loss logging  -----
            if trajs[aid]:
                # recompute returns/advs on the full rollout for logging
                obs = np.vstack([t.obs for t in trajs[aid]])
                acts = np.array([t.action for t in trajs[aid]], dtype=np.int64)
                oldlogp = np.array([t.logp for t in trajs[aid]], dtype=np.float32)
                rews = np.array([t.reward for t in trajs[aid]], dtype=np.float32)
                dones_arr = np.array([t.done for t in trajs[aid]], dtype=np.float32)
                vals = np.array([t.value for t in trajs[aid]], dtype=np.float32)

                N = len(rews)
                returns = np.zeros_like(rews)
                advs = np.zeros_like(rews)
                lastgaelam = 0.0
                nextval = 0.0
                for t in reversed(range(N)):
                    nonterminal = 1.0 - dones_arr[t]
                    delta = rews[t] + 0.99 * nextval * nonterminal - vals[t]
                    lastgaelam = delta + 0.99 * 0.95 * nonterminal * lastgaelam
                    advs[t] = lastgaelam
                    nextval = vals[t]
                returns = advs + vals
                if advs.std() > 1e-8:
                    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

                mask_arr = np.array(mask_lists[aid], dtype=bool)
                rows_all_false = (~mask_arr).all(axis=1)
                if rows_all_false.any():
                    mask_arr[rows_all_false, 2] = True

                _, pl, vl, ent = agents[aid].compute_loss(obs, acts, oldlogp,
                                                         returns, advs, mask_arr)
                agent_losses[aid]['policy'].append(pl)
                agent_losses[aid]['value'].append(vl)
                agent_losses[aid]['entropy'].append(ent)

        # ---------- logging ----------
        for aid in range(env.M):
            avg_pl = np.mean(agent_losses[aid]['policy']) if agent_losses[aid]['policy'] else 0.0
            avg_vl = np.mean(agent_losses[aid]['value']) if agent_losses[aid]['value'] else 0.0
            avg_ent = np.mean(agent_losses[aid]['entropy']) if agent_losses[aid]['entropy'] else 0.0

            lr = agents[aid].opt.param_groups[0]['lr']
            metrics = loggers[aid].log_iteration(env, avg_pl, avg_vl, avg_ent, lr)

            if it % print_freq == 0 or it == start_iteration:
                loggers[aid].print_summary(metrics)

            # reset loss buffers for next iteration
            agent_losses[aid] = {'policy': [], 'value': [], 'entropy': []}

        # ---------- overall stats ----------
        if it % print_freq == 0 or it == start_iteration:
            avg_total = np.mean(env.total_rewards_per_agent)
            success_rate = env.successful_deliveries / max(1, env.total_order)
            total_dist = sum(env.dist_travel)
            avg_dist = total_dist / env.M
            avg_orders_per_step = env.total_order / env.t if env.t > 0 else 0.0
            objective = 0.5 * (10 * (1 - success_rate) + (total_dist / env.M) / 480)

            print("=" * 80)
            print(f"ITERATION {it} (global) - OVERALL")
            print("=" * 80)
            print(f"Avg total reward : {avg_total:.3f}")
            print(f"Success rate     : {success_rate:.3f}")
            print(f"Avg distance     : {avg_dist:.3f}")
            print(f"Orders/step      : {avg_orders_per_step:.3f}")
            print(f"Counter-objective: {objective:.5f}")
            print(f"Elapsed time     : {time.time() - start_time:.1f}s")
            print("=" * 80)

        # ---------- checkpoint ----------
        if it % save_freq == 0:
            ckpt_dir = f"/content/drive/MyDrive/models_ippo/iter_{it+250}"
            os.makedirs(ckpt_dir, exist_ok=True)
            for aid, a in enumerate(agents):
                agent_dir = os.path.join(ckpt_dir, f"agent_{aid}")
                os.makedirs(agent_dir, exist_ok=True)

                torch.save(a.policy.state_dict(),
                           os.path.join(agent_dir, "policy.pth"))
                torch.save(a.value.state_dict(),
                           os.path.join(agent_dir, "value.pth"))
                torch.save(a.opt.state_dict(),
                           os.path.join(agent_dir, "optimizer.pth"))

                # save logger counters so we can resume exactly
                torch.save({
                    "iteration": loggers[aid].i_so_far,
                    "timesteps": loggers[aid].t_so_far
                }, os.path.join(agent_dir, "meta.pth"))

            print(f"Checkpoint saved to {ckpt_dir}")

    # ------------------- final save -------------------
        final_dir = "/content/drive/MyDrive/models_ippo/final"
        os.makedirs(final_dir, exist_ok=True)

        for aid, agent in enumerate(agents):
            agent_dir = os.path.join(final_dir, f"agent_{aid}")
            os.makedirs(agent_dir, exist_ok=True)

            torch.save(agent.policy.state_dict(),
                      os.path.join(agent_dir, "policy.pth"))
            torch.save(agent.value.state_dict(),
                      os.path.join(agent_dir, "value.pth"))
            torch.save(agent.opt.state_dict(),
                      os.path.join(agent_dir, "optimizer.pth"))

            torch.save({
                "iteration": loggers[aid].i_so_far,
                "timesteps": loggers[aid].t_so_far
            }, os.path.join(agent_dir, "meta.pth"))

    print(f"Final models saved to {final_dir} ")

    for logger in loggers:
        logger.close()

    return agents

if __name__ == "__main__":

    env = MultiDVRPEnv(seed=42)


    agents = run_ippo(
        env,
        num_iterations=5000,
        rollout_steps=480,
        print_freq=10,
        save_freq=50,
        load_path=None
    )

    print("Continued training finished!")