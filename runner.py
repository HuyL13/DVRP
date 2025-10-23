import os
import time

import numpy as np
import torch

from IPPOAgent import IPPOAgent, flatten_observation, Transition
from dvrp_env import MultiDVRPEnv
from logger import IPPOLogger, TENSORBOARD_AVAILABLE


# -------------------------
# Runner with Logging
# -------------------------
def run_ippo(env: MultiDVRPEnv,
             num_iterations: int = 5000,
             rollout_steps: int = 512,
             epochs: int = 4,
             batch_size: int = 256,
             print_freq: int = 10,
             save_freq: int = 50,
             save_dir: str = "models"):
    sample_obs = env.reset()[0]
    sample_vec = flatten_observation(sample_obs)
    input_dim = sample_vec.shape[0]
    action_dim = env.N + 4

    agents = [IPPOAgent(input_dim, action_dim) for _ in range(env.M)]
    loggers = [IPPOLogger(i, use_tensorboard=TENSORBOARD_AVAILABLE) for i in range(env.M)]

    print(f"IPPO start: agents={env.M}, obs_dim={input_dim}, action_dim={action_dim}")
    print(f"TensorBoard: {'Enabled' if TENSORBOARD_AVAILABLE else 'Disabled (CSV only)'}")
    print(f"Logging to: logs/")
    print("=" * 80)

    start_time = time.time()

    # For tracking losses
    agent_losses = [{'policy': [], 'value': [], 'entropy': []} for _ in range(env.M)]

    for it in range(1, num_iterations + 1):
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
                if len(obs_vectors[aid]) > 1:
                    obs_array = np.vstack(obs_vectors[aid])
                    mean = obs_array.mean(axis=0)
                    std = obs_array.std(axis=0) + 1e-8
                    vec = (vec - mean) / std
                mask = env.get_action_mask(aid)
                a, logp, val, probs = agents[aid].act(vec, mask)
                actions[aid] = a
                step_info[aid] = (vec, a, logp, val, mask)
            next_obs, rewards, dones, infos = env.step(actions)

            for aid in range(env.M):
                vec, a, logp, val, mask = step_info[aid]
                trajs[aid].append(
                    Transition(obs=vec, action=a, logp=logp, reward=rewards[aid], done=float(dones[aid]), value=val))
                mask_lists[aid].append(mask)
            obs_all = next_obs
            if all(dones.values()):
                break

        # Update agents and collect losses
        for aid in range(env.M):
            # Store old parameters to track losses during update
            old_policy = agents[aid].policy.state_dict()

            agents[aid].update(trajs[aid], mask_lists[aid], epochs=epochs, batch_size=batch_size)

            # Compute average losses for this iteration (approximate)
            if trajs[aid]:
                obs = np.vstack([t.obs for t in trajs[aid]])
                acts = np.array([t.action for t in trajs[aid]], dtype=np.int64)
                oldlogp = np.array([t.logp for t in trajs[aid]], dtype=np.float32)
                rews = np.array([t.reward for t in trajs[aid]], dtype=np.float32)
                dones_arr = np.array([t.done for t in trajs[aid]], dtype=np.float32)
                vals = np.array([t.value for t in trajs[aid]], dtype=np.float32)

                # Compute returns and advantages
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
                if mask_arr.ndim == 2:
                    rows_all_false = (~mask_arr).all(axis=1)
                    if rows_all_false.any():
                        mask_arr[rows_all_false, 2] = True

                # Compute loss on full batch for logging
                _, pl, vl, ent = agents[aid].compute_loss(obs, acts, oldlogp, returns, advs, mask_arr)
                agent_losses[aid]['policy'].append(pl)
                agent_losses[aid]['value'].append(vl)
                agent_losses[aid]['entropy'].append(ent)

        # Log each agent
        for aid in range(env.M):
            avg_policy_loss = np.mean(agent_losses[aid]['policy']) if agent_losses[aid]['policy'] else 0.0
            avg_value_loss = np.mean(agent_losses[aid]['value']) if agent_losses[aid]['value'] else 0.0
            avg_entropy = np.mean(agent_losses[aid]['entropy']) if agent_losses[aid]['entropy'] else 0.0

            lr = agents[aid].opt.param_groups[0]['lr']
            metrics = loggers[aid].log_iteration(env, avg_policy_loss, avg_value_loss, avg_entropy, lr)

            if it % print_freq == 0 or it == 1:
                loggers[aid].print_summary(metrics)

            # Clear losses for next iteration
            agent_losses[aid] = {'policy': [], 'value': [], 'entropy': []}

        # Print overall statistics
        if it % print_freq == 0 or it == 1:
            avg_total = np.mean(env.total_rewards_per_agent)
            success_rate = env.successful_deliveries / max(1, env.total_order)
            total_dist = sum(env.dist_travel)
            avg_dist = total_dist / env.M
            avg_orders_per_step = env.total_order / env.t if env.t > 0 else 0.0
            objective = 0.5 * (10 * (1 - success_rate) + (total_dist / env.M) / 480)

            print("=" * 80)
            print(f"ITERATION {it}/{num_iterations} - OVERALL STATISTICS")
            print("=" * 80)
            print(f"Average Total Reward: {avg_total:.3f}")
            print(f"Success Rate: {success_rate:.3f}")
            print(f"Average Distance: {avg_dist:.3f}")
            print(f"Average Orders/Step: {avg_orders_per_step:.3f}")
            print(f"Counter-Objective Function: {objective:.5f}")
            print(f"Elapsed Time: {time.time() - start_time:.1f}s")
            print("=" * 80)
            print()

        # Save models
        if it % save_freq == 0:
            os.makedirs(save_dir, exist_ok=True)
            for i, a in enumerate(agents):
                agent_save_dir = os.path.join(save_dir, f"iter_{it}", f"agent_{i}")
                os.makedirs(agent_save_dir, exist_ok=True)
                torch.save(a.policy.state_dict(), os.path.join(agent_save_dir, "policy.pth"))
                torch.save(a.value.state_dict(), os.path.join(agent_save_dir, "value.pth"))
            print(f"Models saved at iteration {it}")

    # Close all loggers
    for logger in loggers:
        logger.close()

    return agents


if __name__ == "__main__":
    env = MultiDVRPEnv(seed=42)
    start = time.time()
    agents = run_ippo(env, num_iterations=3000, rollout_steps=480, print_freq=10, save_freq=50)
    print("Training finished in {:.1f}s".format(time.time() - start))

    # Final save
    os.makedirs("models/final", exist_ok=True)
    for i, a in enumerate(agents):
        torch.save(a.policy.state_dict(), f"models/final/agent_{i}_policy.pth")
        torch.save(a.value.state_dict(), f"models/final/agent_{i}_value.pth")
    print("Final models saved to models/final/")