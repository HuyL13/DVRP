import os
import time

import numpy as np
import torch

from IPPOAgent import IPPOAgent, flatten_observation, Transition
from dvrp_env import MultiDVRPEnv


def run_ippo(env: MultiDVRPEnv,
             num_iterations: int = 5000,
             rollout_steps: int = 480,
             epochs: int = 4,
             batch_size: int = 256):

    sample_obs = env.reset()[0]
    sample_vec = flatten_observation(sample_obs)
    input_dim = sample_vec.shape[0]
    action_dim = env.N + 4

    agents = [IPPOAgent(input_dim, action_dim) for _ in range(env.M)]
    print(f"IPPO start: agents={env.M}, obs_dim={input_dim}, action_dim={action_dim}")



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

        for aid in range(env.M):
            agents[aid].update(trajs[aid], mask_lists[aid], epochs=epochs, batch_size=batch_size)

        avg_total = np.mean(env.total_rewards_per_agent)



        success_rate = env.successful_deliveries / max(1, env.total_order) if env.total_order > 0 else 0.0


        if it % 10 == 0 or it == 1:
            sum = 0
            print("Each agent avg reward this rollout (full episode 480 timestep): ")
            for vid in range(env.M):
                print(
                    f"Reward of agent {vid}: {env.total_rewards_per_agent[vid]}| with acp: {env.acp_component[vid]} and deliver: {env.deliver_component[vid]} and pen: {env.penalty[vid]} and cost: {env.avg_cost[vid]} ")
                sum += env.dist_travel[vid]

            avg_orders_per_step = env.total_order / env.t if env.t > 0 else 0.0

            print(
                f"Iter {it}/{num_iterations} | avg reward: {avg_total:.3f} | success rate: {success_rate:.3f} |  avg distance: {sum / env.M}| avg orders/step: {avg_orders_per_step:.3f} |")
            print(
                "-------------------------------------------------------------------------------------------------------------------------------------")

    return agents


if __name__ == "__main__":
    env = MultiDVRPEnv(seed=42)
    start = time.time()
    agents = run_ippo(env, num_iterations=5000, rollout_steps=480)
    print("Training finished in {:.1f}s".format(time.time() - start))
    os.makedirs("models", exist_ok=True)
    for i, a in enumerate(agents):
        torch.save(a.policy.state_dict(), f"models/agent_{i}_policy.pth")
        torch.save(a.value.state_dict(), f"models/agent_{i}_value.pth")