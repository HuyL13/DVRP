import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np


# -------------------------
# Environment for multi agents performing Dynamic Vehicle Routing Task
# with reward shaping and success tracking
# -------------------------

@dataclass
class Order:
    loc: Tuple[int, int]
    status: int         # 0 inactive, 1 available, 2 accepted/in-process
    elapsed: int        # elapsed time since generation
    reward: float       # v_i
    tw: int             # time window (deadline)
    gen_time: int = 0   # absolute time when generated
    accepted_by: Optional[int] = None  # agent id who accepted (for bookkeeping)

class MultiDVRPEnv:
    def __init__(self,
                 n_agents: int = 3,
                 grid_size: int = 10,
                 N_max: int = 10,
                 C: int = 10,
                 T: int = 480,
                 depot_loc: Optional[Tuple[int, int]] = None,
                 p_arrival: float = 0.25,
                 zone_probs: Optional[List[float]] = None,
                 zone_reward_ranges: Optional[List[Tuple[float, float]]] = None,
                 alpha: float = 1/3,
                 penalty_f: float = 50.0,
                 tw: int = 60,
                 m: float = 0.1,
                 seed: Optional[int] = None):
        self.M = n_agents
        self.grid_size = grid_size
        self.N = N_max
        self.C_max = C
        self.T = T

        if depot_loc is None:
            cx = math.ceil(self.grid_size / 2)
            cy = math.ceil(self.grid_size / 2)
            self.depot = (cx, cy)
        else:
            self.depot = depot_loc

        self.p_arrival = p_arrival
        if zone_probs is None:
            self.zone_probs = [0.1, 0.4, 0.4, 0.1]
        else:
            assert len(zone_probs) == 4 and abs(sum(zone_probs) - 1.0) < 1e-6
            self.zone_probs = zone_probs

        if zone_reward_ranges is None:
            self.zone_reward_ranges = [(6, 10), (2, 4), (2, 4), (6, 10)]
        else:
            assert len(zone_reward_ranges) == 4
            self.zone_reward_ranges = zone_reward_ranges

        # Zone x-ranges
        n = self.grid_size
        n1 = int(math.floor(0.30 * n))
        n2 = int(math.floor(0.30 * n))
        n3 = int(math.floor(0.20 * n))
        n4 = n - (n1 + n2 + n3)
        if n1 <= 0: n1 = max(1, n // 4)
        if n2 <= 0: n2 = max(1, n // 4)
        if n3 <= 0: n3 = max(1, n // 8)
        n4 = n - (n1 + n2 + n3)
        x0 = 0
        x1 = x0 + n1 - 1
        x2 = x1 + n2
        x3 = x2 + n3
        x1 = min(x1, n-1)
        x2 = min(x2, n-1)
        x3 = min(x3, n-1)
        self.zone_x_ranges = [(0, x1), (x1 + 1, x2), (x2 + 1, x3), (x3 + 1, n-1)]
        for i in range(4):
            a, b = self.zone_x_ranges[i]
            a = max(0, min(a, n-1))
            b = max(0, min(b, n-1))
            if a > b:
                a, b = 0, n-1
            self.zone_x_ranges[i] = (a, b)

        self.alpha = alpha
        self.penalty_f = penalty_f
        self.tw = tw
        self.m_time = m / 2.0
        self.m_dist = m / 2.0

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        # Tracking
        self.accepted_count = 0
        self.successful_deliveries = 0
        self.total_order = 0
        self.acp_component = [0.0 for _ in range(self.M)]
        self.deliver_component = [0.0 for _ in range(self.M)]
        self.penalty = [0.0 for _ in range(self.M)]
        self.avg_cost = [0.0 for _ in range(self.M)]
        self.dist_travel = [0.0 for _ in range(self.M)]
        self.reward = [0.0 for _ in range(self.M)]
        self.immediate_rewards = [0.0 for _ in range(self.M)]
        max_order_reward = max(b for _, b in self.zone_reward_ranges)
        self.max_reward = max_order_reward  # Max reward from accept + deliver
        self.min_reward = -self.penalty_f - (self.m_time + self.m_dist * 2 * (self.grid_size - 1))  # Min reward: penalty + max move cost


        self.reset()

    def seed(self, s: int):
        self.rng.seed(s)
        self.np_rng = np.random.default_rng(s)

    def reset(self, start_time: int = 0) -> Dict[int, Dict[str, Any]]:
        self.t = start_time
        self.driver_locs: List[Tuple[int, int]] = [self.depot for _ in range(self.M)]
        self.capacities: List[int] = [self.C_max for _ in range(self.M)]
        self.orders: List[Optional[Order]] = [None for _ in range(self.N)]
        self.step_count = 0
        self.total_rewards_per_agent = [0.0 for _ in range(self.M)]
        self.last_info = {i: {} for i in range(self.M)}
        self.accepted_count = 0
        self.successful_deliveries = 0
        self.total_order = 0
        self.acp_component = [0.0 for _ in range(self.M)]
        self.deliver_component = [0.0 for _ in range(self.M)]
        self.penalty = [0.0 for _ in range(self.M)]
        self.avg_cost = [0.0 for _ in range(self.M)]
        self.dist_travel = [0.0 for _ in range(self.M)]
        self.reward = [0.0 for _ in range(self.M)]
        self.immediate_rewards = [0.0 for _ in range(self.M)]

        return {i: self._get_obs_for_agent(i) for i in range(self.M)}

    def _manhattan_distance(self, loc1: Tuple[int, int], loc2: Tuple[int, int]) -> int:
        return abs(loc1[0] - loc2[0]) + abs(loc1[1] - loc2[1])

    def _get_obs_for_agent(self, agent_id: int) -> Dict[str, Any]:
        driver_loc = self.driver_locs[agent_id]
        max_dist = 2 * (self.grid_size - 1) if self.grid_size > 1 else 1
        max_reward = max(b for _, b in self.zone_reward_ranges)

        depot_dist = self._manhattan_distance(driver_loc, self.depot) / max_dist if max_dist > 0 else 0.0

        order_dists = []
        order_remaining_tws = []
        order_statuses = []
        order_rewards = []
        for o in self.orders:
            if o and o.status != 0:
                dist = self._manhattan_distance(driver_loc, o.loc) / max_dist if max_dist > 0 else 0.0
                rem_tw = max(0, o.tw - o.elapsed) / self.tw if self.tw > 0 else 0.0
                reward = o.reward / max_reward if max_reward > 0 else 0.0
                status = o.status / 2.0
            else:
                dist = 0.0
                rem_tw = 0.0
                reward = 0.0
                status = 0.0
            order_dists.append(dist)
            order_remaining_tws.append(rem_tw)
            order_statuses.append(status)
            order_rewards.append(reward)

        my_accepted_idx = next((i for i, o in enumerate(self.orders) if o and o.status == 2 and o.accepted_by == agent_id), -1)
        my_accepted_dist = order_dists[my_accepted_idx] if my_accepted_idx != -1 else 0.0
        my_accepted_remaining_tw = order_remaining_tws[my_accepted_idx] if my_accepted_idx != -1 else 0.0

        other_dists = [self._manhattan_distance(driver_loc, self.driver_locs[j]) / max_dist if max_dist > 0 else 0.0
                       for j in range(self.M) if j != agent_id]

        available_count = sum(1 for o in self.orders if o and o.status == 1) / self.N if self.N > 0 else 0.0

        dist_travel_norm=self.dist_travel[agent_id]/(self.t+1) if self.t else 0

        obs = {
            't_normalized': self.t / self.T if self.T > 0 else 0.0,
            'depot_dist': depot_dist,
            'order_dists': order_dists,
            'order_statuses': order_statuses,
            'order_remaining_tws': order_remaining_tws,
            'order_rewards': order_rewards,
            'my_accepted_dist': my_accepted_dist,
            'my_accepted_remaining_tw': my_accepted_remaining_tw,
            'capacity': self.capacities[agent_id] / self.C_max if self.C_max > 0 else 0.0,
            'other_dists': other_dists,
            'available_count': available_count,
            'dist_travel_norm': dist_travel_norm,
        }
        return obs

    def _empty_order_slot_index(self) -> Optional[int]:
        for i, o in enumerate(self.orders):
            if o is None or o.status == 0:
                return i
        return None

    def _get_new_order(self) -> Order:
        self.total_order += 1
        z = random.choices([0, 1, 2, 3], weights=self.zone_probs, k=1)[0]
        xr = self.zone_x_ranges[z]
        x = self.rng.randrange(xr[0], xr[1] + 1) if xr[0] <= xr[1] else self.rng.randrange(0, self.grid_size)
        y = self.rng.randrange(0, self.grid_size)
        a, b = self.zone_reward_ranges[z]
        v = float(self.rng.uniform(a, b))

        # Time window = min(self.tw, remaining time to horizon)
        remaining_time = max(0, self.T-1- self.t)  # Time left until horizon
        effective_tw = min(self.tw, remaining_time)

        return Order(
            loc=(x, y),
            status=1,
            elapsed=0,
            reward=v,
            tw=effective_tw,  # Use effective time window
            gen_time=self.t
        )

    def _manhattan_step_towards(self, src: Tuple[int, int], dst: Tuple[int, int]) -> Tuple[int, int]:
        sx, sy = src
        dx, dy = dst
        # Move in only ONE direction per step (prioritize x, then y)
        if dx != sx:
            nx = sx + (1 if dx > sx else -1)
            ny = sy
        elif dy != sy:
            nx = sx
            ny = sy + (1 if dy > sy else -1)
        else:
            nx, ny = sx, sy
        nx = min(max(0, nx), self.grid_size - 1)
        ny = min(max(0, ny), self.grid_size - 1)
        return (nx, ny)

    def _available_newest_order_index(self) -> Optional[int]:
        newest_idx = None
        newest_time = -1
        for i, o in enumerate(self.orders):
            if o and o.status == 1 and o.gen_time > newest_time:
                newest_idx = i
                newest_time = o.gen_time
        return newest_idx

    def get_action_mask(self, agent_id: int) -> List[int]:
        mask = [0] * (self.N + 4)
        newest_idx = self._available_newest_order_index()
        if newest_idx is not None:
            if self.capacities[agent_id] >= 1:  # Order size is always 1
                mask[0] = 1  # accept newest
            mask[1] = 1  # reject newest
        mask[2] = 1  # stay
        mask[3] = 1  # move to depot
        for i in range(self.N):
            if self.orders[i] and self.orders[i].status == 2 and self.orders[i].accepted_by == agent_id:
                mask[4 + i] = 1  # move towards accepted order i
        return mask

    def step(self, actions: Dict[int, int]):
        rewards = {i: 0.0 for i in range(self.M)}
        infos = {i: {} for i in range(self.M)}
        dones = {i: False for i in range(self.M)}
        masks = {i: self.get_action_mask(i) for i in range(self.M)}
        for i in range(self.M):
            infos[i]['action_mask'] = masks[i].copy()

        # Phase 1: Accept/Reject
        accept_requests = []
        newest_idx = self._available_newest_order_index()
        if newest_idx is not None:
            for agent_id in range(self.M):
                act = actions.get(agent_id, 2)
                if act is None or act < 0 or act >= (self.N + 4):
                    infos[agent_id].setdefault('forced_action_due_to_out_of_range', []).append({'original': act, 'forced': 2})
                    act = 2
                    actions[agent_id] = act
                if masks[agent_id][act] == 0:
                    infos[agent_id].setdefault('forced_action_due_to_invalid_choice', []).append({'original': act, 'forced': 2})
                    act = 2
                    actions[agent_id] = act
                if act == 0 and self.capacities[agent_id] >= 1:
                    accept_requests.append(agent_id)
                elif act == 1:
                    infos[agent_id].setdefault('rejected_orders', []).append(newest_idx)
            if accept_requests:
                chosen_agent=-1
                min_dist=float('inf')
                order = self.orders[newest_idx]
                for vid in accept_requests:
                    if (min_dist>self._manhattan_distance(self.driver_locs[vid],order.loc)):
                        chosen_agent=vid
                        min_dist=self._manhattan_distance(self.driver_locs[vid],order.loc)

                order.status = 2
                order.accepted_by = chosen_agent
                self.capacities[chosen_agent] -= 1
                rewards[chosen_agent] += self.alpha * order.reward
                self.acp_component[chosen_agent] += self.alpha * order.reward
                infos[chosen_agent].setdefault('accepted_orders', []).append(newest_idx)
                self.accepted_count += 1

        # Phase 2: Routing
        new_locs = self.driver_locs.copy()
        move_dists = [0] * self.M
        for agent_id in range(self.M):
            act = actions.get(agent_id, 2)
            if act is None or act < 0 or act >= (self.N + 4):
                infos[agent_id].setdefault('forced_action_due_to_out_of_range', []).append({'original': act, 'forced': 2})
                act = 2
                actions[agent_id] = act
            if masks[agent_id][act] == 0:
                infos[agent_id].setdefault('forced_action_due_to_invalid_choice', []).append({'original': act, 'forced': 2})
                act = 2
                actions[agent_id] = act

            if newest_idx is not None and act in (0, 1):
                continue
            old_loc = self.driver_locs[agent_id]
            if act == 2:
                new_locs[agent_id] = old_loc
                move_dists[agent_id] = 0
            elif act == 3:
                nl = self._manhattan_step_towards(old_loc, self.depot)
                move_dists[agent_id] = self._manhattan_distance(old_loc, nl)
                new_locs[agent_id] = nl
            else:
                idx = act - 4
                if 0 <= idx < self.N:
                    order = self.orders[idx]
                    if order and order.status == 2 and order.accepted_by == agent_id:
                        nl = self._manhattan_step_towards(old_loc, order.loc)
                        move_dists[agent_id] = self._manhattan_distance(old_loc, nl)
                        new_locs[agent_id] = nl
                    else:
                        new_locs[agent_id] = old_loc
                        move_dists[agent_id] = 0
                else:
                    new_locs[agent_id] = old_loc
                    move_dists[agent_id] = 0
        self.driver_locs = new_locs

        # Deliveries
        for i, o in enumerate(self.orders):
            if o and o.status == 2:
                delivering_agents = [aid for aid, loc in enumerate(self.driver_locs) if loc == o.loc]
                if delivering_agents:
                    deliverer = delivering_agents[0]
                    if o.elapsed <= o.tw:
                        rewards[deliverer] += (1.0 - self.alpha) * o.reward
                        self.deliver_component[deliverer] += (1.0 - self.alpha) * o.reward
                        infos[deliverer].setdefault('delivered', []).append(i)
                        self.successful_deliveries += 1
                    else:
                        rewards[deliverer] -= self.penalty_f
                        self.penalty[deliverer] += self.penalty_f
                        infos[deliverer].setdefault('failed_delivery', []).append(i)
                    o.status = 0
                    o.accepted_by = None

        # Restock & costs
        for agent_id in range(self.M):
            if self.driver_locs[agent_id] == self.depot:
                self.capacities[agent_id] = self.C_max
                infos[agent_id]['restocked_at_depot'] = True
            op_cost = self.m_time + self.m_dist * move_dists[agent_id]
            rewards[agent_id] -= op_cost
            self.avg_cost[agent_id] += op_cost
            self.dist_travel[agent_id] += move_dists[agent_id]

        # Update immediate rewards for next step
        for agent_id in range(self.M):
            self.immediate_rewards[agent_id] = rewards[agent_id]

        # Advance time and new arrivals
        self.t += 1
        for o in self.orders:
            if o and o.status != 0:
                o.elapsed += 1
        if self.rng.random() < self.p_arrival:
            slot = self._empty_order_slot_index()
            if slot is not None:
                new_order = self._get_new_order()
                new_order.gen_time = self.t
                self.orders[slot] = new_order
        # Expired unaccepted orders (status 1)
        for i, o in enumerate(self.orders):
            if o and o.status == 1 and o.elapsed > o.tw:
                shared_penalty = self.penalty_f / self.M
                for aid in range(self.M):
                    rewards[aid] -= shared_penalty
                    self.penalty[aid] += shared_penalty
                    infos[aid].setdefault('shared_penalty_for_expired_unaccepted', []).append(i)
                o.status = 0
        # Expired accepted orders
        for i, o in enumerate(self.orders):
            if o and o.status == 2 and o.elapsed > o.tw:
                penalized_agent = o.accepted_by
                if penalized_agent is not None:
                    rewards[penalized_agent] -= self.penalty_f
                    self.penalty[penalized_agent] += self.penalty_f
                    infos[penalized_agent].setdefault('expired_accepted_order', []).append(i)
                o.status = 0
                o.accepted_by = None

        # Update immediate rewards again to include expiration penalties
        for agent_id in range(self.M):
            self.immediate_rewards[agent_id] = rewards[agent_id]
            self.reward[agent_id]+=self.immediate_rewards[agent_id]
        for i in range(self.M):
            self.total_rewards_per_agent[i] += rewards[i]
            self.last_info[i] = infos[i]
        self.step_count += 1
        done_flag = (self.t >= self.T) or (self.step_count >= 100000)
        dones = {i: done_flag for i in range(self.M)}
        obs_dict = {i: self._get_obs_for_agent(i) for i in range(self.M)}

        return obs_dict, rewards, dones, infos

