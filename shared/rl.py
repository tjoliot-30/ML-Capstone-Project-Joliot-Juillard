"""Public RL evaluation primitives for TDo2.

Students implement the assessed Q-learning loop in their notebook.  Keeping
training out of this public module makes that implementation observable while
retaining one common environment, target definition, baselines and evaluator.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from .inventory_env import ORDER_QUANTITIES, InventoryControlEnv, reorder_action

Policy = Callable[[np.ndarray, np.random.Generator], int]


def q_learning_target(reward: float, next_best_value: float, gamma: float, *, terminated: bool) -> float:
    """Bootstrap across time-limit truncation; zero continuation only at termination."""

    return float(reward) if terminated else float(reward + gamma * next_best_value)


def q_table_policy(q_table: np.ndarray) -> Policy:
    return lambda observation, rng: int(np.argmax(q_table[tuple(int(value) for value in observation)]))


def random_policy(observation: np.ndarray, rng: np.random.Generator) -> int:
    del observation
    return int(rng.integers(0, len(ORDER_QUANTITIES)))


def reorder_policy(observation: np.ndarray, rng: np.random.Generator) -> int:
    del rng
    return reorder_action(observation)


base_stock_policy = reorder_policy


def evaluate_policies(policies: dict[str, Policy], *, seeds: Iterable[int]) -> pd.DataFrame:
    """Evaluate all policies on the exact same episodes and report cost/service."""

    rows: list[dict[str, float | int | str]] = []
    for seed in tuple(int(seed) for seed in seeds):
        for name, policy in policies.items():
            env = InventoryControlEnv()
            observation, _ = env.reset(seed=seed)
            action_rng = np.random.default_rng(seed + 900_001)
            done = False
            total_cost = 0.0
            fill_rates: list[float] = []
            unmet_demand = 0
            stock_levels: list[float] = []
            order_quantities: list[int] = []
            while not done:
                action = policy(observation, action_rng)
                observation, _, terminated, truncated, info = env.step(action)
                total_cost += float(info["cost"])
                fill_rates.append(float(info["fill_rate"]))
                unmet_demand += int(info["unmet_demand"])
                stock_levels.append(float(info["stock_level"]))
                order_quantities.append(int(info["order_quantity"]))
                done = terminated or truncated
            rows.append(
                {
                    "seed": seed,
                    "policy": name,
                    "episode_cost": total_cost,
                    "mean_fill_rate": float(np.mean(fill_rates)),
                    "unmet_demand": unmet_demand,
                    "mean_stock": float(np.mean(stock_levels)),
                    "mean_order_quantity": float(np.mean(order_quantities)),
                }
            )
            env.close()
    return pd.DataFrame(rows)
