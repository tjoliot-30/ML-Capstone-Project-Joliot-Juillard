"""Finite Gymnasium inventory MDP aligned with CM-B2."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ORDER_QUANTITIES: tuple[int, ...] = (0, 1, 2)
LOW_DEMAND = 0
HIGH_DEMAND = 1


@dataclass(frozen=True)
class InventoryConfig:
    max_stock: int = 8
    horizon: int = 40
    order_cost: float = 0.45
    holding_cost: float = 0.30
    unmet_demand_cost: float = 3.0
    regime_transition: tuple[tuple[float, float], tuple[float, float]] = ((0.85, 0.15), (0.20, 0.80))


class InventoryControlEnv(gym.Env):
    """Single-item continuing task observed as `(stock_level, demand_regime)`.

    Actions index physical order quantities `(0, 1, 2)`. Replenishment is
    immediate, demand is conditional on a persistent low/high Markov regime,
    and the fixed horizon produces Gymnasium truncation, never termination.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: InventoryConfig | None = None):
        super().__init__()
        self.config = config or InventoryConfig()
        self.observation_space = spaces.MultiDiscrete([self.config.max_stock + 1, 2])
        self.action_space = spaces.Discrete(len(ORDER_QUANTITIES))
        self._stock = 0
        self._regime = LOW_DEMAND
        self._step = 0

    def _observation(self) -> np.ndarray:
        return np.asarray([self._stock, self._regime], dtype=np.int64)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._stock = int(self.np_random.integers(0, self.config.max_stock + 1))
        self._regime = int(self.np_random.integers(0, 2))
        self._step = 0
        return self._observation(), {"time_limit": self.config.horizon}

    def _sample_demand(self) -> int:
        if self._regime == LOW_DEMAND:
            return int(self.np_random.choice([0, 1, 2], p=[0.25, 0.60, 0.15]))
        return int(self.np_random.choice([1, 2, 3], p=[0.15, 0.55, 0.30]))

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action index: {action}")
        c = self.config
        order_quantity = ORDER_QUANTITIES[int(action)]
        available = min(c.max_stock, self._stock + order_quantity)
        demand = self._sample_demand()
        served = min(available, demand)
        unmet = demand - served
        self._stock = available - served
        cost = c.order_cost * order_quantity + c.holding_cost * self._stock + c.unmet_demand_cost * unmet
        current_regime = self._regime
        self._regime = int(self.np_random.choice([LOW_DEMAND, HIGH_DEMAND], p=c.regime_transition[current_regime]))
        self._step += 1
        terminated = False
        truncated = self._step >= c.horizon
        info = {
            "cost": float(cost),
            "order_quantity": order_quantity,
            "demand": demand,
            "served": served,
            "unmet_demand": unmet,
            "fill_rate": 1.0 if demand == 0 else float(served / demand),
            "stock_level": self._stock,
            "demand_regime": self._regime,
        }
        return self._observation(), -float(cost), terminated, truncated, info


def reorder_action(observation: np.ndarray, low_target: int = 2, high_target: int = 4) -> int:
    """Transparent regime-aware reorder heuristic returning an action index."""

    stock_level, demand_regime = (int(value) for value in observation)
    target = high_target if demand_regime == HIGH_DEMAND else low_target
    quantity = int(np.clip(target - stock_level, ORDER_QUANTITIES[0], ORDER_QUANTITIES[-1]))
    return ORDER_QUANTITIES.index(quantity)

