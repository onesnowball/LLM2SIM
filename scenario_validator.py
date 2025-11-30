from __future__ import annotations

import math
import random
from typing import List, Tuple

from scenario_types import Scenario, AgentSpec


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def validate_and_repair(s: Scenario) -> Tuple[Scenario, List[str]]:
    """
    Basic structural + geometric validation and light auto-repair.

    Repairs:
    - Clamp all agent start/goal positions into map bounds.
    - Jitter overlapping start positions apart.
    - Fill missing norms.passing_side and norms.min_distance.
    """
    logs: List[str] = []

    # 1. Ensure at least one robot
    robots = [a for a in s.agents if a.role == "robot"]
    if not robots:
        raise ValueError("Scenario must contain at least one agent with role == 'robot'.")

    # 2. Clamp start/goal points into map bounds
    xmin, ymin, xmax, ymax = s.map.bounds
    for a in s.agents:
        sx, sy = a.start
        gx, gy = a.goal

        sx_c = _clamp(sx, xmin, xmax)
        sy_c = _clamp(sy, ymin, ymax)
        gx_c = _clamp(gx, xmin, xmax)
        gy_c = _clamp(gy, ymin, ymax)

        if (sx, sy) != (sx_c, sy_c):
            logs.append(f"Clamped start of agent {a.id} from ({sx:.2f},{sy:.2f}) to ({sx_c:.2f},{sy_c:.2f}).")
        if (gx, gy) != (gx_c, gy_c):
            logs.append(f"Clamped goal of agent {a.id} from ({gx:.2f},{gy:.2f}) to ({gx_c:.2f},{gy_c:.2f}).")

        a.start = (sx_c, sy_c)
        a.goal = (gx_c, gy_c)

    # 3. Resolve overlapping starts via simple jitter
    if s.agents:
        max_radius = max(a.radius for a in s.agents)
    else:
        max_radius = 0.3
    # base minimum separation
    min_sep_target = max(0.01, max_radius * 2.1)

    def too_close(a1: AgentSpec, a2: AgentSpec, sep: float) -> bool:
        dx = a1.start[0] - a2.start[0]
        dy = a1.start[1] - a2.start[1]
        return math.hypot(dx, dy) < sep

    changed = True
    loop_count = 0
    while changed and loop_count < 10:
        changed = False
        loop_count += 1
        for i in range(len(s.agents)):
            for j in range(i + 1, len(s.agents)):
                ai = s.agents[i]
                aj = s.agents[j]
                if too_close(ai, aj, min_sep_target):
                    # Jitter both agents in opposite random directions
                    angle = random.random() * 2.0 * math.pi
                    delta = (min_sep_target) / 2.0 + 1e-3
                    dx = delta * math.cos(angle)
                    dy = delta * math.sin(angle)

                    ai.start = (ai.start[0] + dx, ai.start[1] + dy)
                    aj.start = (aj.start[0] - dx, aj.start[1] - dy)
                    logs.append(f"Jittered overlapping agents {ai.id} and {aj.id}.")
                    changed = True

                    # Clamp again to bounds after jitter
                    ai.start = (_clamp(ai.start[0], xmin, xmax),
                                _clamp(ai.start[1], ymin, ymax))
                    aj.start = (_clamp(aj.start[0], xmin, xmax),
                                _clamp(aj.start[1], ymin, ymax))

    # 4. Norm defaults and sanity
    if "min_distance" not in s.norms:
        s.norms["min_distance"] = 0.6
        logs.append("Set default norms.min_distance = 0.6.")
    if "passing_side" not in s.norms:
        s.norms["passing_side"] = "right"
        logs.append("Set default norms.passing_side = 'right'.")

    # Ensure min_distance is at least 2 * max_radius (roughly) but not huge
    min_dist = float(s.norms.get("min_distance", 0.6))
    if max_radius > 0:
        desired_min = max(0.4, 2.0 * max_radius)
        if min_dist < desired_min:
            logs.append(
                f"Increasing norms.min_distance from {min_dist:.2f} to {desired_min:.2f} "
                f"to be compatible with agent radii."
            )
            s.norms["min_distance"] = desired_min

    return s, logs
