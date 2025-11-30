from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from scenario_types import Scenario, MapSpec, AgentSpec, SimSpec


def load_scenario_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_map(map_dict: Dict[str, Any]) -> MapSpec:
    map_type = map_dict["type"]
    bounds_list = map_dict["bounds"]
    if len(bounds_list) != 4:
        raise ValueError("map.bounds must have length 4: [xmin, ymin, xmax, ymax]")
    bounds = (float(bounds_list[0]), float(bounds_list[1]),
              float(bounds_list[2]), float(bounds_list[3]))

    obstacles = []
    for obs in map_dict.get("obstacles", []):
        # Handle both {"p1": [...], "p2": [...]} and [[...], [...]] formats
        if isinstance(obs, dict):
            p1 = obs["p1"]
            p2 = obs["p2"]
        else:
            p1, p2 = obs[0], obs[1]
        p1_t = (float(p1[0]), float(p1[1]))
        p2_t = (float(p2[0]), float(p2[1]))
        obstacles.append((p1_t, p2_t))

    return MapSpec(type=map_type, bounds=bounds, obstacles=obstacles)


def _parse_agent(agent_dict: Dict[str, Any]) -> AgentSpec:
    start = agent_dict["start"]
    goal = agent_dict["goal"]
    # Handle both {"x": ..., "y": ...} and [x, y] formats
    if isinstance(start, dict):
        start_xy = (float(start["x"]), float(start["y"]))
    else:
        start_xy = (float(start[0]), float(start[1]))
    if isinstance(goal, dict):
        goal_xy = (float(goal["x"]), float(goal["y"]))
    else:
        goal_xy = (float(goal[0]), float(goal[1]))

    return AgentSpec(
        id=int(agent_dict["id"]),
        role=str(agent_dict["role"]),
        start=start_xy,
        goal=goal_xy,
        radius=float(agent_dict.get("radius", 0.3)),
        v_pref=float(agent_dict.get("v_pref", 1.0)),
        behavior=str(agent_dict.get("behavior", "social_force")),
        group_id=agent_dict.get("group_id", None),
    )


def _parse_sim(sim_dict: Dict[str, Any]) -> SimSpec:
    return SimSpec(
        dt=float(sim_dict.get("dt", 0.25)),
        max_steps=int(sim_dict.get("max_steps", 200)),
    )


def dict_to_scenario(data: Dict[str, Any]) -> Scenario:
    metadata = dict(data["metadata"])
    map_spec = _parse_map(data["map"])
    agents = [_parse_agent(a) for a in data["agents"]]
    norms = dict(data.get("norms", {}))
    sim_spec = _parse_sim(data["sim"])
    events = list(data.get("events", []))

    return Scenario(
        metadata=metadata,
        map=map_spec,
        agents=agents,
        norms=norms,
        sim=sim_spec,
        events=events,
    )


def load_scenario(path: str | Path) -> Scenario:
    raw = load_scenario_json(path)
    return dict_to_scenario(raw)
