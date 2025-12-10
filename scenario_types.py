from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional


Float2 = Tuple[float, float]
Float4 = Tuple[float, float, float, float]


@dataclass
class AgentSpec:
    id: int
    role: str  # "robot" or "human"

    start: Float2
    goal: Float2

    radius: float = 0.3
    v_pref: float = 1.0

    behavior: str = "social_force"
    group_id: Optional[int] = None


@dataclass
class MapSpec:
    type: str          
    bounds: Float4 

    obstacles: List[Tuple[Float2, Float2]] = field(default_factory=list)


@dataclass
class SimSpec:
    dt: float = 0.25
    max_steps: int = 200


@dataclass
class Scenario:
    metadata: Dict[str, Any]
    map: MapSpec
    agents: List[AgentSpec]
    norms: Dict[str, Any]
    sim: SimSpec
    events: List[Dict[str, Any]] = field(default_factory=list)
