from __future__ import annotations

from collections import defaultdict

from simulator import CrowdSimulator, SocialForceModel
from scenario_types import Scenario, AgentSpec


def scenario_to_simulator(s: Scenario) -> CrowdSimulator:
    """
    Convert a validated Scenario into a configured CrowdSimulator.

    Note: In the provided simulator, `reset()` still calls `set_circle_crossing()`,
    which randomizes human positions. For Week 1, we primarily guarantee:
      - correct number of humans,
      - robot start/goal consistent with scenario,
      - obstacles and groups registered.

    Later, you can modify/reset behavior to use full JSON-specified geometry.
    """
    # Count humans
    humans = [a for a in s.agents if a.role == "human"]
    n_humans = len(humans)

    # Create simulator
    sim = CrowdSimulator(
        time_step=s.sim.dt,
        max_steps=s.sim.max_steps,
        n_humans=n_humans,
        use_full_state=False,
    )

    # Find the robot (assume first robot is the one we control)
    robots = [a for a in s.agents if a.role == "robot"]
    if not robots:
        raise ValueError("scenario_to_simulator: no robot agent found.")
    robot_spec: AgentSpec = robots[0]

    # Add robot
    sim.add_robot(
        x=robot_spec.start[0],
        y=robot_spec.start[1],
        gx=robot_spec.goal[0],
        gy=robot_spec.goal[1],
        radius=robot_spec.radius,
        v_pref=robot_spec.v_pref,
    )

    # Add humans
    for h in humans:
        sim.add_human(
            x=h.start[0],
            y=h.start[1],
            gx=h.goal[0],
            gy=h.goal[1],
            radius=h.radius,
            v_pref=h.v_pref,
        )

    # Add obstacles
    for (p1, p2) in s.map.obstacles:
        sim.add_obstacle(p1, p2)

    # Add groups if group_id is present
    groups = defaultdict(list)
    for a in s.agents:
        if a.group_id is not None:
            groups[a.group_id].append(a.id)
    for g_ids in groups.values():
        if g_ids:
            sim.add_group(g_ids)

    # Attach simple SocialForceModel policies
    robot_policy = SocialForceModel(v0=1.0, max_speed=1.0, time_step=s.sim.dt)
    sim.set_robot_policy(robot_policy)

    for human in sim.humans:
        human_policy = SocialForceModel(v0=1.0, max_speed=1.0, time_step=s.sim.dt)
        sim.set_human_policy(human.id, human_policy)

    return sim
