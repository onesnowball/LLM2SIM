from __future__ import annotations

from collections import defaultdict

from simulator import CrowdSimulator, SocialForceModel
from scenario_types import Scenario, AgentSpec


def scenario_to_simulator(s: Scenario) -> CrowdSimulator:

    humans = [a for a in s.agents if a.role == "human"]
    n_humans = len(humans)

    sim = CrowdSimulator(
        time_step=s.sim.dt,
        max_steps=s.sim.max_steps,
        n_humans=n_humans,
        use_full_state=False,
    )
    sim.randomize_humans = False
    robots = [a for a in s.agents if a.role == "robot"]
    if not robots:
        raise ValueError("scenario_to_simulator: no robot agent found.")
    robot_spec: AgentSpec = robots[0]

    sim.add_robot(
        x=robot_spec.start[0],
        y=robot_spec.start[1],
        gx=robot_spec.goal[0],
        gy=robot_spec.goal[1],
        radius=robot_spec.radius,
        v_pref=robot_spec.v_pref,
    )

    for h in humans:
        sim.add_human(
            x=h.start[0],
            y=h.start[1],
            gx=h.goal[0],
            gy=h.goal[1],
            radius=h.radius,
            v_pref=h.v_pref,
        )

    for (p1, p2) in s.map.obstacles:
        sim.add_obstacle(p1, p2)

    groups = defaultdict(list)
    for a in s.agents:
        if a.group_id is not None:
            groups[a.group_id].append(a.id)
    for g_ids in groups.values():
        if g_ids:
            sim.add_group(g_ids)

    robot_policy = SocialForceModel(v0=1.0, max_speed=1.0, time_step=s.sim.dt)
    sim.set_robot_policy(robot_policy)

    for human in sim.humans:
        human_policy = SocialForceModel(v0=1.0, max_speed=1.0, time_step=s.sim.dt)
        sim.set_human_policy(human.id, human_policy)

    return sim
