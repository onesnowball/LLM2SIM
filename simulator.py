#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from typing import List, Dict, Tuple, Optional
from IPython.display import HTML
import time
import logging
from matplotlib.animation import PillowWriter
import os


class Agent: pass
class MultiHumanTracker: pass

import gymnasium as gym
from gymnasium import spaces
import torch as th

class stateutils:
    @staticmethod
    def normalize(vectors: np.ndarray):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms_safe = np.where(norms == 0, 1, norms)
        return vectors / norms_safe, norms.flatten()

class Agent:
    """
    Base class for agents (humans and robots) in the simulation.
    """
    def __init__(self, agent_id: int, x: float, y: float, gx: float, gy: float,
                 vx: float = 0.0, vy: float = 0.0, radius: float = 0.3,
                 v_pref: float = 1.0, agent_type: str = 'human'):
        self.id = agent_id
        self.px, self.py = x, y
        self.gx, self.gy = gx, gy
        self.vx, self.vy = vx, vy
        self.radius = radius
        self.v_pref = v_pref
        self.agent_type = agent_type
        self.position_history = [(x, y)]
        self.velocity_history = [(vx, vy)]
        self.preferred_velocity = np.array([0.0, 0.0])

        self.init_px, self.init_py = x, y
        self.init_gx, self.init_gy = gx, gy
        
    def get_position(self) -> Tuple[float, float]:
        return np.array([self.px, self.py])

    def get_velocity(self) -> Tuple[float, float]:
        return np.array([self.vx, self.vy])

    def get_goal(self) -> Tuple[float, float]:
        return np.array([self.gx, self.gy])

    def get_full_state(self) -> np.ndarray:
        return np.array([self.px, self.py, self.vx, self.vy, self.gx, self.gy, self.radius])

    def set_position(self, x: float, y: float):
        self.px, self.py = x, y
        self.position_history.append((x, y))

    def set_velocity(self, vx: float, vy: float):
        self.vx, self.vy = vx, vy
        self.velocity_history.append((vx, vy))

    def get_distance_to_goal(self) -> float:
        return np.sqrt((self.px - self.gx)**2 + (self.py - self.gy)**2)

    def is_at_goal(self, threshold: float = 0.05) -> bool:
        """
        Check whether the distance to goal is below a predefined threshold.
        Return True if so, otherwise False.
        """
        return self.get_distance_to_goal() < threshold
    
    def get_preferred_velocity(self) -> np.ndarray:
        """
        Returns the agent's preferred velocity (w), required by the Social Force Model.
        """
        return self.preferred_velocity

    def set_preferred_velocity(self, w: np.ndarray):
        """
        Updates the agent's preferred velocity (w).
        """
        self.preferred_velocity = w

    
class Obstacle:
    """Represents a line segment obstacle."""
    def __init__(self, p1, p2):
        self.p1 = np.array(p1, dtype=float)
        self.p2 = np.array(p2, dtype=float)

class GoalPolicy:
    """
    A simple policy that directs an agent towards its goal at a constant speed,
    ignoring other agents and obstacles.
    """
    def __init__(self, max_speed=1.0):
        self.max_speed = max_speed

    def predict(self, agent, other_agents, obstacles=[]):
        """
        Calculates the velocity vector to move directly to the goal.
        """
        direction_to_goal = agent.get_goal() - agent.get_position()
        distance = np.linalg.norm(direction_to_goal)

        if distance < 1e-6:
            return (0.0, 0.0)

        normalized_direction = direction_to_goal / distance
        desired_velocity = self.max_speed * normalized_direction
        return (desired_velocity[0], desired_velocity[1])
    
class RLPolicy:
    def __init__(self, model, max_speed=1.0, n_humans=2, state_dim=5):
        self.max_speed = 1.0
        self.model = model
        self.n_humans = n_humans
        self.state_dim = state_dim

    def predict(self, obs):
        obs = obs[:int(self.state_dim + self.state_dim * self.n_humans)]
        action, _ = self.model.predict(obs)
        return action
    
class ILPolicy:
    def __init__(self, model, max_speed=1.0, n_humans=2, state_dim=5):
        self.max_speed = 1.0
        self.model = model
        self.n_humans = n_humans
        self.state_dim = state_dim

    def predict(self, obs):
        obs = obs[:int(self.state_dim + self.state_dim * self.n_humans)]
        action = self.model(th.tensor(obs)).cpu().detach().numpy()
        return action

class SocialForceModel:
    """
    Implementation of the Social Force Model.
    """
    def __init__(self, v0: float = 1.0, tau: float = 0.5, A: float = 800.0, B: float = 0.2,
                 max_speed: float = 1.0, radius: float = 0.3,
                 interaction_range: float = 3.0, time_step: float = 0.25,
                 delta_t: float = 0.2, field_of_view_angle_deg: float = 220.0,
                 c_perception: float = 0.5, invisible_robot: bool = False):
        self.v0 = v0                   
        self.tau = tau                 
        self.A = A                    
        self.B = B                     
        self.radius = radius          
        self.max_speed = max_speed
        self.interaction_range = interaction_range
        self.time_step = time_step

        self.delta_t = delta_t        
        self.fov_cos_angle = np.cos(np.deg2rad(field_of_view_angle_deg) / 2.0)
        self.c_perception = c_perception 
        self.last_predicted_w = np.array([0.0, 0.0])
        self.invisible_robot = invisible_robot

    def _compute_desired_acceleration(self, position, goal, current_velocity, desired_speed):
        direction_to_goal = np.array(goal) - np.array(position)
        distance = np.linalg.norm(direction_to_goal)

        if distance < 1e-6:
            return np.array([0.0, 0.0])

        e_i = direction_to_goal / distance
        desired_velocity = desired_speed * e_i
        desired_accleration = (desired_velocity - np.array(current_velocity)) / self.tau
        return desired_accleration
    
    def _compute_social_acceleration(self, agent_pos, agent_desired_dir, other_agents: List['Agent']):
        social_acceleration = np.array([0.0, 0.0])
        for beta in other_agents:
            if self.invisible_robot and beta.agent_type == 'robot':
                continue
            pos_beta = beta.get_position()
            if np.allclose(agent_pos, pos_beta):
                continue
            
            r_alpha_beta = agent_pos - pos_beta
            d_alpha_beta = np.linalg.norm(r_alpha_beta)
            if d_alpha_beta > self.interaction_range:
                continue

            vel_beta = beta.get_velocity()
            v_beta_norm = np.linalg.norm(vel_beta)
            e_beta = vel_beta / v_beta_norm if v_beta_norm > 1e-4 else np.array([0., 0.])
            
            r_prime = r_alpha_beta - v_beta_norm * self.delta_t * e_beta 
            b = 0.5 * np.sqrt(max(0, (d_alpha_beta + np.linalg.norm(r_prime))**2 - (v_beta_norm * self.delta_t)**2))

            force_mag = self.A * np.exp(-b / self.B)
            n_alpha_beta = r_alpha_beta / d_alpha_beta
            f_alpha_beta = force_mag * n_alpha_beta

            cos_phi = np.dot(agent_desired_dir, -n_alpha_beta)
            if cos_phi < self.fov_cos_angle:
                f_alpha_beta *= self.c_perception

            social_acceleration += f_alpha_beta
            
        return social_acceleration
    
    def _compute_obstacle_acceleration(self, position, agent_desired_dir, obstacles: List['Obstacle']):
        """
        Computes repulsive 'forces' from obstacles.
        """
        obstacle_acceleration = np.array([0.0, 0.0])
        for wall in obstacles:
            p1, p2 = wall.p1, wall.p2
            line_vec = p2 - p1
            line_len_sq = np.dot(line_vec, line_vec)
            
            if line_len_sq < 1e-6: continue
            
            t = np.dot(position - p1, line_vec) / line_len_sq
            t = np.clip(t, 0, 1)
            closest_point = p1 + t * line_vec

            d_iB_vec = position - closest_point
            d_iB = np.linalg.norm(d_iB_vec)

            if d_iB > self.interaction_range or d_iB < 1e-6:
                continue
            
            n_iB = d_iB_vec / d_iB
            force_mag = self.A * np.exp(-d_iB / self.B)
            f_iB = force_mag * n_iB
            if np.dot(agent_desired_dir, -n_iB) < self.fov_cos_angle:
                f_iB *= self.c_perception

            obstacle_acceleration += f_iB
        return obstacle_acceleration
    
    def _compute_group_coherence_force(self, agent: Agent, all_agents: List[Agent], groups: List[List[int]]):
        total_force = np.array([0.0, 0.0])
        return total_force * 3.0  
    
    def _compute_group_repulsive_force(self, agent: Agent, all_agents: List[Agent], groups: List[List[int]], threshold=0.55):

        total_force = np.array([0.0, 0.0])
        return total_force * 2
    
    def get_last_predicted_preferred_velocity(self) -> np.ndarray:
        return self.last_predicted_w

    def predict(self, agent: 'Agent', other_agents: List['Agent'], obstacles: List['Obstacle'] = [], groups: List[List[int]] = []) -> Tuple[float, float]:
        position = agent.get_position()
        goal = agent.get_goal()
        actual_velocity = agent.get_velocity()
        preferred_velocity = agent.get_preferred_velocity()
        desired_speed = self.v0

        direction_to_goal = goal - position
        dist_to_goal = np.linalg.norm(direction_to_goal)
        agent_desired_dir = direction_to_goal / dist_to_goal if dist_to_goal > 1e-6 else np.array([0.,0.])

        desired_acceleration = self._compute_desired_acceleration(position, goal, actual_velocity, desired_speed)
        social_acceleration = self._compute_social_acceleration(position, agent_desired_dir, other_agents)
        obstacle_acceleration = self._compute_obstacle_acceleration(position, agent_desired_dir, obstacles)

        group_coherence = self._compute_group_coherence_force(agent, other_agents, groups)
        group_repulsive = self._compute_group_repulsive_force(agent, other_agents, groups)
        
        total_acceleration = desired_acceleration + social_acceleration + obstacle_acceleration + group_repulsive +  group_coherence 
        noise_magnitude = 0.01 
        noise = noise_magnitude * self.A * (np.random.rand(2) - 0.5)
        total_acceleration+= noise

        new_preferred_velocity = preferred_velocity + total_acceleration * self.time_step
        self.last_predicted_w = new_preferred_velocity

        speed_w = np.linalg.norm(new_preferred_velocity)
        if speed_w > self.max_speed:
            new_actual_velocity = (new_preferred_velocity / speed_w) * self.max_speed
        else:
            new_actual_velocity = new_preferred_velocity

        return new_actual_velocity[0], new_actual_velocity[1]

class CrowdSimulator(gym.Env):
    """
    Simulates the physical movement of agents in a crowd.
    """
    def __init__(self, time_step: float = 0.25, max_steps: int = 200, n_humans=0, use_full_state=False):
        super().__init__()
        self.randomize_humans = True
        self.time_step = time_step
        self.max_steps = max_steps
        self.current_step = 0
        self.robot: Optional[Agent] = None
        self.obstacles: List[Obstacle] = []
        self.humans: List[Agent] = []
        self.n_humans = n_humans
        self.groups: List[List[int]] = []
        self.all_agents: List[Agent] = []
        self.robot_policy: Optional[SocialForceModel] = None
        self.human_policies: Dict[int, SocialForceModel] = {}
        self.done = False
        self.info = {}
        self.status = 'InProgress'
        self.use_full_state = use_full_state

        self.reward_function = self.social_reward
        self.aps = 0
        self.Rps = 0
        self.aprog = 0
        self.rg = 0
        self.col = 0

        # TODO: Task 1.1.1.
        # Initialize a two-dimensional Box space which takes values in [-1, 1] for the action space.
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # TODO: Task 1.1.2.
        # Calculate the dimension for the observation space based on the number of humans and the number of features for the human and the robot
        # Initialize a Box space which takes values from -inf to inf of the appropriate dimension for the observation space.
        self.observation_space = spaces.Box(low=-np.inf,high=np.inf,shape=(self.n_humans * 5 + 5,),dtype=np.float32)

    def add_robot(self, *args, **kwargs) -> Agent:
        self.robot = Agent(agent_id=0, agent_type='robot', *args, **kwargs)
        self.all_agents.append(self.robot)
        return self.robot

    def add_human(self, *args, **kwargs) -> Agent:
        human_id = len(self.humans) + 1
        human = Agent(agent_id=human_id, agent_type='human', *args, **kwargs)
        self.humans.append(human)
        self.all_agents.append(human)
        return human
    
    def add_obstacle(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> Obstacle:
        obstacle = Obstacle(p1, p2)
        self.obstacles.append(obstacle)
        return obstacle

    def set_robot_policy(self, policy: SocialForceModel):
        self.robot_policy = policy
        self.robot_policy.time_step = self.time_step

    def set_human_policy(self, human_id: int, policy: SocialForceModel):
        self.human_policies[human_id] = policy
        self.human_policies[human_id].time_step = self.time_step
        
    def add_group(self, agent_ids: List[int]):
        """Register a group by list of agent IDs."""
        self.groups.append(agent_ids)

    def has_group(self):
        return len(self.groups) > 0
    
    def set_reward_params(self, aps, Rps, aprog, rg, col):
        """
        Sets constants which should be used in the social reward function.
        """
        self.aps = aps
        self.Rps = Rps
        self.aprog = aprog
        self.rg = rg
        self.col = col
    
    def social_reward(self, state: np.ndarray, action: np.ndarray):
        """
        Reward function used to train a social robot navigation policy.
        Returns reward based on goal reaching, collision, and personal space violation.
        """
        reward = 0.0

        # TODO: Task 1.2.1.
        # If the robot has reached the goal (use the predefined function) provide a large reward.
        # Compute the distance between the robot and the goal and the distance between the robot state after executing the action and the goal.
        # Add a reward proportional to the increase/decrease in distance.
        # Use self.rg and self.aprog as the constants for goal reaching and progress, respectively.
        robot = self.robot
        if robot.is_at_goal():
            reward += self.rg
        if self._check_collisions():
            reward += self.col
            
        p_t = np.array([robot.px, robot.py])
        g = np.array([robot.gx, robot.gy])
        p_next = p_t + action * self.time_step
        
        reward += self.aprog * (np.linalg.norm(p_t - g) - np.linalg.norm(p_next - g))

       
        if self.humans:
            d_min = min(np.linalg.norm(p_t - np.array([h.px, h.py])) for h in self.humans)
            if 0.0 < d_min < self.Rps:
                reward += self.aps * (d_min - self.Rps) 
            
        # TODO: Task 1.2.2.
        # If the robot is in collision (use the predefined function) provide a large negative reward (self.col)
        # Calculate the minimum distance between the robot and all agents
        # If the distance is below a threshold (self.Rps) apply a penalty proportional to the personal space violation
        # Use self.col and self.aps as the constants for collision and personal space violation, respectively.
        


        return reward
    
    def _get_obs(self):
        """
        Creates an array with the observable parts of the joint state for use by the robot policy network.
        Returns a np.ndarray of shape (5 + 5 * n_humans,) with the relevant robot and human features.
        """
        # TODO: Task 1.1.3.
        # Initialize an array of the correct size.
        # Populate the array with the robot features in the first five positions, followed by the features for each human in each successive five positions.
        # Ensure all features are relative to the robot (positions and goals relative to the robot position, velocities relative to the robot velocity).
        robot = self.robot
        humans = self.humans
        o_R = np.array([robot.gx-robot.px, robot.gy-robot.py, robot.vx, robot.vy, robot.radius])
        o_H = []
        for human in humans:
            o = np.array([human.px-robot.px, human.py-robot.py, human.vx-robot.vx, human.vy-robot.vy, human.radius])
            o_H.append(o)
        obs = np.concatenate([o_R] + o_H)

        return obs
    
    def step(self, action):
        """
        Advances the simulation by one step.
        """
        if self.done:
            return None, 0, self.done, self.info

        next_velocities = {}
        next_preferred_velocities = {}
        for agent in self.all_agents:
            if agent.is_at_goal():
                next_velocities[agent.id] = (0.0, 0.0)
                continue
            if agent.agent_type == 'robot':
                if action is not None:
                    next_velocities[agent.id] = action
                elif self.robot_policy:
                    if isinstance(self.robot_policy, SocialForceModel):
                        vx, vy = self.robot_policy.predict(agent, self.all_agents, obstacles=self.obstacles)
                        next_velocities[agent.id] = (vx, vy)
                        new_w = self.robot_policy.get_last_predicted_preferred_velocity()
                        next_preferred_velocities[agent.id] = new_w
                    elif isinstance(self.robot_policy, RLPolicy) or isinstance(self.robot_policy, ILPolicy):
                        next_velocities[agent.id] = self.robot_policy.predict(self._get_obs())
                    else:
                        next_velocities[agent.id] = self.robot_policy.predict(agent, self.all_agents, obstacles=self.obstacles)
            
            elif agent.agent_type == 'human':
                if agent.id in self.human_policies:
                    policy = self.human_policies[agent.id]
                    if isinstance(policy, SocialForceModel):
                        vx, vy = policy.predict(agent, self.all_agents, obstacles=self.obstacles, groups=self.groups)
                        next_velocities[agent.id] = (vx, vy)
                        new_w = policy.get_last_predicted_preferred_velocity()
                        next_preferred_velocities[agent.id] = new_w
                    else:
                        next_velocities[agent.id] = policy.predict(agent, self.all_agents, obstacles=self.obstacles)

        for agent in self.all_agents:
            if agent.id in next_velocities:
                vx, vy = next_velocities[agent.id]
                
                agent.set_velocity(vx, vy)
                agent.set_position(agent.px + vx * self.time_step, agent.py + vy * self.time_step)
                if agent.id in next_preferred_velocities:
                    agent.set_preferred_velocity(next_preferred_velocities[agent.id])

        self.current_step += 1
        if action is not None:
            reward = self.reward_function(self.robot.get_position(), action)
        else:
            reward = 0
        
        # Check for success
        if self.robot.is_at_goal():
            self.done = True
            self.info['status'] = True
            reward = True
            self.status = 'AllAtGoal'
        # Check for timeout
        elif self.current_step >= self.max_steps:
            self.done = True
            self.info['status'] = True
            self.status = 'Timeout'
        # Check for collisions
        elif self._check_collisions():
            self.done = True
            self.info['status'] = True
            reward = True
            self.status = 'Collision'

        obs = self._get_obs()
            
        return obs, float(reward), bool(self.done and self.status in ('AllAtGoal', 'Collision')), bool(self.done and self.status == 'Timeout'), self.info
    
    def set_circle_crossing(self):

        for a, agent in enumerate(self.all_agents):
            if agent.id == self.robot.id:
                agent.px, agent.py = agent.position_history[0]
                agent.vx, agent.vy = 0.0, 0.0
                agent.position_history = [(agent.px, agent.py)]
                agent.preferred_velocity = np.array([0.0, 0.0])
            else:
                collides = True
                while collides:
                    theta = np.random.uniform(low=0.0, high=2*np.pi)
                    x = np.cos(theta) * 2.5 #hard coded radius
                    y = np.sin(theta) * 2.5
                    c_hyp = False
                    for o, oa in enumerate(self.all_agents):
                        if oa.id == agent.id:
                            continue
                        if np.linalg.norm(np.array([x, y]) - oa.get_position()) < agent.radius + oa.radius:
                            c_hyp = True
                            break
                    if not c_hyp:
                        collides = False
                        agent.px = x
                        agent.py = y
                        agent.gx = -1 * x
                        agent.gy = -1 * y
                        agent.position_history = [(agent.px, agent.py)]
                        agent.preferred_velocity = np.array([0.0, 0.0])
                        break
                

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)
        self.current_step = 0
        self.done = False
        self.info = {}
    
        if self.randomize_humans:
            self.set_circle_crossing()
        else:
            # Restore each agent to its scenario-defined initial state
            for agent in self.all_agents:
                # positions
                agent.px, agent.py = agent.init_px, agent.init_py
                # goals
                agent.gx, agent.gy = agent.init_gx, agent.init_gy
                # velocities reset
                agent.vx, agent.vy = 0.0, 0.0
                agent.position_history = [(agent.px, agent.py)]
                agent.velocity_history = [(0.0, 0.0)]
                agent.preferred_velocity = np.array([0.0, 0.0])
    
        obs = self._get_obs()
        return obs, self.info


    def _check_collisions(self) -> bool:
        """
        Checks if the robot is in collision with any humans.
        Return True if in collision, False otherwise.
        """
        if not self.robot: return False
        for human in self.humans:
            dist = np.sqrt((self.robot.px - human.px)**2 + (self.robot.py - human.py)**2)
            if dist < (self.robot.radius + human.radius):
                return True
        return False

    def visualize_simulation(self, tracker: Optional[MultiHumanTracker] = None, output_file: Optional[str] = None, show_plot: bool = True):
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
        ax.set_aspect('equal')
        ax.set_title("Crowd Simulator")
        ax.grid(True, linestyle='--', alpha=0.6)
 

        for wall in self.obstacles:
            ax.plot([wall.p1[0], wall.p2[0]], [wall.p1[1], wall.p2[1]], color='black', linewidth=3, zorder=2)

        human_colors = plt.cm.viridis(np.linspace(0, 1, len(self.humans)))
        if self.robot:
            ax.plot(self.robot.gx, self.robot.gy, 'k*', markersize=15, label="Robot Goal")
        for i, human in enumerate(self.humans):
            ax.plot(human.gx, human.gy, '*', color=human_colors[i], markersize=15)

        robot_artist = Circle((0,0), self.robot.radius, color='black', zorder=10, label="Robot")
        ax.add_patch(robot_artist)
        human_artists = [Circle((0,0), h.radius, color=c, zorder=8, label=f"Human {h.id}") for i, (h, c) in enumerate(zip(self.humans, human_colors))]
        for artist in human_artists: ax.add_patch(artist)

        est_artists, particle_artists = [], []
        if tracker:
            est_artists = [ax.plot([], [], 'x', color='red', markersize=8, zorder=9, label="Estimate")[0] for _ in self.humans]
            particle_artists = [ax.plot([], [], '.', color=c, markersize=1, alpha=0.5, zorder=5)[0] for c in human_colors]

        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
        def init():
            # place everything at frame 0
            robot_artist.center = self.robot.position_history[0]
            for i, human in enumerate(self.humans):
                human_artists[i].center = human.position_history[0]
            for a in est_artists:
                a.set_data([], [])
            for p in particle_artists:
                p.set_data([], [])
            time_text.set_text('Time: 0.00s')
            return [robot_artist] + human_artists + est_artists + particle_artists + [time_text]
        def update(frame):
            robot_artist.center = self.robot.position_history[frame]
            for i, human in enumerate(self.humans):
                human_artists[i].center = human.position_history[frame]
                if tracker and frame > 0 and frame <= len(tracker.tracking_history[human.id]['estimated']):
                    est_pos = tracker.tracking_history[human.id]['estimated'][frame-1]
                    est_artists[i].set_data([est_pos[0]], [est_pos[1]])
                    if 'particles' in tracker.tracking_history[human.id] and frame-1 < len(tracker.tracking_history[human.id]['particles']):
                        particles = tracker.tracking_history[human.id]['particles'][frame-1]
                        particle_artists[i].set_data(particles[:, 0], particles[:, 1])

            time_text.set_text(f'Time: {frame * self.time_step:.2f}s')
            return [robot_artist] + human_artists + est_artists + particle_artists + [time_text]

        num_frames = len(self.robot.position_history)
        max_frames = max(len(agent.position_history) for agent in self.all_agents)
        for agent in self.all_agents:
            while len(agent.position_history) < max_frames:
                agent.position_history.append(agent.position_history[-1])
                agent.velocity_history.append(agent.velocity_history[-1])
        num_frames = max_frames
        anim = animation.FuncAnimation(fig, update, frames=num_frames, interval=150, init_func=init, blit=False)
        
        if output_file:
            # Ensure we write a GIF using Pillow
            root, ext = os.path.splitext(output_file)
            if ext.lower() != ".gif":
                output_file = root + ".gif"
        
            print(f"Saving animation to {output_file}...")
            writer = PillowWriter(fps=12)
            anim.save(output_file, writer=writer)
            print("Save complete.")
        
        if show_plot:
            return HTML(anim.to_jshtml())

    def calculate_metrics(self) -> Dict:
        """
        Calculates simulation metrics after a run is complete.
        Includes both per-agent and global measures, as well as
        robot-focused navigation metrics (time-to-goal, collisions, speed, etc.).
        """
        if not self.done:
            logging.warning("Cannot calculate metrics until the simulation is done.")
            return {}

        metrics = {
            'global': {},
            'per_agent': {agent.id: {} for agent in self.all_agents},
            'robot': {}
        }

        # ----------------
        # Robot-focused metrics
        # ----------------
        robot = self.robot
        metrics['robot']['time_to_goal'] = self.current_step
        metrics['robot']['collision'] = (self.info.get('status') == 'Collision')

        positions = np.array(robot.position_history)
        if len(positions) > 1:
            path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            straight_dist = np.linalg.norm(positions[-1] - positions[0])
            metrics['robot']['path_efficiency'] = path_length / (straight_dist + 1e-6)
        else:
            metrics['robot']['path_efficiency'] = float('inf')

        velocities = np.array(robot.velocity_history)
        if len(velocities) > 0:
            speeds = np.linalg.norm(velocities, axis=1)
            metrics['robot']['avg_speed'] = np.mean(speeds)
        else:
            metrics['robot']['avg_speed'] = 0.0

        # Minimum distance to humans
        min_dist = float('inf')
        if self.humans:
            for step in range(len(self.humans[0].position_history)):
                robot_pos = positions[min(step, len(positions)-1)]
                for h in self.humans:
                    human_pos = np.array(h.position_history[min(step, len(h.position_history)-1)])
                    dist = np.linalg.norm(robot_pos - human_pos)
                    min_dist = min(min_dist, dist)
        metrics['robot']['min_human_dist'] = min_dist

        # ----------------
        # Per-agent metrics
        # ----------------
        for agent in self.all_agents:
            # Average acceleration
            accel_magnitudes = []
            for i in range(1, len(agent.velocity_history)):
                v_curr = np.array(agent.velocity_history[i])
                v_prev = np.array(agent.velocity_history[i-1])
                accel = (v_curr - v_prev) / self.time_step
                accel_magnitudes.append(np.linalg.norm(accel))
            avg_accel = np.mean(accel_magnitudes) if accel_magnitudes else 0.0
            metrics['per_agent'][agent.id]['avg_acceleration'] = avg_accel

            # Path efficiency
            path_length = 0.0
            for i in range(1, len(agent.position_history)):
                p_curr = np.array(agent.position_history[i])
                p_prev = np.array(agent.position_history[i-1])
                path_length += np.linalg.norm(p_curr - p_prev)
            start_pos = np.array(agent.position_history[0])
            end_pos = np.array(agent.position_history[-1])
            straight_line_dist = np.linalg.norm(end_pos - start_pos)
            efficiency = path_length / straight_line_dist if straight_line_dist > 1e-6 else float('inf')
            metrics['per_agent'][agent.id]['path_efficiency'] = efficiency

        # ----------------
        # Global metrics
        # ----------------
        min_dist_overall = float('inf')
        num_steps = len(self.all_agents[0].position_history)
        for t in range(num_steps):
            for i in range(len(self.all_agents)):
                for j in range(i + 1, len(self.all_agents)):
                    pos1 = np.array(self.all_agents[i].position_history[t])
                    pos2 = np.array(self.all_agents[j].position_history[t])
                    dist = np.linalg.norm(pos1 - pos2)
                    if dist < min_dist_overall:
                        min_dist_overall = dist
        metrics['global']['min_inter_agent_distance'] = min_dist_overall

        # cache and return
        self.metrics = metrics
        return metrics


def create_test_scenario(scenario_type: str = 'scenario1') -> Tuple[List[Tuple], List[Tuple], List[Tuple]]:
    """
    Returns human start/goal positions and obstacles for some set scenarios.
    """
    if scenario_type == 'scenario1':
        human_starts = [
            (-5.0, -1.0),   
            (5.0,  -1.0),   
            (-2.0,  -4.0)   
        ]
        human_goals  = [
            (-5.0, 1.0),   
            (5.0, 1.0),  
            (2.0,  -4.0)     
        ]
        obstacles = []

    elif scenario_type == 'scenario2':
        human_starts = [
            (-4.0, 0.0), (4.0, 0.0), (0.0, 4.0), (0.0, -4.0),
            (-2.8, 2.8), (2.8, 2.8), (-2.8, -2.8), (2.8, -2.8)
        ]
        human_goals  = [
            (4.0, 0.0), (-4.0, 0.0), (0.0, -4.0), (0.0, 4.0),
            (2.8, -2.8), (-2.8, -2.8), (2.8, 2.8), (-2.8, 2.8)
        ]
        obstacles = []
    elif scenario_type == 'scenario3':
        human_starts = [(-5, 1), (5, -1), (-5, -1), (5, 1), (0, -5), (0, 5)]
        human_goals = [(5, 1), (-5, -1), (5, -1), (-5, 1), (0, 5), (0, -5)]
        obstacles = []
    elif scenario_type == 'scenario4':
        human_starts = [(-2.0, -2.0), (2.0, 2.0)]
        human_goals  = [(2.0, -2.0), (-2.0, 2.0)]

        obstacles = [
            ((-1.0, 0.0), (1.0, 0.0)), 
        ]
    elif scenario_type == 'scenario5':      
        # Humans sweep across the *upper* gap (around y â‰ˆ +0.8)
        human_starts = [(-2.5, -1.80), ( 2.5, 0.80), (-2.5, 0.92)]
        human_goals  = [( 2.5, -1.80), (-2.5, 0.80), ( 2.5, 0.92)]

        # Vertical barrier at x=0 with TWO gaps:
        # - Upper gap: narrow (yâˆˆ[0.60, 1.00]) â€” but crowded by humans
        # - Lower gap: wider (yâˆˆ[-1.20, -0.60]) â€” clear but longer route
        obstacles = [
            ((2.0,  2.00), (4.0,  2.00)),    # middle solid (blocks the center)
        ]
    else:
        human_starts = [(-5.0, 1.5), (5.0, -1.5), (0.0, 5.0)]
        human_goals  = [(5.0, 1.5), (-5.0, -1.5), (0.0, -5.0)]
        obstacles = []

    return human_starts, human_goals, obstacles