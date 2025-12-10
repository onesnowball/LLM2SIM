
# This module provides:
# 1. Quality analysis (read-only, generates scores/warnings)
# 2. Geometric repair (modifies scenario to fix critical issues)
# 3. Shared utility functions (BFS, distance calculations, etc.)


from __future__ import annotations
import math
import random
import numpy as np
from typing import List, Tuple, Dict, Any
from collections import deque
from dataclasses import dataclass

from scenario_types import Scenario, AgentSpec

def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, value))


def _segment_point_distance(p1, p2, p):
    """Distance from point p to line segment p1->p2, and the projection point."""
    x1, y1 = p1
    x2, y2 = p2
    x0, y0 = p

    dx = x2 - x1
    dy = y2 - y1
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 1e-12:
        return math.hypot(x0 - x1, y0 - y1), (x1, y1)

    t = ((x0 - x1) * dx + (y0 - y1) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(x0 - proj_x, y0 - proj_y), (proj_x, proj_y)


def _check_path_exists_bfs(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    obstacles: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    clearance: float,
    debug: bool = False
) -> bool:
    """
    BFS-based reachability check using pixel-based grid.
    Returns True if a valid path exists from start to goal.
    """
    cell_size = 0.15

    all_x = [start[0], goal[0]]
    all_y = [start[1], goal[1]]
    for p1, p2 in obstacles:
        all_x.extend([p1[0], p2[0]])
        all_y.extend([p1[1], p2[1]])
    
    margin = 3.0
    xmin, xmax = min(all_x) - margin, max(all_x) + margin
    ymin, ymax = min(all_y) - margin, max(all_y) + margin
    
    grid_width = int(math.ceil((xmax - xmin) / cell_size))
    grid_height = int(math.ceil((ymax - ymin) / cell_size))

    if grid_width > 400 or grid_height > 400:
        if debug:
            print(f"Grid too large ({grid_width}x{grid_height}), using straight-line")
        return _check_straight_line_path(start, goal, obstacles, clearance)
    
    def point_to_cell(x: float, y: float) -> Tuple[int, int]:
        ix = int((x - xmin) / cell_size)
        iy = int((y - ymin) / cell_size)
        return (max(0, min(grid_width - 1, ix)), max(0, min(grid_height - 1, iy)))
    
    def cell_to_point(ix: int, iy: int) -> Tuple[float, float]:
        return (xmin + (ix + 0.5) * cell_size, ymin + (iy + 0.5) * cell_size)

    grid = [[False for _ in range(grid_width)] for _ in range(grid_height)]
    
    for iy in range(grid_height):
        for ix in range(grid_width):
            cx, cy = cell_to_point(ix, iy)
            for p1, p2 in obstacles:
                dist, _ = _segment_point_distance(p1, p2, (cx, cy))
                if dist < clearance:
                    grid[iy][ix] = True
                    break
    
    start_cell = point_to_cell(start[0], start[1])
    goal_cell = point_to_cell(goal[0], goal[1])
    
    if grid[start_cell[1]][start_cell[0]] or grid[goal_cell[1]][goal_cell[0]]:
        return False

    visited = [[False for _ in range(grid_width)] for _ in range(grid_height)]
    queue = deque([start_cell])
    visited[start_cell[1]][start_cell[0]] = True
    
    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    while queue:
        curr_x, curr_y = queue.popleft()
        
        if (curr_x, curr_y) == goal_cell:
            return True
        
        for dx, dy in directions:
            next_x, next_y = curr_x + dx, curr_y + dy
            
            if next_x < 0 or next_x >= grid_width or next_y < 0 or next_y >= grid_height:
                continue
            
            if visited[next_y][next_x] or grid[next_y][next_x]:
                continue
            
            # Prevent corner cutting
            if dx != 0 and dy != 0:
                if grid[curr_y][next_x] or grid[next_y][curr_x]:
                    continue
            
            visited[next_y][next_x] = True
            queue.append((next_x, next_y))
    
    return False


def _check_straight_line_path(
    start: Tuple[float, float],
    goal: Tuple[float, float],
    obstacles: List[Tuple[Tuple[float, float], Tuple[float, float]]],
    clearance: float
) -> bool:
    """Fallback: simple straight-line check."""
    num_samples = 50
    for i in range(num_samples + 1):
        t = i / num_samples
        x = start[0] * (1 - t) + goal[0] * t
        y = start[1] * (1 - t) + goal[1] * t
        
        for p1, p2 in obstacles:
            dist, _ = _segment_point_distance(p1, p2, (x, y))
            if dist < clearance:
                return False
    return True

@dataclass
class QualityAnalysis:
    """Results of quality analysis."""
    overall_score: float
    scores: Dict[str, float]
    warnings: List[str]
    info: Dict[str, Any]
    
    def __str__(self) -> str:
        lines = [
            f"Overall Quality Score: {self.overall_score:.1f}/100",
            "",
            "Component Scores:"
        ]
        for key, score in self.scores.items():
            lines.append(f"  - {key.replace('_', ' ').title()}: {score:.1f}/100")
        
        if self.warnings:
            lines.append(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings[:10]:
                lines.append(f"  ⚠ {warning}")
            if len(self.warnings) > 10:
                lines.append(f"  ... and {len(self.warnings) - 10} more")
        else:
            lines.append("\n✓ No warnings - scenario looks good!")
        
        return "\n".join(lines)

def analyze_quality(scenario: Scenario, fast_mode: bool = False) -> QualityAnalysis:
    """
    Analyze scenario quality WITHOUT modifying it.
    Returns scores, warnings, and diagnostic information.
    
    Args:
        scenario: The scenario to analyze
        fast_mode: If True, skip expensive BFS reachability checks
    """
    scores = {}
    warnings = []
    info = {}
    
    obstacles = scenario.map.obstacles
    agents = scenario.agents
    bounds = scenario.map.bounds
    
    score, warns = _analyze_agent_placement(agents, obstacles, bounds)
    scores['agent_placement'] = score
    warnings.extend(warns)

    score, warns = _analyze_obstacles(obstacles, bounds)
    scores['obstacle_quality'] = score
    warnings.extend(warns)

    if fast_mode:
        scores['reachability'] = 100.0
        info['reachability_note'] = 'Skipped in fast mode'
    else:
        score, warns = _analyze_reachability(agents, obstacles)
        scores['reachability'] = score
        warnings.extend(warns)
    
    score, dist_info = _analyze_spatial_distribution(agents, bounds)
    scores['spatial_distribution'] = score
    info['spatial_distribution'] = dist_info

    if fast_mode:
        weights = {
            'agent_placement': 0.4, 
            'obstacle_quality': 0.3, 
            'reachability': 0.0,    
            'spatial_distribution': 0.3
        }
    else:
        weights = {
            'agent_placement': 0.3,
            'obstacle_quality': 0.2,
            'reachability': 0.3,
            'spatial_distribution': 0.2
        }
    
    overall = sum(scores[k] * weights[k] for k in weights)
    
    return QualityAnalysis(
        overall_score=overall,
        scores=scores,
        warnings=warnings,
        info=info
    )

def print_analysis(scenario: Scenario, fast_mode: bool = False):
    """Print quality analysis in human-readable format."""
    analysis = analyze_quality(scenario, fast_mode=fast_mode)
    print(str(analysis))
    print()
    
    if fast_mode:
        print("⚡ FAST MODE: BFS reachability checks were skipped")
        print()
    
    if 'spatial_distribution' in analysis.info:
        sd = analysis.info['spatial_distribution']
        print("Spatial Distribution:")
        print(f"  - Agent centroid: ({sd['centroid'][0]:.2f}, {sd['centroid'][1]:.2f})")
        print(f"  - Spread ratio: {sd['spread_ratio']:.2f}")
        print(f"  - {sd['note']}")

def _analyze_agent_placement(agents, obstacles, bounds) -> Tuple[float, List[str]]:
    """Check if agents are well-placed."""
    score = 100.0
    warnings = []
    xmin, ymin, xmax, ymax = bounds
    
    for agent in agents:
        sx, sy = agent.start
        gx, gy = agent.goal
        margin = agent.radius + 0.3
        
        # Bounds check
        if not (xmin + margin <= sx <= xmax - margin and ymin + margin <= sy <= ymax - margin):
            score -= 10
            warnings.append(f"Agent {agent.id} start too close to bounds")
        
        if not (xmin + margin <= gx <= xmax - margin and ymin + margin <= gy <= ymax - margin):
            score -= 10
            warnings.append(f"Agent {agent.id} goal too close to bounds")
        
        # Wall clearance
        min_dist_start = float('inf')
        min_dist_goal = float('inf')
        
        for p1, p2 in obstacles:
            dist_s, _ = _segment_point_distance(p1, p2, agent.start)
            dist_g, _ = _segment_point_distance(p1, p2, agent.goal)
            min_dist_start = min(min_dist_start, dist_s)
            min_dist_goal = min(min_dist_goal, dist_g)
        
        required = agent.radius + 0.3
        if min_dist_start < required:
            score -= 15
            warnings.append(
                f"Agent {agent.id} start too close to wall "
                f"({min_dist_start:.2f}m, need {required:.2f}m)"
            )
        
        if min_dist_goal < required:
            score -= 15
            warnings.append(
                f"Agent {agent.id} goal too close to wall "
                f"({min_dist_goal:.2f}m, need {required:.2f}m)"
            )
        
        # Path distance
        dist = math.hypot(gx - sx, gy - sy)
        if dist < 2.0:
            score -= 5
            warnings.append(f"Agent {agent.id} has short path ({dist:.2f}m)")
    
    # Overlapping starts
    for i, a1 in enumerate(agents):
        for a2 in agents[i+1:]:
            dist = math.hypot(a1.start[0] - a2.start[0], a1.start[1] - a2.start[1])
            min_sep = (a1.radius + a2.radius) * 2
            if dist < min_sep:
                score -= 10
                warnings.append(f"Agents {a1.id} and {a2.id} have overlapping starts")
    
    return max(0, score), warnings


def _analyze_obstacles(obstacles, bounds) -> Tuple[float, List[str]]:
    """Check obstacle structure."""
    score = 100.0
    warnings = []
    
    if not obstacles:
        return score, warnings

    for i, (p1, p2) in enumerate(obstacles):
        length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if length < 0.01:
            score -= 5
            warnings.append(f"Obstacle {i} is degenerate (length {length:.3f}m)")

    endpoint_counts = {}
    for p1, p2 in obstacles:
        for pt in [p1, p2]:
            key = (round(pt[0], 2), round(pt[1], 2))
            endpoint_counts[key] = endpoint_counts.get(key, 0) + 1
    
    isolated = sum(1 for count in endpoint_counts.values() if count == 1)
    if isolated > 4:
        warnings.append(f"Found {isolated} isolated wall endpoints")
        score -= 5
    
    enclosed_boxes = _detect_enclosed_boxes(obstacles)
    if enclosed_boxes:
        score -= 50  # MAJOR PENALTY
        for box in enclosed_boxes:
            warnings.append(
                f"🚨 FULLY ENCLOSED BOX DETECTED at approx center {box['center']}! "
                f"This region has NO doorway and will TRAP agents inside. "
                f"You MUST add a doorway by removing part of one wall."
            )
    
    return max(0, score), warnings


def _detect_enclosed_boxes(obstacles) -> List[Dict]:
    """
    Detect fully enclosed rectangular regions with NO openings.
    Returns list of detected boxes with their approximate centers.
    """
    if len(obstacles) < 4:
        return [] 
    
    boxes = []

    adjacency = {}
    for p1, p2 in obstacles:
        p1_key = (round(p1[0], 1), round(p1[1], 1))
        p2_key = (round(p2[0], 1), round(p2[1], 1))
        
        if p1_key not in adjacency:
            adjacency[p1_key] = []
        if p2_key not in adjacency:
            adjacency[p2_key] = []
        
        adjacency[p1_key].append(p2_key)
        adjacency[p2_key].append(p1_key)

    visited = set()
    
    for start_point in adjacency.keys():
        if start_point in visited:
            continue

        cycle = _find_rectangular_cycle(start_point, adjacency, max_length=4)
        
        if cycle and len(cycle) == 4:
            xs = [p[0] for p in cycle]
            ys = [p[1] for p in cycle]
            
            unique_xs = len(set(xs))
            unique_ys = len(set(ys))

            if unique_xs == 2 and unique_ys == 2:
                center_x = sum(xs) / 4
                center_y = sum(ys) / 4
                boxes.append({
                    'center': (center_x, center_y),
                    'corners': cycle
                })

                for point in cycle:
                    visited.add(point)
    
    return boxes


def _find_rectangular_cycle(start, adjacency, max_length=4):
    """
    BFS to find a cycle of specific length forming a rectangle.
    Returns list of points forming the cycle, or None if not found.
    """
    from collections import deque
    
    queue = deque([(start, [start], {start})])
    
    while queue:
        current, path, visited = queue.popleft()
        
        if len(path) > max_length:
            continue

        if len(path) == max_length:
            for neighbor in adjacency[current]:
                if neighbor == start:
                    return path
            continue

        for neighbor in adjacency[current]:
            if neighbor not in visited:
                new_path = path + [neighbor]
                new_visited = visited | {neighbor}
                queue.append((neighbor, new_path, new_visited))
    
    return None


def _analyze_reachability(agents, obstacles) -> Tuple[float, List[str]]:
    """Check if agents can reach goals."""
    score = 100.0
    warnings = []
    
    for agent in agents:
        clearance = agent.radius + 0.2
        
        if not _check_path_exists_bfs(agent.start, agent.goal, obstacles, clearance):
            score -= 30
            warnings.append(
                f"Agent {agent.id} CANNOT reach goal - no valid path exists!"
            )
        else:
            has_straight = _check_straight_line_path(
                agent.start, agent.goal, obstacles, clearance
            )
            if not has_straight:
                warnings.append(
                    f"Agent {agent.id} must navigate around obstacles (path exists but not straight)"
                )
    
    return score, warnings


def _analyze_spatial_distribution(agents, bounds) -> Tuple[float, Dict]:
    """Check spatial distribution."""
    score = 100.0
    info = {}
    
    xmin, ymin, xmax, ymax = bounds
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2
    
    starts = np.array([a.start for a in agents])
    centroid = starts.mean(axis=0)
    
    info['centroid'] = tuple(centroid)
    info['bounds_center'] = (center_x, center_y)
    
    max_dist = max(
        math.hypot(s[0] - centroid[0], s[1] - centroid[1])
        for s in starts
    )
    
    bounds_size = math.hypot(xmax - xmin, ymax - ymin)
    spread_ratio = max_dist / (bounds_size / 2)
    info['spread_ratio'] = spread_ratio
    
    if spread_ratio < 0.3:
        score -= 15
        info['note'] = "Agents clustered in small area"
    elif spread_ratio > 0.7:
        info['note'] = "Good spatial distribution"
    else:
        info['note'] = "Moderate spatial distribution"
    
    return score, info

def repair_scenario(scenario: Scenario, aggressive: bool = False) -> Tuple[Scenario, List[str]]:
    """
    Fix critical geometric issues that prevent scenario from running.
    
    Args:
        scenario: The scenario to repair
        aggressive: If True, applies more aggressive fixes (relocates agents more freely)
    
    Returns:
        Tuple of (repaired_scenario, log_messages)
    """
    logs = []

    robots = [a for a in scenario.agents if a.role == "robot"]
    if not robots:
        raise ValueError("Scenario must contain at least one robot agent")
    xmin, ymin, xmax, ymax = scenario.map.bounds
    for a in scenario.agents:
        sx, sy = a.start
        gx, gy = a.goal
        
        sx_c = _clamp(sx, xmin + 0.5, xmax - 0.5)
        sy_c = _clamp(sy, ymin + 0.5, ymax - 0.5)
        gx_c = _clamp(gx, xmin + 0.5, xmax - 0.5)
        gy_c = _clamp(gy, ymin + 0.5, ymax - 0.5)
        
        if (sx, sy) != (sx_c, sy_c):
            logs.append(f"Clamped agent {a.id} start to bounds")
            a.start = (sx_c, sy_c)
        if (gx, gy) != (gx_c, gy_c):
            logs.append(f"Clamped agent {a.id} goal to bounds")
            a.goal = (gx_c, gy_c)

    if aggressive:
        obstacles = scenario.map.obstacles
        for a in scenario.agents:
            clearance = a.radius + 0.3
            
            min_dist_start = float('inf')
            for p1, p2 in obstacles:
                dist, _ = _segment_point_distance(p1, p2, a.start)
                min_dist_start = min(min_dist_start, dist)
            
            if min_dist_start < clearance:
                for p1, p2 in obstacles:
                    dist, proj = _segment_point_distance(p1, p2, a.start)
                    if dist < clearance:
                        dx = a.start[0] - proj[0]
                        dy = a.start[1] - proj[1]
                        norm = math.hypot(dx, dy)
                        if norm > 1e-6:
                            dx, dy = dx / norm, dy / norm
                            nudge = clearance - dist + 0.1
                            new_x = a.start[0] + dx * nudge
                            new_y = a.start[1] + dy * nudge
                            a.start = (
                                _clamp(new_x, xmin + 0.5, xmax - 0.5),
                                _clamp(new_y, ymin + 0.5, ymax - 0.5)
                            )
                            logs.append(f"Nudged agent {a.id} start away from wall")
                            break
    
    max_radius = max((a.radius for a in scenario.agents), default=0.3)
    min_sep = max(1.0, max_radius * 2.5)
    
    for i in range(len(scenario.agents)):
        for j in range(i + 1, len(scenario.agents)):
            ai, aj = scenario.agents[i], scenario.agents[j]
            dist = math.hypot(ai.start[0] - aj.start[0], ai.start[1] - aj.start[1])
            if dist < min_sep:
                angle = random.random() * 2 * math.pi
                delta = (min_sep - dist) / 2 + 0.1
                dx = delta * math.cos(angle)
                dy = delta * math.sin(angle)
                
                ai.start = (
                    _clamp(ai.start[0] + dx, xmin + 0.5, xmax - 0.5),
                    _clamp(ai.start[1] + dy, ymin + 0.5, ymax - 0.5)
                )
                aj.start = (
                    _clamp(aj.start[0] - dx, xmin + 0.5, xmax - 0.5),
                    _clamp(aj.start[1] - dy, ymin + 0.5, ymax - 0.5)
                )
                logs.append(f"Separated overlapping starts of agents {ai.id} and {aj.id}")
    
    if "min_distance" not in scenario.norms:
        scenario.norms["min_distance"] = max(0.6, 2.0 * max_radius)
        logs.append(f"Set default norms.min_distance = {scenario.norms['min_distance']:.2f}")
    
    if "passing_side" not in scenario.norms:
        scenario.norms["passing_side"] = "right"
        logs.append("Set default norms.passing_side = 'right'")
    
    return scenario, logs


def validate_and_repair(scenario: Scenario) -> Tuple[Scenario, List[str]]:
    """
    Backward compatibility wrapper.
    Does minimal repair (non-aggressive).
    """
    return repair_scenario(scenario, aggressive=False)
