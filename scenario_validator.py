from __future__ import annotations

import math
import random
from typing import List, Tuple

from scenario_types import Scenario, AgentSpec


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _segment_point_distance(p1, p2, p):
    """
    Distance from point p to line segment p1->p2, and the projection point.
    """
    x1, y1 = p1
    x2, y2 = p2
    x0, y0 = p

    dx = x2 - x1
    dy = y2 - y1
    seg_len2 = dx * dx + dy * dy
    if seg_len2 <= 1e-12:
        # Degenerate segment: treat as a point
        return math.hypot(x0 - x1, y0 - y1), (x1, y1)

    t = ((x0 - x1) * dx + (y0 - y1) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(x0 - proj_x, y0 - proj_y), (proj_x, proj_y)


def _push_point_off_walls(point, obstacles, min_clear):
    """
    Iteratively push a point away from any walls closer than min_clear.
    """
    x, y = point
    # Limit iterations so we don't get stuck in a loop
    for _ in range(8):
        moved = False
        for p1, p2 in obstacles:
            dist, proj = _segment_point_distance(p1, p2, (x, y))
            if dist < min_clear:
                px, py = proj
                nx, ny = x - px, y - py
                norm = math.hypot(nx, ny)
                if norm < 1e-6:
                    # If we're exactly on the wall, pick a normal perpendicular to the wall
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    nx, ny = -dy, dx
                    norm = math.hypot(nx, ny)
                    if norm < 1e-6:
                        continue
                nx /= norm
                ny /= norm
                move = (min_clear - dist) + 1e-3
                x += nx * move
                y += ny * move
                moved = True
        if not moved:
            break
    return (x, y)


def _enforce_min_gap_widths(s: Scenario, factor: float = 3.0) -> List[str]:
    """
    Ensure that small gaps between colinear wall segments (doors / narrow openings)
    are at least `factor * max_agent_radius` wide.

    Works for axis-aligned walls (horizontal/vertical), which is what the prompts encourage.
    """
    logs: List[str] = []
    obstacles = getattr(s.map, "obstacles", [])
    if not obstacles:
        return logs

    # Estimate max radius to decide how wide a doorway should be
    max_radius = 0.0
    for a in s.agents:
        try:
            max_radius = max(max_radius, float(a.radius))
        except Exception:
            continue
    if max_radius <= 0.0:
        max_radius = 0.3  # sensible default

    min_gap = max(1.0, factor * max_radius)

    # Group horizontal and vertical segments by approximate coordinate
    horiz_groups = {}  # key: y, value: list of dicts with x1,x2,idx,y
    vert_groups = {}   # key: x, value: list of dicts with y1,y2,idx,x

    for idx, (p1, p2) in enumerate(obstacles):
        x1, y1 = p1
        x2, y2 = p2
        if abs(y1 - y2) < 1e-6:
            # horizontal
            y_key = round(0.5 * (y1 + y2), 3)
            x_lo = min(x1, x2)
            x_hi = max(x1, x2)
            horiz_groups.setdefault(y_key, []).append(
                {"idx": idx, "x1": x_lo, "x2": x_hi, "y": y_key, "orig": (p1, p2)}
            )
        elif abs(x1 - x2) < 1e-6:
            # vertical
            x_key = round(0.5 * (x1 + x2), 3)
            y_lo = min(y1, y2)
            y_hi = max(y1, y2)
            vert_groups.setdefault(x_key, []).append(
                {"idx": idx, "y1": y_lo, "y2": y_hi, "x": x_key, "orig": (p1, p2)}
            )

    # Widen gaps between adjacent horizontal segments on the same line
    for y, segs in horiz_groups.items():
        segs.sort(key=lambda d: d["x1"])
        for i in range(len(segs) - 1):
            left = segs[i]
            right = segs[i + 1]
            gap = right["x1"] - left["x2"]
            if gap <= 0 or gap >= min_gap:
                continue

            # widen the opening symmetrically
            extra = (min_gap - gap) * 0.5
            new_left_x2 = left["x2"] - extra
            new_right_x1 = right["x1"] + extra
            # avoid flipping segments
            if new_left_x2 <= left["x1"] or new_right_x1 >= right["x2"]:
                continue

            logs.append(
                f"Widened horizontal opening at y={y:.2f} from {gap:.2f} to {min_gap:.2f}."
            )
            left["x2"] = new_left_x2
            right["x1"] = new_right_x1

        # write back updated segments
        for seg in segs:
            idx = seg["idx"]
            x1 = seg["x1"]
            x2 = seg["x2"]
            y_val = seg["y"]
            p1_orig, p2_orig = obstacles[idx]
            if p1_orig[0] <= p2_orig[0]:
                obstacles[idx] = ((x1, y_val), (x2, y_val))
            else:
                obstacles[idx] = ((x2, y_val), (x1, y_val))

    # Widen gaps between adjacent vertical segments on the same line
    for x, segs in vert_groups.items():
        segs.sort(key=lambda d: d["y1"])
        for i in range(len(segs) - 1):
            bottom = segs[i]
            top = segs[i + 1]
            gap = top["y1"] - bottom["y2"]
            if gap <= 0 or gap >= min_gap:
                continue

            extra = (min_gap - gap) * 0.5
            new_bottom_y2 = bottom["y2"] - extra
            new_top_y1 = top["y1"] + extra
            if new_bottom_y2 <= bottom["y1"] or new_top_y1 >= top["y2"]:
                continue

            logs.append(
                f"Widened vertical opening at x={x:.2f} from {gap:.2f} to {min_gap:.2f}."
            )
            bottom["y2"] = new_bottom_y2
            top["y1"] = new_top_y1

        for seg in segs:
            idx = seg["idx"]
            y1 = seg["y1"]
            y2 = seg["y2"]
            x_val = seg["x"]
            p1_orig, p2_orig = obstacles[idx]
            if p1_orig[1] <= p2_orig[1]:
                obstacles[idx] = ((x_val, y1), (x_val, y2))
            else:
                obstacles[idx] = ((x_val, y2), (x_val, y1))

    return logs


def _repair_agent_positions(s: Scenario) -> List[str]:
    """
    Push agent start/goal away from walls and clamp them inside bounds,
    so they are not sitting on walls or exactly in door edges.
    """
    logs: List[str] = []
    obstacles = getattr(s.map, "obstacles", [])
    if not obstacles:
        return logs

    # Estimate max radius again
    max_radius = 0.0
    for a in s.agents:
        try:
            max_radius = max(max_radius, float(a.radius))
        except Exception:
            continue
    if max_radius <= 0.0:
        max_radius = 0.3

    # clearance from walls / entrances
    base_clear = max_radius * 1.5

    xmin, ymin, xmax, ymax = s.map.bounds

    for a in s.agents:
        margin = float(getattr(a, "radius", max_radius)) + 0.1

        sx, sy = a.start
        gx, gy = a.goal

        # clamp to bounds first
        sx = _clamp(sx, xmin + margin, xmax - margin)
        sy = _clamp(sy, ymin + margin, ymax - margin)
        gx = _clamp(gx, xmin + margin, xmax - margin)
        gy = _clamp(gy, ymin + margin, ymax - margin)

        start_before = (sx, sy)
        goal_before = (gx, gy)

        sx, sy = _push_point_off_walls((sx, sy), obstacles, base_clear)
        gx, gy = _push_point_off_walls((gx, gy), obstacles, base_clear)

        if (sx, sy) != start_before:
            logs.append(f"Moved agent {a.id} start away from nearby walls/entrances.")
        if (gx, gy) != goal_before:
            logs.append(f"Moved agent {a.id} goal away from nearby walls/entrances.")

        a.start = (sx, sy)
        a.goal = (gx, gy)

    return logs

def _snap_to_corridor_band(s: Scenario, tol: float = 0.2) -> List[str]:
    """If the map type suggests a corridor, keep agents inside the corridor band.

    We detect two dominant parallel wall chains (either vertical or horizontal)
    and clamp the cross-corridor coordinate of every agent start/goal so they
    lie between those walls with a small margin.
    """
    logs: List[str] = []
    obstacles = getattr(s.map, "obstacles", [])
    if not obstacles:
        return logs

    # Estimate max radius for a reasonable margin from walls
    max_radius = 0.0
    for a in s.agents:
        try:
            max_radius = max(max_radius, float(a.radius))
        except Exception:
            continue
    if max_radius <= 0.0:
        max_radius = 0.3
    margin = max_radius + 0.05

    def _cluster_segments_vertical():
        groups = {}  # key -> dict(x, length)
        for p1, p2 in obstacles:
            x1, y1 = p1
            x2, y2 = p2
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            # Vertical: x nearly constant, non-degenerate in y
            if dx > tol or dy < tol:
                continue
            x_mid = 0.5 * (x1 + x2)
            key = round(x_mid / tol) * tol
            length = abs(y2 - y1)
            if key not in groups:
                groups[key] = {"x": x_mid, "length": 0.0}
            groups[key]["length"] += length
        return groups

    def _cluster_segments_horizontal():
        groups = {}  # key -> dict(y, length)
        for p1, p2 in obstacles:
            x1, y1 = p1
            x2, y2 = p2
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            # Horizontal: y nearly constant, non-degenerate in x
            if dy > tol or dx < tol:
                continue
            y_mid = 0.5 * (y1 + y2)
            key = round(y_mid / tol) * tol
            length = abs(x2 - x1)
            if key not in groups:
                groups[key] = {"y": y_mid, "length": 0.0}
            groups[key]["length"] += length
        return groups

    def _pick_two_strongest(groups):
        if len(groups) < 2:
            return None
        items = sorted(groups.items(), key=lambda kv: kv[1]["length"], reverse=True)
        (k1, g1), (k2, g2) = items[0], items[1]
        # Require some separation so we don't pick the same wall twice
        if abs(k1 - k2) < tol:
            return None
        return (g1, g2)

    vert_groups = _cluster_segments_vertical()
    horiz_groups = _cluster_segments_horizontal()

    vert_pair = _pick_two_strongest(vert_groups)
    horiz_pair = _pick_two_strongest(horiz_groups)

    corridor_orientation = None
    if vert_pair and horiz_pair:
        vert_len = vert_pair[0]["length"] + vert_pair[1]["length"]
        horiz_len = horiz_pair[0]["length"] + horiz_pair[1]["length"]
        corridor_orientation = "vertical" if vert_len >= horiz_len else "horizontal"
    elif vert_pair:
        corridor_orientation = "vertical"
    elif horiz_pair:
        corridor_orientation = "horizontal"

    if corridor_orientation is None:
        # Nothing that looks like a corridor
        return logs

    if corridor_orientation == "vertical":
        g1, g2 = vert_pair
        x_left = min(g1["x"], g2["x"])
        x_right = max(g1["x"], g2["x"])
        band_min = x_left + margin
        band_max = x_right - margin
        if band_min >= band_max:
            return logs

        for a in s.agents:
            sx, sy = a.start
            gx, gy = a.goal
            new_sx = _clamp(sx, band_min, band_max)
            new_gx = _clamp(gx, band_min, band_max)
            if new_sx != sx:
                logs.append(
                    f"Snapped agent {a.id} start.x from {sx:.2f} to {new_sx:.2f} "
                    f"inside corridor band [{x_left:.2f}, {x_right:.2f}]."
                )
            if new_gx != gx:
                logs.append(
                    f"Snapped agent {a.id} goal.x from {gx:.2f} to {new_gx:.2f} "
                    f"inside corridor band [{x_left:.2f}, {x_right:.2f}]."
                )
            a.start = (new_sx, sy)
            a.goal = (new_gx, gy)
    else:
        g1, g2 = horiz_pair
        y_low = min(g1["y"], g2["y"])
        y_high = max(g1["y"], g2["y"])
        band_min = y_low + margin
        band_max = y_high - margin
        if band_min >= band_max:
            return logs

        for a in s.agents:
            sx, sy = a.start
            gx, gy = a.goal
            new_sy = _clamp(sy, band_min, band_max)
            new_gy = _clamp(gy, band_min, band_max)
            if new_sy != sy:
                logs.append(
                    f"Snapped agent {a.id} start.y from {sy:.2f} to {new_sy:.2f} "
                    f"inside corridor band [{y_low:.2f}, {y_high:.2f}]."
                )
            if new_gy != gy:
                logs.append(
                    f"Snapped agent {a.id} goal.y from {gy:.2f} to {new_gy:.2f} "
                    f"inside corridor band [{y_low:.2f}, {y_high:.2f}]."
                )
            a.start = (sx, new_sy)
            a.goal = (gx, new_gy)

    return logs


def _repair_reachability_via_grid(s: Scenario, max_cells: int = 80) -> List[str]:
    """Approximate free space by an occupancy grid and ensure each agent's
    start and goal lie in the same connected component.

    - Cells within (radius + small_margin) of any wall segment are marked blocked.
    - Starts/goals in blocked cells are snapped to the nearest free cell.
    - If a goal lies in a different component than the start, it is moved to the
      nearest free cell in the start's component.
    """
    logs: List[str] = []
    obstacles = getattr(s.map, "obstacles", [])
    if not obstacles:
        return logs

    xmin, ymin, xmax, ymax = s.map.bounds
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0 or height <= 0:
        return logs

    # Estimate maximum radius and choose grid resolution adaptively
    max_radius = 0.0
    for a in s.agents:
        try:
            max_radius = max(max_radius, float(a.radius))
        except Exception:
            continue
    if max_radius <= 0.0:
        max_radius = 0.3

    # Keep the grid reasonably sized: longest side ~ max_cells cells
    longest = max(width, height)
    cell_size = longest / max_cells
    cell_size = max(0.1, min(0.5, cell_size))  # clamp to [0.1, 0.5] meters

    nx = max(4, int(math.ceil(width / cell_size)))
    ny = max(4, int(math.ceil(height / cell_size)))

    # Inflate walls by radius + a bit
    inflate = max_radius + 0.05

    # Precompute occupancy grid: True = blocked, False = free
    grid = [[False for _ in range(nx)] for _ in range(ny)]
    for j in range(ny):
        cy = ymin + (j + 0.5) * cell_size
        for i in range(nx):
            cx = xmin + (i + 0.5) * cell_size
            min_dist = float("inf")
            for p1, p2 in obstacles:
                dist, _ = _segment_point_distance(p1, p2, (cx, cy))
                if dist < min_dist:
                    min_dist = dist
                if min_dist <= inflate:
                    break
            if min_dist <= inflate:
                grid[j][i] = True  # blocked

    def _point_to_cell(x: float, y: float) -> Tuple[int, int]:
        ix = int((x - xmin) / cell_size)
        iy = int((y - ymin) / cell_size)
        ix = max(0, min(nx - 1, ix))
        iy = max(0, min(ny - 1, iy))
        return ix, iy

    def _cell_to_point(ix: int, iy: int) -> Tuple[float, float]:
        cx = xmin + (ix + 0.5) * cell_size
        cy = ymin + (iy + 0.5) * cell_size
        return cx, cy

    from collections import deque

    def _nearest_free_cell(ix: int, iy: int) -> Tuple[int, int]:
        """Find nearest free cell using BFS in grid space."""
        ix = max(0, min(nx - 1, ix))
        iy = max(0, min(ny - 1, iy))
        if not grid[iy][ix]:
            return ix, iy
        visited = [[False for _ in range(nx)] for _ in range(ny)]
        q = deque()
        q.append((ix, iy))
        visited[iy][ix] = True
        while q:
            x0, y0 = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x1, y1 = x0 + dx, y0 + dy
                if x1 < 0 or x1 >= nx or y1 < 0 or y1 >= ny:
                    continue
                if visited[y1][x1]:
                    continue
                visited[y1][x1] = True
                if not grid[y1][x1]:
                    return x1, y1
                q.append((x1, y1))
        # Fallback: just return the clamped original cell
        return ix, iy

    def _component_from(start_ix: int, start_iy: int):
        visited = [[False for _ in range(nx)] for _ in range(ny)]
        comp = set()
        if grid[start_iy][start_ix]:
            return comp  # blocked cell => empty component
        q = deque()
        q.append((start_ix, start_iy))
        visited[start_iy][start_ix] = True
        comp.add((start_ix, start_iy))
        while q:
            x0, y0 = q.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x1, y1 = x0 + dx, y0 + dy
                if x1 < 0 or x1 >= nx or y1 < 0 or y1 >= ny:
                    continue
                if visited[y1][x1] or grid[y1][x1]:
                    continue
                visited[y1][x1] = True
                comp.add((x1, y1))
                q.append((x1, y1))
        return comp

    # 1) Ensure every start/goal lies in a free cell (snap to nearest if needed)
    for a in s.agents:
        sx, sy = a.start
        gx, gy = a.goal
        six, siy = _point_to_cell(sx, sy)
        gix, giy = _point_to_cell(gx, gy)

        new_six, new_siy = _nearest_free_cell(six, siy)
        new_gix, new_giy = _nearest_free_cell(gix, giy)

        if (new_six, new_siy) != (six, siy):
            new_sx, new_sy = _cell_to_point(new_six, new_siy)
            logs.append(
                f"Moved agent {a.id} start from ({sx:.2f}, {sy:.2f}) "
                f"to nearest free cell ({new_sx:.2f}, {new_sy:.2f})."
            )
            a.start = (new_sx, new_sy)
            sx, sy = new_sx, new_sy
            six, siy = new_six, new_siy

        if (new_gix, new_giy) != (gix, giy):
            new_gx, new_gy = _cell_to_point(new_gix, new_giy)
            logs.append(
                f"Moved agent {a.id} goal from ({gx:.2f}, {gy:.2f}) "
                f"to nearest free cell ({new_gx:.2f}, {new_gy:.2f})."
            )
            a.goal = (new_gx, new_gy)
            gx, gy = new_gx, new_gy
            gix, giy = new_gix, new_giy

    # 2) For each agent, ensure goal is in the same connected component as start
    for a in s.agents:
        sx, sy = a.start
        gx, gy = a.goal
        six, siy = _point_to_cell(sx, sy)
        gix, giy = _point_to_cell(gx, gy)

        comp = _component_from(six, siy)
        if not comp:
            # Start is in a blocked cell or isolated one; we already tried to move it above.
            continue

        if (gix, giy) not in comp:
            # Pick the free cell in the component closest to the original goal
            best_cell = None
            best_dist = float("inf")
            for (cx, cy) in comp:
                wx, wy = _cell_to_point(cx, cy)
                d = math.hypot(wx - gx, wy - gy)
                if d < best_dist:
                    best_dist = d
                    best_cell = (cx, cy)
            if best_cell is not None:
                bx, by = best_cell
                new_gx, new_gy = _cell_to_point(bx, by)
                logs.append(
                    f"Adjusted agent {a.id} goal to reachable point "
                    f"({new_gx:.2f}, {new_gy:.2f}) inside free-space component of its start."
                )
                a.goal = (new_gx, new_gy)

    return logs


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

    # === NEW: geometric repairs for doors/corridors and agent positions ===
    # 1) Corridor-specific snapping (for corridor-style maps)
    map_type = getattr(s.map, "type", "").lower()
    if "corridor" in map_type:
        logs.extend(_snap_to_corridor_band(s))

    # 2) Ensure doors are wide enough and starts/goals are not on walls
    logs.extend(_enforce_min_gap_widths(s))
    logs.extend(_repair_agent_positions(s))

    # 3) Enforce reachability in the free space defined by walls
    logs.extend(_repair_reachability_via_grid(s))

    return s, logs


