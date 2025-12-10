#!/usr/bin/env python3
"""
Unified scenario generator with automatic iterative refinement.
FIXED: Proper model selection and improved prompting
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import uuid
import os
import numpy as np
import json

from scenario_generator import generate_scenario, save_json
from scenario_io import load_scenario, dict_to_scenario
from scenario_validation import repair_scenario, analyze_quality, print_analysis
from scenario_adapter import scenario_to_simulator

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError(
        "OPENAI_API_KEY environment variable must be set. "
        "Export it in your shell: export OPENAI_API_KEY='your-key-here'"
    )

DEFAULTS = {
    "provider": "openai",
    "model": "gpt-4.1",  
    "temperature": 0.0,
    "max_tokens": 16000, 
    "no_repair": False,
    "map_type": "auto",
    "seed": 42,
    "output_dir": "scenarios/generated",
}


def visualize_scenario_with_path(sim, scenario):
    """
    Enhanced visualization showing:
    - Initial positions and goals
    - BFS pathfinding routes (if reachable)
    - No simulation required
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        import matplotlib.cm as cm
    except ImportError:
        print("⚠️  matplotlib not available, skipping visualization")
        return
    
    from scenario_validation import _check_path_exists_bfs
    
    fig, ax = plt.subplots(figsize=(14, 12))
    

    all_x = []
    all_y = []
    
    if sim.obstacles:
        for wall in sim.obstacles:
            all_x.extend([wall.p1[0], wall.p2[0]])
            all_y.extend([wall.p1[1], wall.p2[1]])
    
    if sim.robot:
        all_x.extend([sim.robot.px, sim.robot.gx])
        all_y.extend([sim.robot.py, sim.robot.gy])
    
    for human in sim.humans:
        all_x.extend([human.px, human.gx])
        all_y.extend([human.py, human.gy])
    
    if all_x and all_y:
        margin = 2.0
        xmin, xmax = min(all_x) - margin, max(all_x) + margin
        ymin, ymax = min(all_y) - margin, max(all_y) + margin
        
        x_range = xmax - xmin
        y_range = ymax - ymin
        if x_range < 10:
            center_x = (xmin + xmax) / 2
            xmin, xmax = center_x - 5, center_x + 5
        if y_range < 10:
            center_y = (ymin + ymax) / 2
            ymin, ymax = center_y - 5, center_y + 5
        
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_xlim(-7, 7)
        ax.set_ylim(-7, 7)
    
    ax.set_aspect('equal')
    ax.set_title("Scenario with BFS Paths", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('X (meters)', fontsize=11)
    ax.set_ylabel('Y (meters)', fontsize=11)

    obstacle_drawn = False
    for wall in sim.obstacles:
        ax.plot(
            [wall.p1[0], wall.p2[0]], 
            [wall.p1[1], wall.p2[1]], 
            color='black', 
            linewidth=4, 
            zorder=2, 
            label='_nolegend_',
            solid_capstyle='round'
        )
        obstacle_drawn = True
    
    if obstacle_drawn:
        ax.plot([], [], color='black', linewidth=4, label='Walls', solid_capstyle='round')

    obstacles = scenario.map.obstacles

    if sim.robot:
        robot_circle = Circle(
            (sim.robot.px, sim.robot.py), 
            sim.robot.radius,
            facecolor='red',
            edgecolor='darkred',
            linewidth=2,
            zorder=10, 
            label='Robot',
            alpha=0.8
        )
        ax.add_patch(robot_circle)
        
        ax.plot(
            sim.robot.gx, sim.robot.gy, 
            marker='*', 
            color='red', 
            markersize=25, 
            label='Robot Goal', 
            zorder=11,
            markeredgecolor='darkred',
            markeredgewidth=1.5
        )
        
        clearance = sim.robot.radius + 0.2
        path = _find_bfs_path(
            (sim.robot.px, sim.robot.py),
            (sim.robot.gx, sim.robot.gy),
            obstacles,
            clearance
        )
        
        if path:
            path_array = np.array(path)
            ax.plot(
                path_array[:, 0], path_array[:, 1],
                color='red',
                linewidth=2,
                alpha=0.5,
                linestyle='-',
                label='Robot BFS Path',
                zorder=6
            )
    
    if sim.humans:
        colors = cm.tab10(np.linspace(0, 1, max(10, len(sim.humans))))
        
        for i, human in enumerate(sim.humans):
            color = colors[i % len(colors)]
            
            human_circle = Circle(
                (human.px, human.py), 
                human.radius,
                facecolor=color,
                edgecolor='black',
                linewidth=1.5,
                zorder=8, 
                label=f'Human {human.id}',
                alpha=0.7
            )
            ax.add_patch(human_circle)
            
            ax.plot(
                human.gx, human.gy, 
                marker='*', 
                color=color, 
                markersize=20, 
                label='_nolegend_', 
                zorder=9,
                markeredgecolor='black',
                markeredgewidth=1
            )
            
            clearance = human.radius + 0.2
            path = _find_bfs_path(
                (human.px, human.py),
                (human.gx, human.gy),
                obstacles,
                clearance
            )
            
            if path:
                path_array = np.array(path)
                ax.plot(
                    path_array[:, 0], path_array[:, 1],
                    color=color,
                    linewidth=1.5,
                    alpha=0.4,
                    linestyle='-',
                    label='_nolegend_',
                    zorder=6
                )
    
    ax.legend(
        loc='upper right', 
        fontsize=9,
        framealpha=0.9,
        edgecolor='gray'
    )
    
    plt.tight_layout()
    plt.show()
    print("\n✓ Visualization displayed with BFS pathfinding routes")


def _find_bfs_path(start, goal, obstacles, clearance):
    """
    Run BFS to find actual path from start to goal.
    Returns list of (x, y) waypoints or None if no path.
    """
    import math
    from collections import deque
    
    def segment_point_distance(p1, p2, p):
        x1, y1 = p1
        x2, y2 = p2
        x0, y0 = p
        dx = x2 - x1
        dy = y2 - y1
        seg_len2 = dx * dx + dy * dy
        if seg_len2 <= 1e-12:
            return math.hypot(x0 - x1, y0 - y1)
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / seg_len2
        t = max(0.0, min(1.0, t))
        proj_x = x1 + t * dx
        proj_y = y1 + t * dy
        return math.hypot(x0 - proj_x, y0 - proj_y)
    
    cell_size = 0.2
    
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
        return None
    
    def point_to_cell(x, y):
        ix = int((x - xmin) / cell_size)
        iy = int((y - ymin) / cell_size)
        return (max(0, min(grid_width - 1, ix)), max(0, min(grid_height - 1, iy)))
    
    def cell_to_point(ix, iy):
        return (xmin + (ix + 0.5) * cell_size, ymin + (iy + 0.5) * cell_size)
    
    # Build grid
    grid = [[False for _ in range(grid_width)] for _ in range(grid_height)]
    for iy in range(grid_height):
        for ix in range(grid_width):
            cx, cy = cell_to_point(ix, iy)
            for p1, p2 in obstacles:
                dist = segment_point_distance(p1, p2, (cx, cy))
                if dist < clearance:
                    grid[iy][ix] = True
                    break
    
    start_cell = point_to_cell(start[0], start[1])
    goal_cell = point_to_cell(goal[0], goal[1])
    
    if grid[start_cell[1]][start_cell[0]] or grid[goal_cell[1]][goal_cell[0]]:
        return None
    
    # BFS with parent tracking
    visited = [[False for _ in range(grid_width)] for _ in range(grid_height)]
    parent = {}
    queue = deque([start_cell])
    visited[start_cell[1]][start_cell[0]] = True
    parent[start_cell] = None
    
    directions = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    found = False
    while queue and not found:
        curr_x, curr_y = queue.popleft()
        
        if (curr_x, curr_y) == goal_cell:
            found = True
            break
        
        for dx, dy in directions:
            next_x, next_y = curr_x + dx, curr_y + dy
            
            if next_x < 0 or next_x >= grid_width or next_y < 0 or next_y >= grid_height:
                continue
            
            if visited[next_y][next_x] or grid[next_y][next_x]:
                continue
            
            if dx != 0 and dy != 0:
                if grid[curr_y][next_x] or grid[next_y][curr_x]:
                    continue
            
            visited[next_y][next_x] = True
            parent[(next_x, next_y)] = (curr_x, curr_y)
            queue.append((next_x, next_y))
    
    if not found:
        return None
    
    path = []
    current = goal_cell
    while current is not None:
        path.append(cell_to_point(current[0], current[1]))
        current = parent.get(current)
    
    path.reverse()

    if len(path) > 10:
        path = [path[0]] + path[2::3] + [path[-1]]
    
    return path


def build_refinement_prompt(
    original_prompt: str, 
    analysis, 
    iteration: int,
    previous_json: dict = None
) -> str:
    """Build CONCISE feedback prompt - trimmed for better model performance."""

    top_warnings = analysis.warnings[:5]
    warnings_text = "\n".join(f"  - {w}" for w in top_warnings)
    
    feedback = f"""
ITERATION {iteration} - REFINEMENT NEEDED

Score: {analysis.overall_score:.1f}/100 (target: 80+)

CRITICAL ISSUES:
{warnings_text}

KEY SCORES:
  - Reachability: {analysis.scores.get('reachability', 0):.1f}/100 (MOST IMPORTANT)
  - Agent Placement: {analysis.scores.get('agent_placement', 0):.1f}/100
  - Obstacle Quality: {analysis.scores.get('obstacle_quality', 0):.1f}/100
"""
    
    # Only show previous attempt summary if reachability failed
    if analysis.scores.get('reachability', 100) < 80 and previous_json:
        feedback += f"""

YOUR PREVIOUS ATTEMPT HAD:
- {len(previous_json['map']['obstacles'])} wall segments
- {len(previous_json['agents'])} agents
"""
    
        if previous_json['map']['obstacles']:
            feedback += "\nFirst 3 walls:\n"
            for i, obs in enumerate(previous_json['map']['obstacles'][:3]):
                if isinstance(obs, dict):
                    p1, p2 = obs['p1'], obs['p2']
                else:
                    p1, p2 = obs[0], obs[1]
                feedback += f"  {i+1}. ({p1[0]:.1f},{p1[1]:.1f}) → ({p2[0]:.1f},{p2[1]:.1f})\n"
    
    feedback += f"""

ORIGINAL REQUEST: {original_prompt}

FIXES NEEDED:
"""

    if analysis.scores.get('reachability', 100) < 80:
        feedback += """
🚨 REACHABILITY FAILURE - agents cannot reach goals!
  Fix: Create CONNECTED rooms via hallways
  Fix: Ensure doorways are ≥1.2m wide
  Fix: Check walls form proper rectangles (no gaps/overlaps)
  Fix: Place agents INSIDE rooms, not in doorways
"""
    
    if analysis.scores.get('agent_placement', 100) < 80:
        feedback += """
⚠️  Place agents ≥0.6m from walls, centered in rooms
"""
    
    feedback += """

Generate CORRECTED JSON (raw JSON only, no markdown):
"""
    
    return feedback


def generate_with_refinement(
    prompt_text: str,
    args,
    min_score: float = 80.0,
    max_iterations: int = 3,
    verbose: bool = True
):
    """
    Generate scenario with automatic iterative refinement.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"=== Generating Scenario (Target: {min_score}/100) ===")
        print(f"{'='*60}\n")
    
    best_scenario = None
    best_analysis = None
    best_score = 0
    previous_json = None
    
    current_prompt = prompt_text
    
    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"\n{'─'*60}")
            print(f"Iteration {iteration}/{max_iterations}")
            print(f"{'─'*60}")
        
        # Generate scenario
        try:
            raw_json, repaired_json, logs = generate_scenario(current_prompt, args)
            previous_json = repaired_json
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            if iteration == max_iterations and best_scenario:
                print(f"\n⚠️  Using best result from previous iteration...")
                break
            continue
        
        if verbose and logs:
            print("\n📋 Repair logs:")
            for line in logs[:3]:
                print(f"   - {line}")
            if len(logs) > 3:
                print(f"   ... and {len(logs)-3} more")
        
        # Analyze quality
        scenario = dict_to_scenario(repaired_json)
        analysis = analyze_quality(scenario)
        score = analysis.overall_score
        
        if verbose:
            print(f"\n📊 Quality Score: {score:.1f}/100")
            if analysis.warnings:
                print(f"   ⚠️  {len(analysis.warnings)} warnings")
                for w in analysis.warnings[:3]:
                    print(f"      • {w}")
        
        
        if score > best_score:
            best_score = score
            best_scenario = repaired_json
            best_analysis = analysis
            if verbose and score > 0:
                print(f"   ✨ New best score!")
        
        if score >= min_score:
            if verbose:
                print(f"\n✅ Target reached! ({score:.1f} ≥ {min_score})")
            return best_scenario, best_analysis, iteration

        if iteration >= 2 and score <= best_score - 5:
            if verbose:
                print(f"   ⚠️  Score declining, stopping early")
            break

        if iteration < max_iterations:
            if verbose:
                print(f"   🔄 Refining (score {score:.1f} < target {min_score})...")
            current_prompt = build_refinement_prompt(
                prompt_text, 
                analysis, 
                iteration + 1,
                previous_json=previous_json
            )

    if verbose:
        print(f"\n⚠️  Max iterations reached. Best score: {best_score:.1f}/100")
    
    return best_scenario, best_analysis, max_iterations


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate navigation scenarios from natural language prompts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple corridor
  python generate.py "robot and 3 humans in a corridor passing each other"
  
  # Multi-room with high quality
  python generate.py "office with 3 connected rooms" --min-score 85
  
  # Use GPT-5 for better quality
  python generate.py "busy intersection" --model gpt-5
        """
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Natural-language description of the scenario",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=75.0,
        help="Minimum quality score to accept (default: 75)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum refinement attempts (default: 3)",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip initial visualization",
    )
    parser.add_argument(
        "--no-analysis",
        action="store_true",
        help="Skip printing detailed quality analysis",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output (only show final results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="scenarios/generated",
        help="Output directory for scenarios",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini", "gpt-5"],
        default="gpt-4.1",
        help="OpenAI model to use (default: gpt-4.1)",
    )
    return parser.parse_args()


def main():
    cli_args = parse_args()
    prompt_text = cli_args.prompt
    
    scenario_id = f"scenario_{uuid.uuid4().hex[:8]}"
    

    defaults_copy = DEFAULTS.copy()
    defaults_copy['output_dir'] = cli_args.output_dir
    defaults_copy['model'] = cli_args.model

    if cli_args.model == "gpt-5":
        defaults_copy['max_tokens'] = 32000
    elif cli_args.model == "gpt-4.1-nano":
        defaults_copy['max_tokens'] = 4000
    
    args = SimpleNamespace(
        **defaults_copy,
        scenario_id=scenario_id,
        model_name=cli_args.model,
    )
    
    verbose = not cli_args.quiet
    
    if verbose:
        print(f"\nPrompt: {prompt_text}")
        print(f"Scenario ID: {scenario_id}")
        print(f"Model: {cli_args.model}")
        print(f"Target Quality: {cli_args.min_score}/100")

    best_scenario, best_analysis, iterations = generate_with_refinement(
        prompt_text,
        args,
        min_score=cli_args.min_score,
        max_iterations=cli_args.max_iterations,
        verbose=verbose
    )
    
    if best_scenario is None:
        print("\n❌ Failed to generate any valid scenario.")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scenario_id}.json"
    save_json(best_scenario, out_path)

    print(f"\n{'='*60}")
    print("=== Results ===")
    print(f"{'='*60}")
    print(f"\n💾 Saved: {out_path}")
    print(f"🎯 Quality Score: {best_analysis.overall_score:.1f}/100")
    print(f"🔄 Iterations: {iterations}/{cli_args.max_iterations}")
    if not cli_args.no_analysis:
        print(f"\n{'='*60}")
        print("=== Quality Analysis ===")
        print(f"{'='*60}\n")
        print(str(best_analysis))

    if not cli_args.no_visualize:
        print(f"\n{'='*60}")
        print("=== Visualization ===")
        print(f"{'='*60}\n")
        
        try:
            scenario = load_scenario(str(out_path))
            scenario, _ = repair_scenario(scenario, aggressive=False)
            sim = scenario_to_simulator(scenario)
            
            visualize_scenario_with_path(sim, scenario)
        except Exception as e:
            print(f"⚠️  Visualization failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Next Steps:")
    print(f"{'='*60}")
    print(f"\n  Run simulation:")
    print(f"    python run_scenario.py {out_path} --visualize")
    print(f"\n  Analyze quality:")
    print(f"    python analyze_scenario.py {out_path}")
    print()


if __name__ == "__main__":
    main()