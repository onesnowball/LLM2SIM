#!/usr/bin/env python3
"""
Visualize only the initial state of a scenario (time=0).
No simulation, no animation - just the map with agent positions.
"""
import argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.cm as cm
import numpy as np

from scenario_io import load_scenario
from scenario_validation import repair_scenario
from scenario_adapter import scenario_to_simulator


def visualize_initial_state(scenario_path: str, save_path: str = None):
    """
    Visualize the initial state (t=0) of a scenario.
    
    Args:
        scenario_path: Path to scenario JSON file
        save_path: Optional path to save figure (PNG/PDF/SVG)
    """

    scenario = load_scenario(scenario_path)
    scenario, _ = repair_scenario(scenario, aggressive=False)
    sim = scenario_to_simulator(scenario)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    bounds = scenario.map.bounds
    xmin, ymin, xmax, ymax = bounds

    margin = 1.0
    ax.set_xlim(xmin - margin, xmax + margin)
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.set_aspect('equal')

    ax.set_title(
        f"Scenario: {scenario.metadata.get('scenario_id', 'Unknown')}\n"
        f"Initial State (t=0)",
        fontsize=14,
        fontweight='bold'
    )
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlabel('X (meters)', fontsize=11)
    ax.set_ylabel('Y (meters)', fontsize=11)

    if sim.obstacles:
        for wall in sim.obstacles:
            ax.plot(
                [wall.p1[0], wall.p2[0]], 
                [wall.p1[1], wall.p2[1]], 
                color='black', 
                linewidth=4, 
                zorder=2,
                solid_capstyle='round'
            )

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
                label='_nolegend_',
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

    ax.legend(
        loc='upper right', 
        fontsize=9,
        framealpha=0.9,
        edgecolor='gray'
    )
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved visualization to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize initial state of a scenario (no simulation/animation)"
    )
    parser.add_argument(
        "scenario_json",
        type=str,
        help="Path to scenario JSON file"
    )
    parser.add_argument(
        "--save",
        type=str,
        help="Save figure to file (e.g., output.png, output.pdf)"
    )
    
    args = parser.parse_args()
    
    visualize_initial_state(args.scenario_json, args.save)


if __name__ == "__main__":
    main()