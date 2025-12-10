import argparse

from scenario_io import load_scenario
from scenario_validation import repair_scenario
from scenario_adapter import scenario_to_simulator


def run_once(path: str, visualize: bool = False, output_file: str = None):

    scenario = load_scenario(path)
    scenario, logs = repair_scenario(scenario, aggressive=False)

    print(f"Loaded scenario_id = {scenario.metadata.get('scenario_id')}")
    print("Validator logs:")
    if logs:
        for line in logs:
            print("  -", line)
    else:
        print("  (none)")

    sim = scenario_to_simulator(scenario)

    seed = scenario.metadata.get("seed", 0)
    obs, info = sim.reset(seed=seed)

    terminated = False
    truncated = False
    step_count = 0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = sim.step(action=None)
        step_count += 1

    print(f"Simulation finished in {step_count} steps.")
    status = info.get("status", "unknown")
    print("Final sim.status:", status)

    metrics = sim.calculate_metrics()
    robot_metrics = metrics.get("robot", {})
    for k, v in robot_metrics.items():
        print(f"  {k}: {v}")

    anim_html = None
    if visualize or output_file is not None:
        try:
            anim_html = sim.visualize_simulation(
                output_file=output_file,
                show_plot=visualize,
            )
        except Exception as e:
            print("Visualization failed:", e)

    return metrics, anim_html

def main():
    parser = argparse.ArgumentParser(description="Run a JSON-defined crowd scenario.")
    parser.add_argument("scenario_json", type=str, help="Path to scenario JSON file.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, call visualize_simulation() after the run."
    )
    args = parser.parse_args()
    run_once(args.scenario_json, visualize=args.visualize)


if __name__ == "__main__":
    main()
