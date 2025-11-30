from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Tuple

from llm_client import LLMClient, LLMConfig
from prompt_builder import build_system_prompt, build_user_prompt
from scenario_io import dict_to_scenario
from scenario_validator import validate_and_repair


JSON_FENCE_RE = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json_block(text: str) -> str:
    """
    Remove markdown fences or reasoning content, returning the JSON payload.
    """
    match = JSON_FENCE_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def load_prompts(prompt_arg: str | None, prompt_file: str | None) -> List[str]:
    if prompt_arg:
        return [prompt_arg.strip()]
    if not prompt_file:
        raise ValueError("Either --prompt or --prompt-file must be provided.")
    path = Path(prompt_file)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def scenario_to_dict(scenario) -> dict:
    """
    Convert Scenario dataclass (with tuples) into a JSON-serializable dict.
    """
    data = asdict(scenario)
    return data


def generate_scenario(prompt_text: str, args) -> Tuple[dict, dict, List[str]]:
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(
        prompt_text,
        map_type=args.map_type,
        human_count=args.humans,
        metadata_overrides={
            "scenario_id": args.scenario_id or "generated_temp",
            "seed": str(args.seed),
            "model_name": args.model_name,
        },
    )

    config = LLMConfig(
        provider=args.provider,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,
    )
    client = LLMClient(config)
    
    # Retry loop for JSON parsing
    last_err = None
    for attempt in range(3):
        raw_text = client.generate(system_prompt, user_prompt)
        cleaned = extract_json_block(raw_text)
        try:
            parsed = json.loads(cleaned)
            break
        except json.JSONDecodeError as e:
            last_err = e
            print(f"  JSON parse error (attempt {attempt+1}/3): {e}")
            if attempt == 2:
                print(f"  Raw LLM output:\n{raw_text[:500]}...")
                raise RuntimeError(f"Failed to parse JSON after 3 attempts: {last_err}") from e
    
    scenario = dict_to_scenario(parsed)
    scenario, logs = validate_and_repair(scenario)
    return parsed, scenario_to_dict(scenario), logs


def simulate_scenario(scenario_dict: dict, visualize: bool = False) -> dict:
    from scenario_adapter import scenario_to_simulator  # Lazy import

    scenario = dict_to_scenario(scenario_dict)
    sim = scenario_to_simulator(scenario)
    obs, info = sim.reset(seed=scenario.metadata.get("seed", 0))
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        obs, reward, terminated, truncated, info = sim.step(action=None)
        steps += 1
    metrics = sim.calculate_metrics()
    if visualize:
        try:
            sim.visualize_simulation(show_plot=True)
        except Exception as exc:  # pragma: no cover - visualization best effort
            print("Visualization failed:", exc)
    return {"steps": steps, "metrics": metrics, "status": getattr(sim, "status", None)}


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scenarios via LLM.")
    parser.add_argument("--prompt", help="Single prompt text.")
    parser.add_argument("--prompt-file", help="File with one prompt per line.")
    parser.add_argument("--provider", choices=["llama", "gemini"], default="llama")
    parser.add_argument("--model", help="Override model name for provider.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--map-type", default="corridor")
    parser.add_argument("--humans", type=int, help="Approximate human count.")
    parser.add_argument("--scenario-id", help="Override scenario_id.")
    parser.add_argument("--model-name", default="llm-generated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="scenarios/generated")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--count", type=int, default=1, help="Prompts per line to generate.")
    return parser.parse_args()


def main():
    args = parse_args()
    prompts = load_prompts(args.prompt, args.prompt_file)
    output_dir = Path(args.output_dir)

    for idx, prompt_text in enumerate(prompts[: args.count]):
        print(f"=== Generating scenario {idx+1}/{min(len(prompts), args.count)} ===")
        raw, repaired, logs = generate_scenario(prompt_text, args)

        scenario_id = repaired["metadata"].get("scenario_id", f"generated_{idx}")
        raw_path = output_dir / f"{scenario_id}_raw.json"
        repaired_path = output_dir / f"{scenario_id}.json"

        save_json(raw, raw_path)
        save_json(repaired, repaired_path)
        if logs:
            print("Validator logs:")
            for line in logs:
                print("  -", line)
        else:
            print("Validator logs: none")

        if args.simulate:
            summary = simulate_scenario(repaired, args.visualize)
            print("Simulation summary:", summary["status"], "steps:", summary["steps"])
            print("Robot metrics:", summary["metrics"].get("robot"))


if __name__ == "__main__":
    main()

