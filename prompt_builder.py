from __future__ import annotations

from textwrap import dedent
from typing import Dict, Optional, Sequence


BASE_SCHEMA_HINT = dedent(
    """
    Produce strictly valid JSON with these top-level keys:
      - metadata: {scenario_id (string), seed (int), prompt_text (string), model_name (string)}
      - map: {type, bounds [xmin, ymin, xmax, ymax], obstacles:[{p1:[x,y], p2:[x,y]}]}
      - agents: list of agents; each agent has id, role ("robot" or "human"),
                start{x,y}, goal{x,y}, radius, v_pref, behavior ("social_force"), group_id
      - norms: {passing_side ("right" or "left"), min_distance}
      - sim: {dt, max_steps}
      - events: list (empty allowed)
    Constraints:
      * Include >=1 robot (id 0) and >=0 humans (ids >=1).
      * Clamp starts/goals to map bounds.
      * Use meters and seconds (radius≈0.3, v_pref≈1.0 unless specified).
      * Output JSON only. NO narration, code fences, or comments.
    """
).strip()


def build_system_prompt(extra_guidance: Optional[str] = None) -> str:
    prompt = [
        "You help generate crowd-navigation scenarios for a social robot simulator.",
        "Follow the provided schema exactly and respond with raw JSON only.",
        BASE_SCHEMA_HINT,
    ]
    if extra_guidance:
        prompt.append(extra_guidance.strip())
    return "\n".join(prompt)


def build_user_prompt(
    prompt_text: str,
    *,
    map_type: str = "corridor",
    bounds: Sequence[float] | None = None,
    human_count: int | None = None,
    metadata_overrides: Optional[Dict[str, str]] = None,
) -> str:
    bounds_text = (
        f"bounds {list(bounds)}"
        if bounds
        else "reasonable bounds such as [-6.0, -3.0, 6.0, 3.0]"
    )
    human_text = (
        f"Include about {human_count} humans."
        if human_count is not None
        else "Include at least two humans unless the description says otherwise."
    )
    meta = metadata_overrides or {}
    scenario_id = meta.get("scenario_id", "generated_temp")
    model_name = meta.get("model_name", "llm-generated")
    seed = meta.get("seed", "42")

    return dedent(
        f"""
        Prompt: "{prompt_text}"
        Requirements:
          - map.type = "{map_type}" with {bounds_text}.
          - {human_text}
          - Use metadata.scenario_id="{scenario_id}", metadata.seed={seed}, metadata.model_name="{model_name}".
          - metadata.prompt_text MUST echo the original prompt.
          - Ensure events is an array (use [] if no events).
          - Provide diverse agent start/goal positions without overlaps.
        Output: raw JSON only. Do not wrap in markdown fences.
        """
    ).strip()

