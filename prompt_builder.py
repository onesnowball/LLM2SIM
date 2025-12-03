from __future__ import annotations

from textwrap import dedent
from typing import Dict, Optional, Sequence


from textwrap import dedent
from typing import Dict, Optional, Sequence


BASE_SCHEMA_HINT = dedent(
    """
    Produce strictly valid JSON with these top-level keys:
      - metadata: {
          scenario_id (string),
          seed (int),
          prompt_text (string),
          model_name (string)
        }
      - map: {
          type (string),
          bounds: [xmin, ymin, xmax, ymax],
          obstacles: [{p1:[x,y], p2:[x,y]}]
        }
      - agents: list of agents; each agent has:
          id (int),
          role ("robot" or "human"),
          start: {x, y},
          goal: {x, y},
          radius (float, ≈0.3 by default),
          v_pref (float, ≈1.0 by default),
          behavior (string, e.g. "social_force"),
          group_id (int or null)
      - norms: {
          passing_side ("right" or "left"),
          min_distance (float)
        }
      - sim: {
          dt (float, seconds),
          max_steps (int)
        }
      - events: list (may be empty)

    Geometry and map rules:
      * The map is a 2D plane with coordinates in meters.
      * map.obstacles is a list of solid line-segment walls. Agents must not pass through
        these segments.
      * You may create any layout that matches the text description: corridors, rooms,
        intersections, plazas, streets, alleys, etc., using only straight line segments.
      * To create continuous walls, connect segments by making the end of one segment
        equal (or extremely close) to the start of the next.
      * To approximate curved boundaries, use several short straight segments.
      * To create a doorway or opening, leave a deliberate gap between colinear (or nearly
        colinear) wall segments. The gap should be wide enough for agents to pass
        (≥ 0.8 meters is reasonable for radius≈0.3).
      * Corridors or streets are formed by two roughly parallel chains of wall segments
        that create a clear band of free space between them.
      * All agent start and goal positions must lie inside map.bounds, and should not be
        inside any wall (keep at least radius + 0.1 meters away from every obstacle).
      * The placement of agents and goals must respect the geometric intent of the map:
        they should stay inside the described rooms, corridors, or plazas, not on the
        wrong side of solid walls.
      * Doors are clear gaps in walls wide enough for agents to pass. Do NOT place goals
        directly on top of walls, exactly in the middle of a door gap, or just outside
        of a corridor or room boundary.
      * In corridor scenes, all humans and the robot should remain inside the corridor
        band formed by the parallel wall chains, unless the description explicitly says
        they leave the corridor.
      * For every agent, there must exist at least one continuous collision-free path
        from its start to its goal that stays entirely inside the free space bounded by
        the walls and door openings.

    Constraints:
      * Include at least one robot (id = 0) and zero or more humans (ids ≥ 1).
      * Use meters and seconds (radius≈0.3, v_pref≈1.0 unless the text specifies otherwise).
      * Output JSON only. NO narration, code fences, or comments.

    Example (very small scenario):

    {
      "metadata": {
        "scenario_id": "example_01",
        "seed": 42,
        "prompt_text": "short description",
        "model_name": "example-model"
      },
      "map": {
        "type": "corridor",
        "bounds": [-5.0, -3.0, 5.0, 3.0],
        "obstacles": [
          {"p1": [-5.0, -1.0], "p2": [5.0, -1.0]},
          {"p1": [-5.0,  1.0], "p2": [5.0,  1.0]}
        ]
      },
      "agents": [
        {
          "id": 0,
          "role": "robot",
          "start": {"x": -4.0, "y": 0.0},
          "goal": {"x": 4.0, "y": 0.0},
          "radius": 0.3,
          "v_pref": 1.0,
          "behavior": "social_force",
          "group_id": null
        }
      ],
      "norms": {
        "passing_side": "right",
        "min_distance": 0.6
      },
      "sim": {
        "dt": 0.1,
        "max_steps": 600
      },
      "events": []
    }

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
    # How to handle map bounds
    bounds_text = (
        f"bounds {list(bounds)}"
        if bounds is not None
        else "reasonable bounds such as [-6.0, -3.0, 6.0, 3.0]"
    )

    if human_count is not None:
        human_text = (
            f"Include exactly {human_count} humans in addition to a single robot. "
            "Each distinct person described in the prompt (e.g., 'a human', "
            "'another human', 'a pedestrian') should correspond to a human agent."
        )
    else:
        human_text = (
            "Create one human agent for each distinct person explicitly or implicitly "
            "described in the prompt (e.g., 'one human', 'another person', 'a pedestrian'), "
            "in addition to a single robot. If the description is vague (e.g., 'busy crowd'), "
            "choose a small, reasonable number of humans consistent with how crowded it sounds."
        )
    


    # How to talk about map.type
    if map_type == "auto":
        map_type_clause = (
            "Choose map.type as a short, descriptive snake_case string that matches "
            "the environment described in the prompt (e.g., \"corridor\", "
            "\"intersection\", \"plaza\", \"two_way_street_with_doors\")."
        )
    else:
        map_type_clause = f'map.type = "{map_type}"'

    meta = metadata_overrides or {}
    scenario_id = meta.get("scenario_id", "generated_temp")
    model_name = meta.get("model_name", "llm-generated")
    seed = meta.get("seed", "42")

    return dedent(
        f"""
        Prompt: "{prompt_text}"
        Requirements:
          - {map_type_clause} with {bounds_text}.
          - {human_text}
          - Use map.obstacles to create any necessary walls, streets, rooms, doors, or other linear structures; all solid geometry must be built from line segments.
          - Use metadata.scenario_id="{scenario_id}", metadata.seed={seed}, metadata.model_name="{model_name}".
          - metadata.prompt_text MUST echo the original prompt.
          - Ensure events is an array (use [] if no events).
          - Provide diverse agent start/goal positions without overlaps.
        Output: raw JSON only. Do not wrap in markdown fences.
        """
    ).strip()


