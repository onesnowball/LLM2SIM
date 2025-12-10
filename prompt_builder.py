from textwrap import dedent
from typing import Dict, Optional, Sequence


ENHANCED_SCHEMA_HINT = dedent(
    """
    Produce strictly valid JSON with these top-level keys:
      - metadata: {scenario_id, seed, prompt_text, model_name}
      - map: {type, bounds: [xmin, ymin, xmax, ymax], obstacles: [{p1:[x,y], p2:[x,y]}]}
      - agents: list of agents with {id, role, start: {x,y}, goal: {x,y}, radius, v_pref, behavior, group_id}
      - norms: {passing_side, min_distance}
      - sim: {dt, max_steps}
      - events: list (may be empty)

    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║  🚨 CRITICAL RULE: NEVER CREATE FULLY ENCLOSED SPACES                    ║
    ║                                                                           ║
    ║  Every enclosed area (room, cell, region) MUST have at least ONE         ║
    ║  opening/doorway that is at least 1.0m wide.                             ║
    ║                                                                           ║
    ║  ❌ NEVER: 4 walls with no gaps = trapped agents                         ║
    ║  ✓ ALWAYS: 4 walls with at least 1 doorway = accessible room            ║
    ╚═══════════════════════════════════════════════════════════════════════════╝

    CRITICAL SPATIAL REASONING PROCESS:
    
    Before generating coordinates, YOU MUST mentally work through this process:
    
    1. IDENTIFY REGIONS:
       - What distinct spaces does the prompt describe? (corridors, rooms, plazas, intersections)
       - Sketch a mental layout: Where does each region sit within the bounds?
       - Example: "L-shaped corridor" → horizontal leg from (-10,-1) to (0,-1), vertical leg from (0,-1) to (0,10)
    
    2. PLACE OBSTACLES TO DEFINE REGIONS:
       - Use line segments to create walls that bound each region
       - Connect segments precisely: end of one segment = start of next (same coordinates)
       - Leave deliberate gaps for doorways (≥ 1.0m wide for radius=0.3)
       - Example corridor: two parallel walls at y=-1.5 and y=1.5 from x=-10 to x=10
       
       🚨 MANDATORY RULE: EVERY ENCLOSED ROOM MUST HAVE AT LEAST ONE DOORWAY
       
       - **A room with 4 complete walls and NO gaps is INVALID**
       - **Check EVERY enclosed region before finalizing**
       - **If you create walls that form a closed shape, ADD A DOORWAY**
       
       Example of INVALID room (NO DOORWAY):
         ❌ [(-5,-5), (-5,5)], [(-5,5), (5,5)], [(5,5), (5,-5)], [(5,-5), (-5,-5)]
         This creates a box with NO EXIT - agents will be TRAPPED
       
       Example of VALID room (WITH DOORWAY):
         ✓ [(-5,-5), (-5,5)],     # left wall
           [(-5,5), (5,5)],       # top wall  
           [(5,5), (5,-5)],       # right wall
           [(5,-5), (2,-5)],      # bottom wall part 1
           [(0,-5), (-5,-5)]      # bottom wall part 2
         Gap from x=0 to x=2 creates a 2m doorway on the bottom wall
    
    3. VERIFY CONNECTIVITY:
       - Can agents walk from their start to their goal without crossing walls?
       - Are doorways wide enough? (minimum 1.0m, prefer 1.2-1.5m)
       - Do paths make intuitive sense?
       - **RUN MENTAL CHECK: Could I walk from any agent's start to their goal?**
    
    4. PLACE AGENTS LOGICALLY:
       - Start/goal should be in the FREE SPACE inside their intended region
       - Keep agents ≥ (radius + 0.3m) away from all walls
       - Never place agents or goals IN doorways or ON walls
       - Spread agents out: maintain ≥ 1.0m between different agents' starts
       - Match the narrative: "entering from the left" → start on left side
    
    5. ENSURE GOALS ARE MEANINGFUL:
       - Goals should be far enough from starts to create interesting navigation (≥ 3m)
       - Goals should be in open space, not corners or doorways
       - Don't cluster multiple agents' goals in the same spot unless intentional
    
    GEOMETRY RULES:
    
    * Coordinates are in meters. Bounds define the world: [-12, -12, 12, 12] is 24×24m
    * Agent radius ≈ 0.3m (human body). Personal space ≈ 0.6-1.0m
    * Corridor width should be 2-4m for comfortable passing
    * Room dimensions should be 3-8m per side
    * Doorways: 1.0-1.5m wide minimum
    * **MANDATORY**: Every enclosed room must have at least one doorway
    * **MANDATORY**: If agents need to travel between rooms, those rooms must be connected
    
    WALL CONSTRUCTION:
    
    * To make a continuous wall from (x1,y1) to (x2,y2) to (x3,y3):
      obstacles: [
        {"p1": [x1,y1], "p2": [x2,y2]},  // segment 1
        {"p1": [x2,y2], "p2": [x3,y3]}   // segment 2 starts where segment 1 ends
      ]
    
    * To make a room (4 walls with a door on one side):
      
      STEP 1: Identify which wall will have the doorway
      STEP 2: Create 3 complete walls
      STEP 3: Split the 4th wall into 2 segments with a gap ≥ 1.0m
      
      Example: room from (0,0) to (5,5) with door on bottom wall centered at x=2.5:
        {"p1": [0,0], "p2": [0,5]},      // left wall (complete)
        {"p1": [0,5], "p2": [5,5]},      // top wall (complete)
        {"p1": [5,5], "p2": [5,0]},      // right wall (complete)
        {"p1": [5,0], "p2": [2.0,0]},    // bottom wall part 1 (x: 5→2)
        {"p1": [3.0,0], "p2": [0,0]}     // bottom wall part 2 (x: 3→0)
      
      The gap between x=2.0 and x=3.0 creates a 1.0m doorway ✓
    
    * For multiple connected rooms:
      - Room A needs doorway to hallway
      - Room B needs doorway to hallway
      - Room C needs doorway to hallway
      - Doorways must align spatially (e.g., all open to same hallway)
      
      Example: Three rooms around a central hallway
        Hallway: x=[-2, 2], y=[-2, 2]
        Room 1: x=[-8, -2], y=[-2, 2] with doorway on right wall (facing hallway)
        Room 2: x=[2, 8], y=[-2, 2] with doorway on left wall (facing hallway)
        Room 3: x=[-2, 2], y=[2, 8] with doorway on bottom wall (facing hallway)
    
    * For curved boundaries, use multiple short segments to approximate the curve
    
    AGENT PLACEMENT EXAMPLES:
    
    * Corridor (y ∈ [-1.5, 1.5], x ∈ [-10, 10]):
      ✓ start: {x: -8, y: 0}, goal: {x: 8, y: 0}  // centered, long path
      ✓ start: {x: -8, y: -0.5}, goal: {x: 8, y: 0.5}  // slight diagonal
      ✗ start: {x: -8, y: -1.5}, goal: {x: 8, y: 1.5}  // ON the walls!
      ✗ start: {x: -8, y: -2.0}, goal: {x: 8, y: 2.0}  // OUTSIDE corridor!
    
    * Room with door (room: [0,0] to [5,5], door at [2,3] on bottom wall):
      ✓ start: {x: 2.5, y: 2.5}, goal: {x: -3, y: 0}  // inside → exit through door
      ✗ start: {x: 2.5, y: 0}, goal: {x: -3, y: 0}    // start is IN the doorway!
    
    VALIDATION CHECKLIST (before outputting JSON):
    
    [ ] All agent starts are inside bounds and in valid free space
    [ ] All agent goals are inside bounds and in valid free space
    [ ] No agent start/goal is within 0.5m of any wall
    [ ] No agent start/goal is placed in a doorway opening
    [ ] Wall segments connect properly (shared endpoints)
    [ ] Doorways are ≥ 1.0m wide
    [ ] **CRITICAL**: Every enclosed room has at least one doorway/opening
    [ ] **CRITICAL**: All rooms that agents need to travel between are connected
    [ ] Each agent has a clear path from start to goal
    [ ] Agent starts are spread out (≥ 1.0m apart)
    [ ] At least one robot (id=0) and zero or more humans (id≥1)
    [ ] **CRITICAL**: NO fully enclosed boxes exist (check by counting walls around each region)
    
    HOW TO CHECK FOR ENCLOSED BOXES:
    
    1. List all regions you've created (rooms, corridors, etc.)
    2. For each region, count the walls:
       - If 4 walls with no gaps → INVALID (add doorway)
       - If 4 walls with 1+ gap ≥1.0m → VALID
       - If <4 walls (open space) → VALID
    3. Fix any fully enclosed boxes by removing part of one wall
    
    OUTPUT FORMAT:
    
    {
      "metadata": {...},
      "map": {
        "type": "corridor",  // or "intersection", "plaza", "l_corridor", "rooms", etc.
        "bounds": [-12.0, -12.0, 12.0, 12.0],
        "obstacles": [
          {"p1": [x1, y1], "p2": [x2, y2]},
          ...
        ]
      },
      "agents": [
        {
          "id": 0,
          "role": "robot",
          "start": {"x": -8.0, "y": 0.0},
          "goal": {"x": 8.0, "y": 0.0},
          "radius": 0.3,
          "v_pref": 1.0,
          "behavior": "social_force",
          "group_id": null
        },
        ...
      ],
      "norms": {"passing_side": "right", "min_distance": 0.6},
      "sim": {"dt": 0.1, "max_steps": 600},
      "events": []
    }
    """
).strip()


def build_system_prompt(extra_guidance: Optional[str] = None) -> str:
    prompt = [
        "You are an expert in spatial reasoning and crowd simulation scenario design.",
        "Your task is to generate geometrically coherent navigation scenarios.",
        "",
        "🚨 CRITICAL RULE #1: NEVER CREATE FULLY ENCLOSED SPACES",
        "   Every room/region you create MUST have at least one opening (doorway) ≥1.0m wide.",
        "   A box with 4 walls and NO gaps will TRAP agents inside.",
        "",
        "IMPORTANT: Before writing any coordinates, you must mentally plan the layout:",
        "1. Identify all regions (corridors, rooms, etc.)",
        "2. Decide where each region sits in the coordinate space",
        "3. Place walls to define regions with proper connections",
        "4. **FOR EACH ENCLOSED REGION: Ensure at least ONE doorway exists**",
        "5. Only then place agents in logical positions within free space",
        "",
        "⚠️ BEFORE FINALIZING: Count walls around each region:",
        "   - If 4 walls + 0 gaps = INVALID (add doorway immediately)",
        "   - If 4 walls + 1+ gaps ≥1.0m = VALID",
        "",
        "Follow the schema exactly and respond with raw JSON only.",
        "",
        ENHANCED_SCHEMA_HINT,
    ]
    if extra_guidance:
        prompt.append("\nADDITIONAL GUIDANCE:")
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
        if bounds is not None
        else "bounds [-12.0, -12.0, 12.0, 12.0]"
    )

    if human_count is not None:
        human_text = (
            f"Include exactly {human_count} humans (in addition to 1 robot). "
            "Each human should have a distinct, meaningful start and goal."
        )
    else:
        human_text = (
            "Create one human for each distinct person in the prompt. "
            "If the prompt is vague (e.g., 'busy crowd'), use 3-5 humans."
        )

    if map_type == "auto":
        map_type_clause = (
            "Choose map.type as a descriptive string matching the environment "
            "(e.g., 'corridor', 'intersection', 'plaza', 'l_corridor', 'three_rooms_with_hallway')."
        )
    else:
        map_type_clause = f'map.type = "{map_type}"'

    meta = metadata_overrides or {}
    scenario_id = meta.get("scenario_id", "generated_temp")
    model_name = meta.get("model_name", "llm-generated")
    seed = meta.get("seed", "42")

    return dedent(
        f"""
        USER PROMPT: "{prompt_text}"
        
        REQUIREMENTS:
          - {map_type_clause} with {bounds_text}
          - {human_text}
          - metadata: scenario_id="{scenario_id}", seed={seed}, model_name="{model_name}"
          - metadata.prompt_text MUST exactly match the user prompt above
          - Use map.obstacles to construct all geometry (walls, rooms, doorways) from line segments
          - Ensure events is an empty array [] unless specific events are described
        
        🚨 CRITICAL INSTRUCTIONS - READ CAREFULLY:
          
          1. BEFORE creating any walls, plan the layout mentally
          2. Identify all regions that will be enclosed (rooms, cells, etc.)
          3. FOR EACH ENCLOSED REGION: Plan where the doorway will be
          4. Build walls segment by segment, leaving gaps for doorways
          5. Place agents in free space, away from walls, with clear paths to goals
          
          6. **FINAL CHECK BEFORE RESPONDING:**
             - Count walls around each region
             - If any region has 4 walls with NO gaps → ADD A DOORWAY NOW
             - Verify: Can I walk from every agent's start to their goal?
             - Verify: Are all doorways at least 1.0m wide?
        
        COMMON MISTAKES TO AVOID:
          ❌ Creating a rectangle with 4 walls and no doorway
          ❌ Making doorways too narrow (<1.0m)
          ❌ Placing agents inside enclosed boxes
          ❌ Forgetting to leave gaps between wall segments
        
        CORRECT APPROACH:
          ✓ Plan doorway locations BEFORE creating walls
          ✓ Split one wall into two segments with a gap
          ✓ Verify connectivity with BFS pathfinding mentally
          ✓ Place agents well inside open spaces
        
        OUTPUT: Raw JSON only. No markdown fences, no explanations.
        """
    ).strip()


def build_example_corridor_prompt():
    """Example of a well-structured corridor prompt"""
    return dedent("""
        Create a straight corridor scenario:
        - Corridor runs horizontally from x=-10 to x=10
        - Corridor walls at y=-2 and y=2 (4 meters wide)
        - Robot starts at (-8, 0) heading to (8, 0)
        - Two humans: one at (-6, -1) heading to (6, 1), another at (6, -0.5) heading to (-6, 0.5)
        - This creates a passing scenario in the middle of the corridor
    """).strip()


def build_example_room_prompt():
    """Example of a well-structured room scenario WITH DOORWAY"""
    return dedent("""
        Create a room with a doorway (IMPORTANT: room must not be fully enclosed):
        - Room boundaries: x ∈ [0, 6], y ∈ [0, 6]
        - Doorway on the left wall (x=0) centered at y=3, width 1.2m
        - Three walls complete: top, right, bottom
        - Left wall split into two segments with 1.2m gap for doorway
        - Robot starts inside room at (3, 3), goal outside at (-3, 3)
        - Two humans start outside at (-4, 2) and (-4, 4), both heading into room toward (4, 3)
        - Creates a doorway congestion scenario
    """).strip()