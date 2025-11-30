D# Scenario JSON Schema — Week 1

Each scenario JSON is a single object with this top-level structure:

    {
      "metadata": { ... },
      "map": { ... },
      "agents": [ ... ],
      "norms": { ... },
      "sim": { ... },
      "events": [ ... ]
    }

All six top-level keys must be present.

---

## metadata

Describes how the scenario was created and identified.

### Required fields

- scenario_id (string)  
  Unique identifier, e.g. "corridor_001".

- seed (integer)  
  Random seed used when running the simulator.

- prompt_text (string)  
  Natural-language description that produced this scenario.

- model_name (string)  
  Name of the generator, e.g. "handwritten" or "gpt-4.1".

### Example

    "metadata": {
      "scenario_id": "corridor_001",
      "seed": 42,
      "prompt_text": "A robot walking down the middle of a corridor while two humans approach from the opposite side.",
      "model_name": "handwritten"
    }

---

## map

Describes the static environment (bounds and obstacles).

### Required fields

- type (string)  
  One of: "corridor", "intersection", "plaza", "rooms".

- bounds (array of 4 numbers)  
  [xmin, ymin, xmax, ymax], e.g. [-6.0, -3.0, 6.0, 3.0].

### Optional fields

- obstacles (array of objects)  
  Each obstacle has:
  - p1: [x, y]
  - p2: [x, y]  

  These two points define a line segment and map directly to CrowdSimulator.add_obstacle(p1, p2).

### Example

    "map": {
      "type": "corridor",
      "bounds": [-6.0, -3.0, 6.0, 3.0],
      "obstacles": [
        { "p1": [-6.0, -3.0], "p2": [6.0, -3.0] },
        { "p1": [-6.0,  3.0], "p2": [6.0,  3.0] }
      ]
    }

---

## agents

Lists all moving agents (one robot and zero or more humans).

### Required fields per agent

- id (int)  
  Unique within the scenario. Convention: robot id = 0, humans id = 1, 2, ...

- role (string)  
  "robot" or "human".

- start (object)  
  - x (float) — starting x-position.  
  - y (float) — starting y-position.

- goal (object)  
  - x (float) — goal x-position.  
  - y (float) — goal y-position.

- radius (float)  
  Agent radius in meters, e.g. 0.3.

- v_pref (float)  
  Preferred walking speed in m/s, e.g. 1.0.

- behavior (string)  
  High-level behavior tag. For Week 1:
  - "social_force" — mapped to SocialForceModel.  
  - "goal" — reserved for simple goal-directed policies (future).  
  - "RL" — reserved for RL policies (future).

### Optional fields

- group_id (int or null)  
  Social group identifier. For Week 1, this is metadata; the adapter may register groups without changing policies.

### Validator constraints (Week 1)

- At least one agent with role == "robot".  
- Zero or more agents with role == "human".  
- All start and goal positions are clamped into map.bounds.  
- Initial positions that overlap are jittered apart until they are separated by at least roughly the sum of their radii.

### Example

    "agents": [
      {
        "id": 0,
        "role": "robot",
        "start": { "x": 0.0, "y": -2.0 },
        "goal":  { "x": 0.0, "y":  2.0 },
        "radius": 0.3,
        "v_pref": 1.0,
        "behavior": "social_force",
        "group_id": null
      },
      {
        "id": 1,
        "role": "human",
        "start": { "x": -1.0, "y": 1.0 },
        "goal":  { "x": -1.0, "y": -1.5 },
        "radius": 0.3,
        "v_pref": 1.0,
        "behavior": "social_force",
        "group_id": 1
      },
      {
        "id": 2,
        "role": "human",
        "start": { "x": 1.0, "y": 1.0 },
        "goal":  { "x": 1.0, "y": -1.5 },
        "radius": 0.3,
        "v_pref": 1.0,
        "behavior": "social_force",
        "group_id": 1
      }
    ]

---

## norms

Stores simple social norms and comfort parameters.

### Required fields

- passing_side (string)  
  "right" or "left".

- min_distance (float)  
  Comfort distance between agents, e.g. 0.6.

### Validator behaviour

- If passing_side is missing, the validator inserts "right".  
- If min_distance is missing, the validator inserts 0.6.  
- The validator may increase min_distance if it is inconsistent with agent radii (e.g. smaller than about twice the maximum radius).

### Example

    "norms": {
      "passing_side": "right",
      "min_distance": 0.6
    }

---

## sim

Configures the simulator.

### Required fields

- dt (float)  
  Simulation time step in seconds. Mapped to CrowdSimulator(time_step=dt).

- max_steps (int)  
  Maximum number of steps per run. Mapped to CrowdSimulator(max_steps=max_steps).

### Example

    "sim": {
      "dt": 0.25,
      "max_steps": 200
    }

---

## events

Describes optional time-based events.

### Required

- events (array)  
  For Week 1, this key must exist but can be an empty list [].  
  Any contents are ignored by the adapter and validator.

### Example

    "events": []

---

## Complete Week 1 Example

A full example JSON scenario that follows this schema:

    {
      "metadata": {
        "scenario_id": "corridor_001",
        "seed": 42,
        "prompt_text": "A robot walking down the middle of a corridor while two humans approach from the opposite side.",
        "model_name": "handwritten"
      },
      "map": {
        "type": "corridor",
        "bounds": [-6.0, -3.0, 6.0, 3.0],
        "obstacles": [
          { "p1": [-6.0, -3.0], "p2": [6.0, -3.0] },
          { "p1": [-6.0,  3.0], "p2": [6.0,  3.0] }
        ]
      },
      "agents": [
        {
          "id": 0,
          "role": "robot",
          "start": { "x": 0.0, "y": -2.0 },
          "goal":  { "x": 0.0, "y":  2.0 },
          "radius": 0.3,
          "v_pref": 1.0,
          "behavior": "social_force",
          "group_id": null
        },
        {
          "id": 1,
          "role": "human",
          "start": { "x": -1.0, "y": 1.0 },
          "goal":  { "x": -1.0, "y": -1.5 },
          "radius": 0.3,
          "v_pref": 1.0,
          "behavior": "social_force",
          "group_id": 1
        },
        {
          "id": 2,
          "role": "human",
          "start": { "x": 1.0, "y": 1.0 },
          "goal":  { "x": 1.0, "y": -1.5 },
          "radius": 0.3,
          "v_pref": 1.0,
          "behavior": "social_force",
          "group_id": 1
        }
      ],
      "norms": {
        "passing_side": "right",
        "min_distance": 0.6
      },
      "sim": {
        "dt": 0.25,
        "max_steps": 200
      },
      "events": []
    }
