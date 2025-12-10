# Example Prompts for Better Scenario Generation

## General Principles

1. **Be specific about geometry**: Mention dimensions, positions, and spatial relationships
2. **Describe agent motivations**: Where are they coming from and going to?
3. **Specify structural elements**: Walls, doorways, rooms with actual positions
4. **Use cardinal directions**: "from the left", "north wall", etc.

## Good Prompt Examples

### 1. Corridor Passing Scenario

**Good:**
```
Create a straight corridor from x=-10 to x=10, y from -2 to +2 (4 meters wide).
Place two parallel walls: one at y=-2 and one at y=+2, both spanning x=-10 to x=10.
Robot starts at (-8, 0) and needs to reach (8, 0).
Human 1 starts at (6, -1) walking toward (-6, 1) - will meet robot head-on.
Human 2 starts at (-6, 0.5) walking toward (6, -0.5) - walking same direction as robot initially.
```

**Why it's good:**
- Exact wall positions given
- Agent positions are explicit
- Creates clear passing interactions
- All coordinates ensure agents stay inside corridor

### 2. Doorway Congestion

**Good:**
```
Create a rectangular room at x=[0, 6], y=[0, 6] with walls on all sides.
Leave a 1.2-meter doorway in the bottom wall (y=0) centered at x=3 (from x=2.4 to x=3.6).
Robot starts inside at (3, 4) heading out to (-2, 0).
Two humans start outside at (-2, -1) and (-2, 1), both heading into room toward (3, 4).
This creates a bidirectional flow through a narrow doorway.
```

**Why it's good:**
- Room dimensions explicit
- Doorway position and width specified
- Clear convergence point (doorway)
- Realistic bidirectional scenario

### 3. T-Intersection

**Good:**
```
Create a T-intersection with:
- Horizontal corridor from x=-8 to x=8, walls at y=-1.5 and y=1.5
- Vertical corridor from y=1.5 to y=8, walls at x=-1.5 and x=1.5
- These corridors connect at the origin (0, 0)

Robot starts at (-6, 0) heading toward (0, 6) - must turn 90° at intersection.
Human 1 at (6, 0) heading toward (-6, 0) - crosses robot's path.
Human 2 at (0, 7) heading toward (0, 0) - approaches from ahead of robot.
```

**Why it's good:**
- Clear structural description
- Junction explicitly described
- Agent paths create interesting navigation challenge
- Coordinates respect the corridor structure

### 4. L-Shaped Corridor

**Good:**
```
Create an L-shaped corridor:
- Horizontal leg: x=[-10, 0], walls at y=-1.5 and y=1.5
- Vertical leg: x=[-1.5, 1.5], y=[0, 10], walls at x=-1.5 and x=1.5
- Corridors connect with a corner at (-1.5, 1.5) to (1.5, 1.5) and (-1.5, 0) to (-1.5, 1.5)

Robot starts at (-8, 0) heading to (0, 8) - must navigate the corner.
Human 1 starts at (0, 8) heading to (-8, 0) - opposite direction, will meet at corner.
Human 2 starts at (-5, 0) heading to (0, 5) - same path as robot but ahead.
```

**Why it's good:**
- Corner geometry precisely defined
- Agents will interact near the challenging corner turn
- Multiple interaction types (head-on, following)

### 5. Plaza with Central Obstacle

**Good:**
```
Create an open plaza from x=[-8, 8], y=[-8, 8].
Place a circular obstacle in the center approximated by an octagon:
- 8 line segments forming a circle of radius 2 meters centered at origin
- Vertices at angles 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

Robot starts at (-6, -6) heading to (6, 6) - must go around obstacle.
Human 1 at (6, -6) heading to (-6, 6) - perpendicular crossing path.
Human 2 at (-6, 0) heading to (6, 0) - passes behind robot.
Human 3 at (0, -6) heading to (0, 6) - passes in front of robot.
```

**Why it's good:**
- Open space clearly defined
- Obstacle shape specified (even with approximation method)
- Multiple crossing patterns
- Forces navigation around central feature

## Poor Prompt Examples (and why)

### ❌ Vague Prompt:
```
"Robot walks down a hallway with some people."
```
**Problems:**
- No dimensions
- "Some people" - how many? Where?
- No goals specified
- No structural detail

### ❌ Contradictory Prompt:
```
"Small narrow corridor 1 meter wide with 5 humans walking side by side."
```
**Problems:**
- Physically impossible (5 × 0.3m radius = 3m needed)
- Will force validator to relocate agents

### ❌ Missing Structure:
```
"Robot in a room with a door, trying to exit while humans enter."
```
**Problems:**
- Room size unknown
- Door location/size unknown
- Agent starting positions unspecified
- Goals unclear

## Template for Custom Scenarios

```
Create a [STRUCTURE_TYPE] scenario:

GEOMETRY:
- Map bounds: [xmin, ymin, xmax, ymax]
- [Structure 1]: walls at [positions]
- [Structure 2]: walls at [positions]
- [Openings/Doors]: at [positions], width [X] meters

AGENTS:
- Robot: starts at (x1, y1), goal (x2, y2)
  Motivation: [why this path]
- Human 1: starts at (x3, y3), goal (x4, y4)
  Motivation: [why this path]
- Human 2: starts at (x5, y5), goal (x6, y6)
  Motivation: [why this path]

INTERACTION:
[Describe what navigation challenge this creates]
```

## Tips for Success

1. **Draw it first**: Sketch your scenario on paper before writing the prompt
2. **Use consistent units**: Always meters
3. **Check feasibility**: Can agents physically fit? Do paths make sense?
4. **Specify wall continuity**: "Wall continues from (x1,y1) to (x2,y2) to (x3,y3)"
5. **Test with simple geometry first**: Start with basic corridors before complex layouts
6. **Think about agent clearance**: Leave 0.5-1m between agents and walls
7. **Create meaningful interactions**: Don't just scatter agents randomly