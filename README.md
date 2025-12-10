# ROB599 - LLM-Powered Crowd Simulation

Generate realistic crowd navigation scenarios from natural language descriptions using LLMs (GPT-4.1/GPT-5), validate them with automatic quality scoring.

## Overview

This project enables rapid creation of robot navigation datasets by:
1. **Generating** scenarios from natural language prompts using OpenAI's API
2. **Validating** scenario quality with BFS pathfinding and geometric checks
3. **Refining** scenarios automatically through iterative feedback
4. **Simulating** navigation using the Social Force Model
5. **Visualizing** results with matplotlib animations

## Quick Start

### Prerequisites

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-key-here"
```

**Required packages**: `openai`, `gymnasium`, `numpy`, `matplotlib`

### Generate a Single Scenario

```bash
# Generate one scenario with automatic quality refinement
python generate.py "robot navigating a corridor with 3 humans passing from opposite direction"
```

This will:
- Generate the scenario using GPT-4.1
- Automatically refine it until quality score ≥75/100 (up to 3 iterations)
- Save to `scenarios/generated/scenario_XXXXXXXX.json`
- Show quality analysis and visualization

### Batch Generate Multiple Scenarios

```bash
# Interactive batch generation
python batch_generate.py

# When prompted, enter:
> 50 scenarios of robot navigating through various corporate environments
```

Options:
- `--model gpt-5` - Use GPT-5 for higher quality (slower)
- `--min-score 85` - Require higher quality threshold
- `--fast` - Skip expensive BFS checks (faster but less thorough)
- `--output-dir dataset_name` - Custom output directory

## Project Structure

### Core Pipeline

```
Text Prompt → LLM → JSON → Validator → Refined JSON → Simulator → Metrics
```

### Key Files

| File | Purpose |
|------|---------|
| **`generate.py`** | Single scenario generator with automatic refinement |
| **`batch_generate.py`** | Batch scenario generator with interactive prompts |
| **`llm_client.py`** | OpenAI API wrapper with retry logic |
| **`prompt_builder.py`** | Constructs schema-aware prompts for the LLM |
| **`scenario_validation.py`** | Quality analysis with BFS pathfinding & geometric repair |
| **`scenario_io.py`** | Load/parse scenario JSON files |
| **`scenario_types.py`** | Dataclass definitions for scenarios |
| **`scenario_adapter.py`** | Convert scenarios to simulator objects |
| **`simulator.py`** | Social Force Model crowd simulator |
| **`run_scenario.py`** | Run simulation on existing JSON files |
| **`visualize_scenario.py`** | Static visualization (t=0 only) |
| **`analyze_scenario.py`** | Quality analysis tool |

## Quality Validation

The validator performs automatic quality analysis with these components:

### 1. Agent Placement
- Agents must be ≥0.5m from walls
- No overlapping start positions
- Minimum 2m path length required

### 2. Obstacle Quality
- No degenerate walls (length < 0.01m)
- Checks for fully enclosed rooms **without doorways** (major penalty)
- Validates wall connectivity

### 3. Reachability
- **BFS pathfinding** verifies agents can reach goals
- Checks clearance around obstacles
- Detects trapped agents

### 4. Spatial Distribution
- Evaluates agent spread across environment
- Penalizes clustering

### Quality Scores
- **80-100**: Excellent - ready for use
- **60-79**: Good - minor issues
- **40-59**: Fair - notable problems
- **0-39**: Poor - major issues

## Advanced Usage

### Custom Model Selection

```bash
# Use GPT-5 for best quality
python generate.py "complex office with 5 connected rooms" --model gpt-5

# Use nano model for speed
python generate.py "simple corridor" --model gpt-4.1-nano
```

### Higher Quality Thresholds

```bash
# Require 85/100 quality score
python batch_generate.py [# + description] --min-score 85
```

### Fast Mode (Skip BFS)

```bash
# Skip pathfinding checks
python batch_generate.py --fast
```

This trades validation thoroughness for speed—useful for rapid prototyping.

### Analyze Existing Scenarios

```bash
# Check quality of a scenario
python analyze_scenario.py scenarios/generated/scenario_abc123.json --verbose

# Visualize without simulation
python visualize_scenario.py scenarios/generated/scenario_abc123.json
```

### Run Simulation

```bash
# Run and visualize
python run_scenario.py scenarios/generated/scenario_abc123.json --visualize

# This will:
# 1. Load and repair the scenario
# 2. Run Social Force Model simulation
# 3. Generate animated GIF
# 4. Display metrics (time-to-goal, path efficiency, collisions, etc.)
```

## Writing Better Prompts

### Good Prompt Structure For Individual Scenario Generation

```
[Environment description] + [Specific geometry] + [Agent behaviors]
```

### Examples

**Good:**
```
"Create a straight corridor from x=-10 to x=10 with walls at y=-2 and y=2. 
Robot starts at (-8, 0) heading to (8, 0). 
Two humans at (6, -1) and (6, 1) walking toward (-6, -1) and (-6, 1), 
creating a head-on passing scenario."
```

**Also Good (less specific):**
```
"Robot navigating an L-shaped corridor while encountering humans 
coming from the perpendicular direction at the corner"
```

**Poor:**
```
"Robot in hallway with people"
```
(Too vague—no geometry, positions, or interaction specified)

See `example_scenario_prompts.md` for detailed prompt engineering guide.

## Common Issues

### Fully Enclosed Rooms

**Problem**: LLMs sometimes create rooms with 4 walls and no doorway, trapping agents.

**Detection**: Validator checks for closed rectangular regions and issues critical warning.

**Fix**: The prompt explicitly instructs the LLM to add doorways. If this fails, manual editing required.

### Unreachable Goals

**Problem**: Obstacles block paths from start to goal.

**Detection**: BFS pathfinding in validator.

**Fix**: Iterative refinement adds feedback: "Agent X cannot reach goal—add doorway or adjust walls."

### Agents Too Close to Walls

**Problem**: Agents placed <0.5m from obstacles.

**Fix**: Automatic geometric repair nudges agents away from walls.

## Simulation Metrics

After running a scenario, metrics include:

### Robot-Specific
- **Time to goal**: Steps taken
- **Collision**: Boolean
- **Path efficiency**: Actual path / straight-line distance
- **Average speed**: Mean velocity
- **Min human distance**: Closest approach to any human

### Per-Agent
- **Average acceleration**: Smoothness of motion
- **Path efficiency**: Deviation from optimal path

### Global
- **Min inter-agent distance**: Closest any two agents came

## Model Comparison

| Model | Quality | Speed | Cost | Use Case |
|-------|---------|-------|------|----------|
| **gpt-5** | ⭐⭐⭐⭐⭐ | Slow | High | Production datasets |
| **gpt-4.1** | ⭐⭐⭐⭐ | Medium | Medium | Default choice |
| **gpt-4.1-mini** | ⭐⭐⭐ | Fast | Low | Rapid prototyping |
| **gpt-4.1-nano** | ⭐⭐ | Very Fast | Very Low | Testing |


## Troubleshooting

**"MissingCredentialError"**: Set `OPENAI_API_KEY` environment variable

**Low quality scores**: Try `--model gpt-5` or increase `--max-iterations`

**Slow generation**: Use `--fast` mode or `--model gpt-4.1-mini`

**Visualization fails**: Ensure matplotlib backend is configured (try `export MPLBACKEND=TkAgg`)


```
