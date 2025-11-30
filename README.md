# ROB599 - LLM-Powered Crowd Simulation Scenario Generator

Generate crowd navigation scenarios from natural language prompts using LLMs, then simulate robot behavior using Social Force Models.

## Overview

This project provides an end-to-end pipeline:
1. **Natural language prompt** → LLM generates scenario JSON
2. **Validation & repair** → Auto-fix common issues
3. **Simulation** → Run robot navigation with Social Force Model
4. **Metrics** → Collision detection, path efficiency, timing

## Quick Start

```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-key-here"

# Generate and simulate a scenario
python scenario_generator.py \
  --prompt "A robot navigating through a busy plaza with 5 humans" \
  --provider gemini \
  --simulate
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Generate Scenario Only
```bash
python scenario_generator.py --prompt "A robot in a corridor with 2 humans approaching" --provider gemini
```

### Generate + Simulate
```bash
python scenario_generator.py --prompt "..." --provider gemini --simulate
```

### Run Existing Scenario
```bash
python run_scenario.py scenarios/handwritten/corridor_001.json --visualize
```

### Batch Generation from Seed File
```bash
python scenario_generator.py --prompt-file seeds/prompts_seed.txt --provider gemini --count 10
```

## Project Structure

```
├── scenario_generator.py   # Main CLI: prompt → LLM → JSON → simulate
├── llm_client.py           # Gemini/Llama API wrapper
├── prompt_builder.py       # Schema-aware prompt construction
├── scenario_io.py          # JSON parsing/serialization
├── scenario_validator.py   # Auto-repair and validation
├── scenario_adapter.py     # Convert scenarios to simulator
├── simulator.py            # CrowdSimulator with Social Force Model
├── run_scenario.py         # Run existing JSON scenarios
├── scenarios/
│   └── handwritten/        # Example scenarios
├── seeds/
│   └── prompts_seed.txt    # Natural language prompts
└── docs/
    └── scenario_schema.md  # JSON schema documentation
```

## API Keys

- **Gemini**: Get a free key from [Google AI Studio](https://aistudio.google.com/)
- Set via environment variable: `export GEMINI_API_KEY="..."`

## Output

Generated scenarios are saved to `scenarios/generated/`:
- `<id>.json` - Validated/repaired scenario
- `<id>_raw.json` - Raw LLM output

## Metrics

After simulation, you'll see:
- `time_to_goal`: Steps to reach destination
- `collision`: Whether robot collided
- `path_efficiency`: Actual vs straight-line distance
- `avg_speed`: Mean robot velocity
- `min_human_dist`: Closest approach to any human

