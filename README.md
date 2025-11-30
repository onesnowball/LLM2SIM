# ROB599 - LLM-Powered Crowd Simulation

Generate crowd navigation scenarios from natural language prompts using LLMs, then simulate them with a Social Force Model.

## Quick Start

```bash
# Set your Gemini API key (or it uses the hardcoded fallback)
export GEMINI_API_KEY="your-key-here"

# Generate and simulate a scenario
python scenario_generator.py --prompt "A robot navigating a corridor with 3 humans" --provider gemini --simulate
```

## Pipeline

```
Text Prompt → LLM (Gemini) → JSON Scenario → Validator/Repair → Simulator → Metrics
```

## Files

| File | Description |
|------|-------------|
| `scenario_generator.py` | CLI to generate scenarios from text prompts |
| `llm_client.py` | Gemini/Llama API wrapper with retry logic |
| `prompt_builder.py` | Builds schema-aware prompts for the LLM |
| `scenario_io.py` | Load/parse scenario JSON files |
| `scenario_validator.py` | Validate and auto-repair scenarios |
| `scenario_adapter.py` | Convert scenarios to simulator objects |
| `simulator.py` | Social Force Model crowd simulator |
| `run_scenario.py` | Run simulation on existing JSON files |

## Usage

### Generate from prompt
```bash
python scenario_generator.py --prompt "A robot in a plaza with 5 humans" --provider gemini
```

### Generate and simulate
```bash
python scenario_generator.py --prompt "..." --provider gemini --simulate
```

### Run existing scenario
```bash
python run_scenario.py scenarios/handwritten/corridor_001.json --visualize
```

## Requirements

```bash
pip install -r requirements.txt
```

- `huggingface-hub` (for Llama provider)
- `gymnasium` (for simulator)
- `numpy`, `matplotlib` (for simulation/visualization)

