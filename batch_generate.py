#!/usr/bin/env python3
"""
Batch scenario generator with interactive prompt.
Generates multiple scenarios based on a single user description.
"""
import argparse
import time
from pathlib import Path
from types import SimpleNamespace
import uuid
import os

from scenario_generator import generate_scenario, save_json
from scenario_io import load_scenario, dict_to_scenario
from scenario_validation import repair_scenario, analyze_quality
from scenario_adapter import scenario_to_simulator

if "OPENAI_API_KEY" not in os.environ:
    raise ValueError(
        "OPENAI_API_KEY environment variable must be set. "
        "Export it in your shell: export OPENAI_API_KEY='your-key-here'"
    )

DEFAULTS = {
    "provider": "openai",
    "model": "gpt-5",
    "temperature": 0.0,
    "max_tokens": 32000,
    "no_repair": False,
    "map_type": "auto",
    "seed": 42,
    "fast_mode": False, 
}


def generate_with_refinement(
    prompt_text: str,
    args,
    min_score: float = 75.0,
    max_iterations: int = 3,
    verbose: bool = False,
    fast_mode: bool = False 
):
    """Generate scenario with automatic iterative refinement."""
    best_scenario = None
    best_analysis = None
    best_score = 0
    previous_json = None
    
    # Build refinement prompt (simplified from generate.py)
    def build_refinement_prompt(original_prompt, analysis, iteration, previous_json=None):
        top_warnings = analysis.warnings[:5]
        warnings_text = "\n".join(f"  - {w}" for w in top_warnings)
        
        feedback = f"""
ITERATION {iteration} - REFINEMENT NEEDED

Score: {analysis.overall_score:.1f}/100 (target: {min_score}+)

CRITICAL ISSUES:
{warnings_text}

ORIGINAL REQUEST: {original_prompt}

Generate CORRECTED JSON (raw JSON only, no markdown):
"""
        return feedback
    
    current_prompt = prompt_text
    
    for iteration in range(1, max_iterations + 1):
        if verbose:
            print(f"  Iteration {iteration}/{max_iterations}", end="", flush=True)
        
        try:
            raw_json, repaired_json, logs = generate_scenario(current_prompt, args)
            previous_json = repaired_json
        except Exception as e:
            if verbose:
                print(f" - Failed: {e}")
            if iteration == max_iterations and best_scenario:
                break
            continue
        
        scenario = dict_to_scenario(repaired_json)
        # NEW: Pass fast_mode to analyze_quality
        analysis = analyze_quality(scenario, fast_mode=fast_mode)
        score = analysis.overall_score
        
        if verbose:
            print(f" - Score: {score:.1f}/100")
        
        if score > best_score:
            best_score = score
            best_scenario = repaired_json
            best_analysis = analysis
        
        if score >= min_score:
            break
        
        if iteration >= 2 and score <= best_score - 5:
            break
        
        if iteration < max_iterations:
            current_prompt = build_refinement_prompt(
                prompt_text, 
                analysis, 
                iteration + 1,
                previous_json=previous_json
            )
    
    return best_scenario, best_analysis, iteration


def generate_scenario_variations(base_prompt: str, count: int) -> list:
    """
    Simply repeat the base prompt - let the LLM's randomness create variations.
    Each call will have a different seed, resulting in natural diversity.
    """
    return [base_prompt] * count


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate multiple scenarios from a single description.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python batch_generate.py
  
  Then enter when prompted:
  > 50 scenarios of robot navigating through various environments in corporate settings
        """
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset3",
        help="Output directory for scenarios (default: dataset)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o4-mini", "gpt-5"],
        default="gpt-5",
        help="OpenAI model to use (default: gpt-5)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=75.0,
        help="Minimum quality score to accept (default: 75)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Maximum refinement attempts per scenario (default: 3)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed progress for each scenario",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: skip expensive BFS reachability checks (faster but less thorough validation)",
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BATCH SCENARIO GENERATOR")
    print("=" * 70)
    print()
    print("Enter your request")
    print("various environments in corporate settings'):")
    print()
    
    user_input = input("> ").strip()
    
    if not user_input:
        print("Error: No input provided.")
        return

    parts = user_input.split(maxsplit=1)
    try:
        count = int(parts[0])
        description = parts[1] if len(parts) > 1 else "robot navigating through various environments"
    except ValueError:
        print("Error: Request should start with a number (e.g., '50 scenarios of...')")
        return
    
    print()
    print(f"Generating {count} scenarios: {description}")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target quality: {args.min_score}/100")
    if args.fast:
        print("Mode: FAST (skipping BFS reachability checks)")
    else:
        print("Mode: FULL (with BFS reachability checks)")
    print()
    
    confirm = input("Proceed? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Cancelled.")
        return
    
    print()
    print("=" * 70)
    print("GENERATING SCENARIOS")
    print("=" * 70)
    print()
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = generate_scenario_variations(description, count)

    defaults_copy = DEFAULTS.copy()
    defaults_copy['output_dir'] = args.output_dir
    defaults_copy['model'] = args.model
    defaults_copy['fast_mode'] = args.fast  # NEW: pass fast mode flag

    start_time = time.time()
    successful = 0
    failed = 0
    total_score = 0
    scores = []

    for i, prompt_text in enumerate(prompts, 1):
        scenario_id = f"scenario_{uuid.uuid4().hex[:8]}"
        
        print(f"[{i}/{count}] Generating {scenario_id}...")
        if args.verbose:
            print(f"  Prompt: {prompt_text[:80]}...")
        
        defaults_copy['seed'] = 42 + i  # Different seed for each scenario
        scenario_args = SimpleNamespace(
            **defaults_copy,
            scenario_id=scenario_id,
            model_name=args.model,
        )
        
        try:
            # Generate
            best_scenario, best_analysis, iterations = generate_with_refinement(
                prompt_text,
                scenario_args,
                min_score=args.min_score,
                max_iterations=args.max_iterations,
                verbose=args.verbose,
                fast_mode=args.fast  # NEW: pass fast mode
            )
            
            if best_scenario is None:
                print(f"  ❌ Failed to generate valid scenario")
                failed += 1
                continue
            
            # Save
            out_path = out_dir / f"{scenario_id}.json"
            save_json(best_scenario, out_path)
            
            score = best_analysis.overall_score
            total_score += score
            scores.append(score)
            successful += 1
            
            print(f"  ✅ Saved with score {score:.1f}/100 (iterations: {iterations})")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Final statistics
    elapsed = time.time() - start_time
    
    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print()
    print(f"⏱️  Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"📊 Success rate: {successful}/{count} ({100*successful/count:.1f}%)")
    
    if successful > 0:
        avg_score = total_score / successful
        min_score = min(scores)
        max_score = max(scores)
        print(f"📈 Quality scores:")
        print(f"   - Average: {avg_score:.1f}/100")
        print(f"   - Range: {min_score:.1f} - {max_score:.1f}")
        print(f"   - Time per scenario: {elapsed/successful:.1f}s")
    
    if failed > 0:
        print(f"⚠️  Failed: {failed}")
    
    print()
    print(f"💾 Output directory: {out_dir.absolute()}")
    print()
    print("Next steps:")
    print(f"  • Review scenarios: ls {out_dir}")
    print(f"  • Analyze quality: python analyze_scenario.py {out_dir}/scenario_*.json")
    print(f"  • Run simulation: python run_scenario.py {out_dir}/scenario_XXXXX.json --visualize")
    print()


if __name__ == "__main__":
    main()