#!/usr/bin/env python3
"""
Simple tool to analyze the quality of a scenario JSON file.
"""
import argparse
from scenario_io import load_scenario
from scenario_validation import analyze_quality, print_analysis


def main():
    parser = argparse.ArgumentParser(
        description="Analyze scenario quality and report issues."
    )
    parser.add_argument(
        "scenario_file",
        type=str,
        help="Path to scenario JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed component scores",
    )
    args = parser.parse_args()
    
    # Load and analyze
    scenario = load_scenario(args.scenario_file)
    analysis = analyze_quality(scenario)
    
    # Print results
    if args.verbose:
        print(f"\n{'='*60}")
        print(f"Analyzing: {args.scenario_file}")
        print(f"{'='*60}\n")
        print_analysis(scenario)
    else:
        print(str(analysis))
    
    # Return exit code based on quality
    if analysis.overall_score >= 80:
        return 0  # Success
    elif analysis.overall_score >= 60:
        return 1  # Warning
    else:
        return 2  # Poor quality


if __name__ == "__main__":
    exit(main())