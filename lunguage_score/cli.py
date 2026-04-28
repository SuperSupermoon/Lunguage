"""
Command Line Interface for LunguageScore
"""

import argparse
import sys
import os
from pathlib import Path
from .scorer import LunguageScorer
from .config import Config, load_config


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="LunguageScore - Medical Report Evaluation Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with default settings (assumes structured reports)
  lunguage-score evaluate predicted.csv ground_truth.csv
  
  # Evaluate with custom config
  lunguage-score evaluate predicted.csv ground_truth.csv --config config.yaml
  
  # Evaluate with structuring (optional)
  lunguage-score evaluate predicted.csv ground_truth.csv --structure-reports
  
  # Structure reports only
  lunguage-score structure reports.csv --output structured.csv
  
  # Calculate LunguageScore only
  lunguage-score lunguage-score predicted_structured.csv ground_truth_structured.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate reports')
    eval_parser.add_argument('predicted', help='Path to predicted reports file')
    eval_parser.add_argument('ground_truth', help='Path to ground truth reports file')
    eval_parser.add_argument('--config', help='Path to configuration file')
    eval_parser.add_argument('--structure-reports', action='store_true', 
                           help='Structure raw reports first (optional)')
    eval_parser.add_argument('--report-column', default='report', 
                           help='Column name containing reports')
    eval_parser.add_argument('--study-id-column', default='study_id', 
                           help='Column name for study IDs')
    eval_parser.add_argument('--save-structured', action='store_true',
                           help='Save structured reports')
    eval_parser.add_argument('--output-dir', help='Output directory for results')
    
    # Structure command
    structure_parser = subparsers.add_parser('structure', help='Structure reports only')
    structure_parser.add_argument('input', help='Path to input reports file')
    structure_parser.add_argument('--output', required=True, help='Output file path')
    structure_parser.add_argument('--config', help='Path to configuration file')
    structure_parser.add_argument('--report-column', default='report', 
                                help='Column name containing reports')
    
    # LunguageScore command
    lunguage_parser = subparsers.add_parser('lunguage-score', help='Calculate LunguageScore only')
    lunguage_parser.add_argument('predicted', help='Path to structured predicted reports')
    lunguage_parser.add_argument('ground_truth', help='Path to structured ground truth reports')
    lunguage_parser.add_argument('--config', help='Path to configuration file')
    lunguage_parser.add_argument('--study-id-column', default='study_id', 
                               help='Column name for study IDs')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Generate default configuration')
    config_parser.add_argument('--output', default='lunguage_config.yaml',
                              help='Output configuration file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'evaluate':
            run_evaluate(args)
        elif args.command == 'structure':
            run_structure(args)
        elif args.command == 'lunguage-score':
            run_lunguage_score(args)
        elif args.command == 'config':
            run_config(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def run_evaluate(args):
    """Run evaluation command"""
    import pandas as pd

    # Load configuration
    config = load_config(args.config)

    # Update output directory if specified
    if args.output_dir:
        config.metrics.output_dir = args.output_dir

    # Initialize scorer
    scorer = LunguageScorer(config)

    # Load reports from file paths
    predicted_reports = pd.read_csv(args.predicted)
    ground_truth_reports = pd.read_csv(args.ground_truth)

    # Run evaluation
    results = scorer.evaluate(
        predicted_reports=predicted_reports,
        ground_truth_reports=ground_truth_reports,
        structure_reports=args.structure_reports,
        report_column=args.report_column,
        study_id_column=args.study_id_column,
        save_structured=args.save_structured
    )

    print("Evaluation completed successfully!")


def run_structure(args):
    """Run structure command"""
    import pandas as pd

    # Load configuration
    config = load_config(args.config)

    # Initialize scorer
    scorer = LunguageScorer(config)

    # Load reports from file path
    reports = pd.read_csv(args.input)

    # Structure reports
    structured_reports = scorer.structure_only(
        reports=reports,
        report_column=args.report_column,
        save_path=args.output
    )

    print(f"Reports structured and saved to {args.output}")


def run_lunguage_score(args):
    """Run LunguageScore calculation command"""
    import pandas as pd

    # Load configuration
    config = load_config(args.config)

    # Initialize scorer
    scorer = LunguageScorer(config)

    # Load structured reports from file paths
    predicted_structured = pd.read_csv(args.predicted)
    ground_truth_structured = pd.read_csv(args.ground_truth)

    # Calculate LunguageScore
    results = scorer.calculate_lunguage_score_only(
        predicted_structured=predicted_structured,
        ground_truth_structured=ground_truth_structured,
        study_id_column=args.study_id_column
    )

    print("LunguageScore calculation completed successfully!")


def run_config(args):
    """Run config command"""
    config = Config()
    config.to_yaml(args.output)
    print(f"Default configuration saved to {args.output}")


if __name__ == '__main__':
    main()
