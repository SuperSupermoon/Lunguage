#!/usr/bin/env python3
"""
Metric runner script.
Run all metrics or a selected subset.
"""

import pandas as pd
import argparse
import os
import json
from pathlib import Path
from typing import List, Optional
import sys

from .sota_metrics import (
    calculate_rate_score,
    calculate_green_score,
    calculate_fineradscore,
    calculate_bleu,
    calculate_bertscore,
    calculate_radgraphF1
)


def load_data(input_path: str, filter_study_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load input data.

    Args:
        input_path: Path to CSV file. Required columns:
            - study_id: study ID
            - report_ref: reference (ground truth) report
            - report_cand: candidate (predicted) report
        filter_study_ids: list of study_ids to evaluate (None = use all)

    Returns:
        DataFrame with required columns
    """
    df = pd.read_csv(input_path)

    # Validate required columns
    required_cols = ["report_ref", "report_cand"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Auto-generate study_id if not present
    if "study_id" not in df.columns:
        df["study_id"] = range(len(df))
        print("'study_id' column not found — auto-generated.")

    # Apply STUDY_IDS filter (single-report setting: 67 studies)
    if filter_study_ids is not None:
        before = len(df)
        df = df[df["study_id"].isin(filter_study_ids)]
        after = len(df)
        print(f"STUDY_IDS filter: {before} -> {after} rows (unique study_ids: {df['study_id'].nunique()})")

    return df


def save_results(df: pd.DataFrame, output_path: str):
    """Save results to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Results saved: {output_path}")


def get_study_ids_from_notebook() -> Optional[List[str]]:
    """
    Return STUDY_IDS list for single-report setting (67 unique study_ids).
    """
    # Single-report setting: 67 unique study_ids
    STUDY_IDS = [
        's53356050', 's56140866', 's58307391', 's59166131', 's56078456',
        's59223989', 's50301215', 's51423353', 's52555178', 's53460154',
        's54849848', 's54962274', 's55957472', 's56034024', 's57211901',
        's58072789', 's50128467', 's53881360', 's59281953', 's59557609',
        's51235553', 's51301343', 's53311302', 's53499416', 's54607940',
        's55644325', 's57365217', 's57966185', 's58938414', 's58992648',
        's59200772', 's59409427', 's53118049', 's53579126', 's54552753',
        's55803143', 's56574351', 's57582717', 's50714348', 's51765753',
        's52616494', 's54692227', 's56093476', 's58215117', 's58897728',
        's53687124', 's54189324', 's56426152', 's57474951', 's58245185',
        's58728926', 's58847709', 's59083566', 's50139124', 's50683984',
        's51858688', 's52227426', 's54655227', 's54657781', 's54683624',
        's56171502', 's56238840', 's56374996', 's56618763', 's56778521',
        's56876464', 's58357438'
    ]
    
    print(f"Using STUDY_IDS filter: {len(STUDY_IDS)} unique study_ids")
    return STUDY_IDS


def run_metrics(
    input_path: str,
    output_path: str,
    metrics: List[str],
    output_dir: Optional[str] = None,
    skip_expensive: bool = False,
    use_study_ids_filter: bool = False
):
    """
    Run selected metrics.

    Args:
        input_path: Path to input CSV file
        output_path: Path to output CSV file
        metrics: List of metrics to run (use ['all'] for all available)
        output_dir: Directory for intermediate results (RaTEScore, FineRadScore)
        skip_expensive: Skip expensive metrics (GPT-4 based)
        use_study_ids_filter: Whether to apply STUDY_IDS filter (single-report setting: 67 studies)
    """
    # Get STUDY_IDS filter
    study_ids_filter = None
    if use_study_ids_filter:
        study_ids_filter = get_study_ids_from_notebook()

    # Load data
    print(f"Loading data: {input_path}")
    df = load_data(input_path, filter_study_ids=study_ids_filter)
    print(f"Loaded: {len(df)} rows (unique study_ids: {df['study_id'].nunique()})")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Check available metrics
    from .sota_metrics import (
        RATESCORE_AVAILABLE,
        GREEN_AVAILABLE,
        FINERADSCORE_AVAILABLE,
        EVALUATE_AVAILABLE,
        RADGRAPH_AVAILABLE
    )

    available_metrics = {}
    if RATESCORE_AVAILABLE:
        available_metrics["ratescore"] = calculate_rate_score
    if GREEN_AVAILABLE:
        available_metrics["green"] = calculate_green_score
    if FINERADSCORE_AVAILABLE:
        available_metrics["fineradscore"] = calculate_fineradscore
    if EVALUATE_AVAILABLE:
        available_metrics["bleu"] = calculate_bleu
        available_metrics["bertscore"] = calculate_bertscore
    if RADGRAPH_AVAILABLE:
        available_metrics["radgraph"] = calculate_radgraphF1

    print(f"\nAvailable metrics: {list(available_metrics.keys())}")

    # 'all' → expand to all available
    if "all" in metrics:
        metrics = list(available_metrics.keys())
        print(f"'all' selected. Metrics to run: {metrics}")

    # Filter out expensive metrics if requested
    expensive_metrics = ["fineradscore"]
    if skip_expensive:
        metrics = [m for m in metrics if m not in expensive_metrics]
        print(f"Skipping expensive metrics: {expensive_metrics}")

    print(f"\nRunning metrics: {metrics}\n")

    # Run each metric
    for metric_name in metrics:
        if metric_name not in available_metrics:
            print(f"Warning: metric '{metric_name}' is not available (missing dependency).")
            continue

        try:
            print(f"\n{'='*60}")
            print(f"Running: {metric_name.upper()}")
            print(f"{'='*60}")

            metric_func = available_metrics[metric_name]

            if metric_name == "ratescore":
                ratescore_path = os.path.join(output_dir, "ratescore_results")
                os.makedirs(ratescore_path, exist_ok=True)
                df = metric_func(df, ratescore_path)

            elif metric_name == "fineradscore":
                fineradscore_path = os.path.join(output_dir, "fineradscore_results.jsonl")
                df, cost = metric_func(df, fineradscore_path)
                print(f"FineRadScore total cost: ${cost:.2f}")

            elif metric_name == "green":
                # GREEN returns a new DataFrame — merge back
                green_result = metric_func(df)
                green_result = green_result.rename(columns={"green_score": "green"})
                df = df.merge(green_result[["study_id", "green"]], on="study_id", how="left")

            else:
                df = metric_func(df)

            print(f"✓ {metric_name.upper()} complete")

        except Exception as e:
            print(f"✗ {metric_name.upper()} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    save_results(df, output_path)

    # Summary with bootstrapped CI
    print(f"\n{'='*60}")
    print("Metric summary:")
    print(f"{'='*60}")
    metric_cols = [col for col in df.columns if col not in ["study_id", "report_ref", "report_cand"]]

    import numpy as np
    n_bootstrap = 1000
    np.random.seed(42)

    for col in metric_cols:
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            values = df[col].dropna()
            if len(values) == 0:
                print(f"  {col}: mean = NaN (no data)")
                continue

            mean_val = values.mean()

            # Bootstrap 95% CI
            boot_means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(values, size=len(values), replace=True)
                boot_means.append(sample.mean())

            boot_means = np.array(boot_means)
            ci_lower = np.percentile(boot_means, 2.5)
            ci_upper = np.percentile(boot_means, 97.5)
            std_val = values.std()
            se_val = std_val / np.sqrt(len(values))

            print(f"  {col}: mean = {mean_val:.4f} ± {se_val:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation metrics on radiology reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all metrics
  python run_all_metrics.py -i input.csv -o output.csv --metrics all

  # Run specific metrics only
  python run_all_metrics.py -i input.csv -o output.csv --metrics bleu bertscore radgraph

  # Skip expensive (GPT-4 based) metrics
  python run_all_metrics.py -i input.csv -o output.csv --metrics all --skip-expensive

Available metrics:
  - ratescore:    RaTEScore
  - green:        GREEN Score
  - fineradscore: FineRadScore (GPT-4 based, incurs API cost)
  - bleu:         BLEU Score
  - bertscore:    BERTScore
  - radgraph:     RadGraph F1 (radgraph and radgraph-xl)
        """
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input CSV file path (requires report_ref and report_cand columns)"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output CSV file path"
    )

    parser.add_argument(
        "-m", "--metrics",
        nargs="+",
        default=["all"],
        help="Metrics to run (default: all). Choices: ratescore, green, fineradscore, bleu, bertscore, radgraph"
    )

    parser.add_argument(
        "--output-dir",
        help="Directory for intermediate results (default: same directory as output file)"
    )

    parser.add_argument(
        "--skip-expensive",
        action="store_true",
        help="Skip expensive (GPT-4 based) metrics"
    )

    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Evaluate all data without STUDY_IDS filter (default: use 67-study filter)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)
    
    # Run metrics
    run_metrics(
        input_path=args.input,
        output_path=args.output,
        metrics=args.metrics,
        output_dir=args.output_dir,
        skip_expensive=args.skip_expensive,
        use_study_ids_filter=not args.no_filter
    )


if __name__ == "__main__":
    main()

