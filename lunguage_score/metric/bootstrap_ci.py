#!/usr/bin/env python3
"""
Bootstrap confidence interval calculation functions and CLI script.
"""

import numpy as np
import pandas as pd
import argparse
import sys
import os
from typing import Union, Tuple


def bootstrap_ci(
    values: Union[pd.Series, np.ndarray, list],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float, float]:
    """
    Compute bootstrap confidence interval for the mean.

    Args:
        values: Array of values (Series, ndarray, or list)
        n_bootstrap: Number of bootstrap iterations (default: 1000)
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        random_seed: Random seed (default: 42)

    Returns:
        Tuple (mean, se, ci_lower, ci_upper)
            - mean: mean value
            - se: standard error
            - ci_lower: lower confidence bound
            - ci_upper: upper confidence bound
    """
    # Convert pandas Series or list to numpy array
    if isinstance(values, pd.Series):
        values = values.values

    values = np.array(values)

    # Remove NaN values
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return np.nan, np.nan, np.nan, np.nan

    # Compute mean
    mean_val = np.mean(values)

    # Compute standard error
    std_val = np.std(values, ddof=1)  # ddof=1 for sample standard deviation
    se_val = std_val / np.sqrt(len(values))

    # Compute bootstrap 95% CI
    np.random.seed(random_seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.array(boot_means)

    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(boot_means, 100 * (alpha / 2))
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))

    return mean_val, se_val, ci_lower, ci_upper


def bootstrap_ci_for_dataframe(
    df: pd.DataFrame,
    metric_cols: list = None,
    exclude_cols: list = None,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    print_results: bool = True
) -> pd.DataFrame:
    """
    Compute bootstrap CI for multiple columns in a DataFrame.

    Args:
        df: Input DataFrame
        metric_cols: Columns to compute CI for (None = auto-select)
        exclude_cols: Columns to exclude (default: ['study_id', 'report_ref', 'report_cand'])
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level
        random_seed: Random seed
        print_results: Whether to print results

    Returns:
        DataFrame with columns: metric, mean, se, ci_lower, ci_upper
    """
    if exclude_cols is None:
        exclude_cols = ["study_id", "report_ref", "report_cand"]

    if metric_cols is None:
        metric_cols = [col for col in df.columns if col not in exclude_cols]

    results = []

    for col in metric_cols:
        if col not in df.columns:
            continue

        # Skip non-numeric columns
        if df[col].dtype not in ['float64', 'float32', 'int64', 'int32']:
            continue

        values = df[col].dropna()
        if len(values) == 0:
            if print_results:
                print(f"  {col}: mean = NaN (no data)")
            continue

        mean_val, se_val, ci_lower, ci_upper = bootstrap_ci(
            values, n_bootstrap, confidence_level, random_seed
        )

        results.append({
            'metric': col,
            'mean': mean_val,
            'se': se_val,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })

        if print_results:
            print(f"  {col}: mean = {mean_val:.4f} ± {se_val:.4f} (95% CI: [{ci_lower:.4f}, {ci_upper:.4f}])")

    return pd.DataFrame(results)


def main():
    """Apply bootstrap CI to metric columns in a CSV file."""
    parser = argparse.ArgumentParser(
        description="Compute bootstrap confidence intervals for metric columns in a CSV file."
    )

    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input CSV file path"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output CSV file path (optional)"
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations (default: 1000)"
    )

    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level (default: 0.95 for 95%% CI)"
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    parser.add_argument(
        "--exclude-cols",
        nargs="+",
        default=["subject_id", "study_id"],
        help="Columns to exclude (default: subject_id study_id)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading data: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded: {len(df)} rows")

    print(f"\n{'='*60}")
    print("Computing bootstrap CI...")
    print(f"{'='*60}")
    print(f"Bootstrap iterations: {args.n_bootstrap}")
    print(f"Confidence level: {args.confidence_level*100:.1f}%")
    print(f"Random seed: {args.random_seed}")
    print(f"Excluded columns: {args.exclude_cols}")
    print(f"{'='*60}\n")

    results_df = bootstrap_ci_for_dataframe(
        df=df,
        metric_cols=None,  # auto-select
        exclude_cols=args.exclude_cols,
        n_bootstrap=args.n_bootstrap,
        confidence_level=args.confidence_level,
        random_seed=args.random_seed,
        print_results=True
    )

    print(f"{'='*60}\n")

    if args.output:
        results_df.to_csv(args.output, index=False)
        print(f"Results saved: {args.output}")
    else:
        print("\nResults (DataFrame):")
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
