"""
Basic usage: calculate LunguageScore on already-structured reports (Stage 3 only).

NO LLM REQUIRED — this example works on pre-structured DataFrames.
This is the fastest way to verify your installation.

NOTE: On first run, the semantic model (FremyCompany/BioLORD-2023, ~500 MB)
will be downloaded automatically from HuggingFace. This takes a few minutes.
Subsequent runs use the cached model and are much faster.

Run from any directory:
    python examples/basic_usage.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from lunguage_score import LunguageScorer
from lunguage_score.config import Config, MetricConfig


def main():
    # Sample structured data (Stage 1 / Stage 2 output format)
    predicted_df = pd.DataFrame({
        'study_id': ['s001', 's001', 's002'],
        'entity':   ['pneumonia', 'pleural effusion', 'cardiomegaly'],
        'dx_status':    ['positive', 'negative', 'positive'],
        'dx_certainty': ['definitive', 'definitive', 'definitive'],
        'location':  ['right lower lobe', None, None],
        'severity':  ['mild', None, 'moderate'],
    })

    ground_truth_df = pd.DataFrame({
        'study_id': ['s001', 's001', 's002'],
        'entity':   ['pneumonia', 'pleural effusion', 'cardiomegaly'],
        'dx_status':    ['positive', 'negative', 'positive'],
        'dx_certainty': ['definitive', 'definitive', 'definitive'],
        'location':  ['right lower lobe', None, None],
        'severity':  ['moderate', None, 'moderate'],
    })

    config = Config(metrics=MetricConfig(output_dir='/tmp/lunguage_basic_results'))
    scorer = LunguageScorer(config)

    results = scorer.calculate_lunguage_score_only(predicted_df, ground_truth_df)

    print(f"\nStructure F1 : {results['avg_structure_score']:.4f}")
    print(f"Precision    : {results['avg_precision']:.4f}")
    print(f"Recall       : {results['avg_recall']:.4f}")


if __name__ == "__main__":
    main()
