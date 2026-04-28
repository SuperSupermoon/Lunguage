"""
Advanced usage: full pipeline from raw reports → singleSR → sequentialSR → LunguageScore.

Prerequisites:
  1. A running vLLM server  (or OpenAI API key)
  2. A workspace directory with:
       {WORKSPACE}/dataset/Lunguage.csv
       {WORKSPACE}/dataset/Lunguage_vocab.csv
       {WORKSPACE}/benchmark/chexagent_results.csv   (your input reports)

Set WORKSPACE, LLM_NAME, API_KEY, PORT below, then run:
    python examples/advanced_usage.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import glob
import pandas as pd
from lunguage_score import LunguageScorer
from lunguage_score.config import (
    Config, MetricConfig, StructuringConfig,
    SingleSRConfig, SequentialSRConfig,
)

# ── EDIT THESE BEFORE RUNNING ────────────────────────────────────────────────
WORKSPACE = "/your/workspace"              # directory with dataset/, benchmark/
LLM_NAME  = "medgemma-27b-text-it"        # or "gpt-4.1"
API_KEY   = "local_LLM"                   # or "sk-..."  for OpenAI
PORT      = 8100                          # vLLM server port (local_LLM only)
GT_CSV    = "/your/ground_truth.csv"       # ground truth structured CSV
# ─────────────────────────────────────────────────────────────────────────────

# Validate that placeholders were replaced
_PLACEHOLDERS = {
    "WORKSPACE": WORKSPACE == "/your/workspace",
    "GT_CSV": GT_CSV == "/your/ground_truth.csv",
}
_unfilled = [k for k, v in _PLACEHOLDERS.items() if v]
if _unfilled:
    raise ValueError(
        f"Please set the following variables before running: {_unfilled}\n"
        "Edit the 'EDIT THESE BEFORE RUNNING' section at the top of this file."
    )


def run_stage1():
    """Stage 1 — singleSR: structure individual CXR reports."""
    single_cfg = SingleSRConfig(
        deployment_name=LLM_NAME,
        api_key=API_KEY,
        port=PORT,
        work_dir=WORKSPACE,
        mode="chexagent",
        candidate_type="vocab_ent_rcg",
        unit="section",
        n_retrieval=5,
        output_format="SROSRO",
        candidate_usage=1.0,
        run_model=True,
        output_dir="./singleSR/data",
    )
    config = Config(structuring=StructuringConfig(mode="single", single=single_cfg))
    scorer = LunguageScorer(config)

    output_dir = scorer.structure_only(
        reports="./benchmark/chexagent_results.csv",  # relative to WORKSPACE
    )
    print(f"Stage 1 output directory: {output_dir}")

    # Canonical path to pred_SR_df.csv produced by singleSR
    pred_sr_path = os.path.join(
        WORKSPACE,
        f"singleSR/eval/chexagent/5_vocab_ent_rcg_{LLM_NAME}/SROSRO/section/1/pred_SR_df.csv",
    )
    return pred_sr_path


def run_stage2(pred_sr_path: str):
    """Stage 2 — sequentialSR: temporal grouping across multi-visit series."""
    pred_sr_df = pd.read_csv(pred_sr_path)

    seq_cfg = SequentialSRConfig(
        LLM_name=LLM_NAME,
        api_key=API_KEY,
        port=PORT,
        work_dir=WORKSPACE,
        input_path=pred_sr_path,
        output_path="./sequentialSR/results",
        batch_path="./sequentialSR/batch_files",
        all_eval=True,
        few_shot=False,       # zero-shot; set True only with max-model-len >= 16384
        model_run=True,
        process_missing=True,
    )
    config = Config(structuring=StructuringConfig(mode="sequential", sequential=seq_cfg))
    scorer = LunguageScorer(config)

    output_dir = scorer.structure_only(reports=pred_sr_df, report_column="entity")
    print(f"Stage 2 output directory: {output_dir}")
    return output_dir


def run_stage3(stage2_dir: str):
    """Stage 3 — LunguageScore metric: semantic F1 against ground truth."""
    # Pick the most recent final_processed*.csv from stage 2 output
    csvs = sorted(glob.glob(os.path.join(WORKSPACE, stage2_dir, "final_processed*.csv")))
    if not csvs:
        raise FileNotFoundError(f"No final_processed*.csv found in {stage2_dir}")
    pred_df = pd.read_csv(csvs[-1])

    gt_df = pd.read_csv(GT_CSV)
    if 'ent' in gt_df.columns and 'entity' not in gt_df.columns:
        gt_df = gt_df.rename(columns={'ent': 'entity'})

    config = Config(metrics=MetricConfig(
        output_dir=os.path.join(WORKSPACE, "results"),
        mode="chexagent",
    ))
    scorer = LunguageScorer(config)
    results = scorer.calculate_lunguage_score_only(pred_df, gt_df)

    print(f"\n=== LunguageScore Results ===")
    print(f"Structure F1 : {results['avg_structure_score']:.4f}")
    print(f"Precision    : {results['avg_precision']:.4f}")
    print(f"Recall       : {results['avg_recall']:.4f}")
    return results


if __name__ == "__main__":
    pred_sr_path = run_stage1()
    stage2_dir   = run_stage2(pred_sr_path)
    run_stage3(stage2_dir)
