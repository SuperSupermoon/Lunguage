<div align="center">
  <a href="https://supersupermoon.github.io/Lunguage/" target="_blank">
    <img alt="Project Page" src="https://img.shields.io/badge/🌐_Project-Page-blue" />
  </a>
  <a href="https://arxiv.org/abs/2505.21190" target="_blank">
    <img alt="Lunguage" src="https://img.shields.io/badge/📄_Lunguage-Paper-b31b1b.svg" />
  </a>
  <a href="https://arxiv.org/abs/2511.04506" target="_blank">
    <img alt="Lunguage++" src="https://img.shields.io/badge/📄_Lunguage%2B%2B-Paper-b31b1b.svg" />
  </a>
  <a href="https://pypi.org/project/lunguage-score/" target="_blank">
    <img alt="PyPI" src="https://img.shields.io/badge/📦_PyPI-lunguage--score-orange.svg" />
  </a>
  <a href="https://physionet.org/content/lunguage" target="_blank">
    <img alt="Lunguage Data Resource" src="https://img.shields.io/badge/💾_Lunguage-Data_Resource-blueviolet.svg" />
  </a>
</div>

<h1>
<p align="center">
  Lunguage: A Benchmark for Structured and Sequential Chest X-ray Interpretation
</p>
</h1>

<p align="center">
  <b>A Python package for structured chest X-ray report evaluation using LunguageScore — supporting single- and sequential-report assessment</b>
</p>

## 📰 News
- [2024/05] Lunguage benchmark paper posted on arXiv.
- [2024/11] Lunguage++ extended version released: [paper](https://arxiv.org/abs/2511.04506), [code](https://github.com/prabaey/lunguage_uncertainty).
- [2024/11] Lunguage dataset (v1.0.0) released on PhysioNet.
- [2025/04] Lunguage dataset (v1.1.0) and Lunguage++ (v1.0.0) datasets are under review and will be made available on PhysioNet.

## 📚 Introduction

**LunguageScore** evaluates chest X-ray report generation models using the [Lunguage benchmark](https://supersupermoon.github.io/Lunguage/). The pipeline has two stages:

| Stage | What it does | Requires |
|-------|--------------|----------|
| **Structuring** | Converts raw reports into SRO (Subject-Relation-Object) structures via LLM | LLM API key or local vLLM server |
| **Metric** | Computes LunguageScore (Precision / Recall / Structure Score) | Two structured CSVs: predicted + ground truth |

Structuring supports **Single SR** (single-visit) and **Sequential SR** (multi-visit temporal grouping).

> Metric computation requires structured reports for both predicted and ground-truth. Run structuring first if you don't have them.

## 📁 Dataset

The Lunguage dataset (v1.0.0) is hosted on [PhysioNet](https://physionet.org/content/lunguage/1.0.0/) and is available to **credentialed MIMIC-CXR users** under the PhysioNet Data Use Agreement.

> **Access:** https://physionet.org/content/lunguage/1.0.0/ (requires PhysioNet credentialed account with MIMIC-CXR access)

Place the dataset files in your workspace:

```
/your/workspace/
├── dataset/
│   ├── Lunguage.csv            # Gold structured annotations (required)
│   └── Lunguage_vocab.csv      # Entity vocabulary for retrieval (required)
├── benchmark/
│   └── my_model_results.csv    # Your model's predicted reports (see format below)
├── singleSR/                   # Created automatically during structuring
└── sequentialSR/               # Created automatically during sequential structuring
```

**Benchmark CSV format** (for evaluating your own model):

| Column | Description |
|--------|-------------|
| `subject_id` | Patient identifier (must match Lunguage.csv) |
| `study_id` | Study identifier (must match Lunguage.csv) |
| `report` | Full generated report text |

## Installation

> Package publication is in progress. For now, use source installation.

```bash
git clone https://github.com/supersupermoon/Lunguage.git
cd Lunguage/lunguage_score_package
pip install -e .
```

When PyPI publication is complete, `pip install lunguage-score` will be supported as the default installation path.

> **Metric stage** works immediately after install — no LLM needed.
> **Structuring stage** requires an LLM API key or a running vLLM server.
> On first use, the semantic model (`FremyCompany/BioLORD-2023`, ~500 MB) downloads automatically.

---

## Quick Start: `run_eval_pipeline.py`

`run_eval_pipeline.py` is the recommended way to run the full pipeline from the command line.

> **Current recommendation (pre-release):** run directly from this repository after source installation.

### `--mode` — input report source

| `--mode` | Use when | `--pred-path` needed? |
|----------|----------|----------------------|
| `gold_eval` | Evaluating against the Lunguage gold standard | No — uses `Lunguage.csv` directly |
| `chexagent` | Your own model's outputs (custom CSV) | **Yes** |
| `rexval` / `maira` / `maira_cascade` / `medversa` / `rgrg` / `medgemma` / `lingshu` | Specific benchmarks | Yes |

### `--stage` — pipeline stages

| Stage | Description |
|-------|-------------|
| `full-single` | Single SR structuring → LunguageScore metric |
| `full-sequential` | Single SR → Sequential SR → LunguageScore metric (default) |
| `structure-single` | Single SR structuring only |
| `structure-sequential` | Single SR → Sequential SR (no metric) |
| `metric-only` | Metric only (structured CSVs must already exist) |

### Supported LLM backends

| Backend | `--model` prefix | `--api-key` |
|---------|-----------------|-------------|
| Local vLLM | any | `local_LLM` (default) — requires `--port` and running server |
| OpenAI | `gpt-` | OpenAI key |
| Anthropic Claude | `claude-` | Anthropic key |
| Google MedGemma | `medgemma-` | Google AI Studio key, or `local_LLM` |
| Fireworks AI | `deepseek-`, `llama4-`, `qwen3-` | Fireworks key |

> **Cloud APIs (OpenAI, Claude) use batch processing** — no local GPU required.

### Examples

```bash
# Evaluate your model's reports using Claude (no GPU needed)
python run_eval_pipeline.py --stage full-single \
    --work-dir /your/workspace \
    --model claude-sonnet-4-6 --api-key sk-ant-... \
    --pred-path benchmark/my_model_results.csv \
    --mode chexagent

# Evaluate on Lunguage gold standard using local MedGemma
python run_eval_pipeline.py --stage full-single \
    --work-dir /your/workspace \
    --model medgemma-27b-text-it --api-key local_LLM \
    --port 8100 --gpu-ids 0,1 \
    --mode gold_eval

# Metric only — structured CSVs already exist
python run_eval_pipeline.py --stage metric-only \
    --work-dir /your/workspace \
    --pred-sr-path ./singleSR/eval/gold_eval/.../pred_SR_df.csv \
    --gt-sr-path   ./singleSR/eval/gold_eval/.../gold_SR_df.csv

# Full sequential pipeline
python run_eval_pipeline.py --stage full-sequential \
    --work-dir /your/workspace \
    --model medgemma-27b-text-it --api-key local_LLM \
    --port 8100 --gpu-ids 0,1 \
    --mode gold_eval --all-eval
```

> If you are validating setup before full data preparation, start with `--stage metric-only` and explicit `--pred-sr-path` / `--gt-sr-path`.

### Starting a local vLLM server

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve google/medgemma-27b-text-it \
    --served-model-name medgemma-27b-text-it \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --port 8100
```

> For reasoning models (Qwen3, DeepSeek) use `--max-model-len 32768`.

### Output

```
{work_dir}/
├── singleSR/eval/{mode}/{n}_{cand}_{model}/...
│   ├── pred_SR_df.csv      # Stage 1 output (structured predictions)
│   └── gold_SR_df.csv      # Matched gold annotations
├── sequentialSR/results/{model}/.../
│   └── final_processed_*.csv   # Stage 2 output (temporal grouping)
└── results/
    └── lunguage_score_results.json   # Final scores
```

---

<details>
<summary><b>Stage 1: Single SR — Report Structuring (Python API)</b></summary>

```python
from lunguage_score import LunguageScorer
from lunguage_score.config import Config, StructuringConfig, SingleSRConfig

single_cfg = SingleSRConfig(
    deployment_name="claude-sonnet-4-6",  # or "gpt-4.1", "medgemma-27b-text-it"
    api_key="sk-ant-...",                  # or "local_LLM" for vLLM
    port=8100,
    work_dir="/your/workspace",
    mode="chexagent",                      # "gold_eval" for Lunguage gold set
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
output_dir = scorer.structure_only(reports="./benchmark/my_model_results.csv")
# Output: {work_dir}/singleSR/eval/{mode}/.../pred_SR_df.csv
```

**Supported modes:**

| mode | Input source |
|------|-------------|
| `gold_eval` | `Lunguage.csv` directly (no custom CSV needed) |
| `chexagent` | Your model's CSV (`subject_id`, `study_id`, `report`) |
| `rexval` / `maira` / `maira_cascade` / `medversa` / `rgrg` / `cvt2distilgpt2` / `libra` | Benchmark-specific formats |

</details>

---

<details>
<summary><b>Stage 2: Sequential SR — Temporal Grouping (Python API)</b></summary>

```python
import pandas as pd
from lunguage_score import LunguageScorer
from lunguage_score.config import Config, StructuringConfig, SequentialSRConfig

pred_sr_path = "/your/workspace/singleSR/eval/gold_eval/.../pred_SR_df.csv"

seq_cfg = SequentialSRConfig(
    LLM_name="claude-sonnet-4-6",   # or "gpt-4.1", "medgemma-27b-text-it"
    api_key="sk-ant-...",            # or "local_LLM"
    port=8100,
    work_dir="/your/workspace",
    input_path=pred_sr_path,
    output_path="./sequentialSR/results",
    batch_path="./sequentialSR/batch_files",
    all_eval=True,
    few_shot=False,   # zero-shot recommended for models with ≤ 8192 token limit
    model_run=True,
    process_missing=True,
)

config = Config(structuring=StructuringConfig(mode="sequential", sequential=seq_cfg))
scorer = LunguageScorer(config)
output_dir = scorer.structure_only(reports=pd.read_csv(pred_sr_path), report_column="entity")
# Output: {work_dir}/sequentialSR/results/{model}/.../final_processed_*.csv
```

</details>

---

<details>
<summary><b>Stage 3: LunguageScore Metric (Python API)</b></summary>

No LLM required.

```python
import pandas as pd
from lunguage_score import LunguageScorer
from lunguage_score.config import Config, MetricConfig

pred_df = pd.read_csv("/path/to/pred_SR_df.csv")
gt_df   = pd.read_csv("/path/to/gold_SR_df.csv")

# Lunguage.csv uses 'ent'; structured output uses 'entity'
if 'ent' in gt_df.columns and 'entity' not in gt_df.columns:
    gt_df = gt_df.rename(columns={'ent': 'entity'})

config = Config(metrics=MetricConfig(output_dir="./results", mode="gold_eval"))
scorer = LunguageScorer(config)
results = scorer.calculate_lunguage_score_only(pred_df, gt_df)

print(f"Structure Score: {results['avg_structure_score']:.4f}")
print(f"Precision:       {results['avg_precision']:.4f}")
print(f"Recall:          {results['avg_recall']:.4f}")
```

**`MetricConfig.mode`** must match the structuring mode used in Stage 1:

| mode | Use when |
|------|----------|
| `gold_eval` | Stage 1 `mode="gold_eval"` |
| `chexagent` / `medversa` / `rgrg` / `rexval` | Matching Stage 1 mode |
| `single_maira` | Stage 1 `mode="maira"` or `"maira_cascade"` |
| `sequential_maira` | Stage 2 sequential output |

**Output JSON:**
```json
{
  "lunguage_score": {
    "avg_structure_score": 0.943,
    "avg_precision": 0.943,
    "avg_recall": 0.943,
    "structure_scores": {"s10000032": 0.95, ...},
    "precision_scores": {"s10000032": 0.95, ...},
    "recall_scores":    {"s10000032": 0.95, ...}
  }
}
```

</details>

---

<details>
<summary><b>SOTA Metrics (GREEN, RadGraph, BLEU, BERTScore, FineRadScore)</b></summary>

```python
from lunguage_score.metric.run_all_metrics import run_metrics

run_metrics(
    input_path='./results/paired_reports.csv',
    output_path='./results/sota_scores.csv',
    metrics=['green', 'radgraph', 'bleu', 'bertscore'],
    output_dir='./results/metric_artifacts',
)
```

Input CSV format:
```csv
study_id,report_ref,report_cand
s10000032,"The chest X-ray shows...","The radiograph demonstrates..."
```

> **Optional feature:** `pip install "lunguage-score[sota]"` for RaTEScore, BLEU, BERTScore.
> Some environments may still require additional dependency/version adjustments.

| Metric | Notes |
|--------|-------|
| GREEN | Install separately: `pip install git+https://github.com/Stanford-AIMI/GREEN.git` |
| RadGraph F1 | Included by default |
| BLEU / BERTScore | Install with `pip install "lunguage-score[sota]"` |
| RaTEScore | Install with `pip install "lunguage-score[sota]"`; may be unavailable on some numpy versions |
| FineRadScore | Requires OpenAI API key |

</details>

---

## `lunguage-score` CLI

```bash
lunguage-score evaluate predicted_SR_df.csv gold_SR_df.csv
lunguage-score evaluate predicted_SR_df.csv gold_SR_df.csv --output-dir ./my_results
```

---

## Citation

```bibtex
@article{moon2025lunguage,
  title={Lunguage: A Benchmark for Structured and Sequential Chest X-ray Interpretation},
  author={Moon, Jonghak and others},
  journal={arXiv preprint arXiv:2505.21190},
  year={2025}
}
```
