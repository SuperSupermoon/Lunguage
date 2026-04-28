"""
Report structuring functionality for LunguageScore.

Dispatches to singleSR or sequentialSR subpackage based on config.mode.
"""

import logging
import os
import shutil
import argparse
import pandas as pd
from contextlib import contextmanager
from typing import Optional, Union
from .config import StructuringConfig, SingleSRConfig, SequentialSRConfig

logger = logging.getLogger(__name__)


@contextmanager
def _working_directory(path: Optional[str]):
    """Context manager: temporarily change CWD to path (if given), restore on exit."""
    if path is None:
        yield
        return
    original = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)



def _config_to_single_args(cfg: SingleSRConfig) -> argparse.Namespace:
    """Convert SingleSRConfig dataclass to argparse.Namespace expected by singleSR."""
    args = argparse.Namespace(
        deployment_name=cfg.deployment_name,
        api_key=cfg.api_key,
        port=cfg.port,
        mode=cfg.mode,
        candidate_type=cfg.candidate_type,
        unit=cfg.unit,
        n_retrieval=cfg.n_retrieval,
        output_format=cfg.output_format,
        multi=cfg.multi,
        dynamic_retrieval=cfg.dynamic_retrieval,
        diverse_retrieval=cfg.diverse_retrieval,
        fast_retrieval=cfg.fast_retrieval,
        candidate_usage=cfg.candidate_usage,
        candidate_discontinuous=cfg.candidate_discontinuous,
        jaccard=cfg.jaccard,
        toy_set=cfg.toy_set,
        run_model=cfg.run_model,
        gold_path=cfg.gold_path,
        vocab_path=cfg.vocab_path,
        output_dir=cfg.output_dir,
        entity_types=cfg.entity_types,
        relation_types=cfg.relation_types,
        attribute_types=cfg.attribute_types,
        dev_list=[],
        batch_itr=None,
        batch_file_size=2,
        context_width=4,
        # Report path placeholders (set per-call)
        rexval_report_path=None,
        report_col_name=['gt_report', 'radgraph', 'bertscore', 's_emb', 'bleu'],
        silver_eval_report_path=None,
        maira2_cascade_report_path=None,
        maira2_report_path=None,
        medversa_report_path=None,
        rgrg_report_path=None,
        cvt2distilgpt2_report_path=None,
        rexerr_report_path=None,
        lingshu_path=None,
        medgemma_path=None,
        libra_path=None,
        chexagent_path=None,
    )
    return args


def _config_to_sequential_args(cfg: SequentialSRConfig) -> argparse.Namespace:
    """Convert SequentialSRConfig dataclass to argparse.Namespace expected by sequentialSR."""
    args = argparse.Namespace(
        LLM_name=cfg.LLM_name,
        api_key=cfg.api_key,
        input_path=cfg.input_path,
        output_path=cfg.output_path,
        batch_path=cfg.batch_path,
        all_eval=cfg.all_eval,
        few_shot=cfg.few_shot,
        model_run=cfg.model_run,
        process_missing=cfg.process_missing,
        eval_section=cfg.eval_section,
        subset=cfg.subset,
        gold_path=cfg.gold_path,
    )
    return args


class ReportStructurer:
    """
    Dispatcher class for structuring raw medical reports into structured format.

    Delegates to singleSR or sequentialSR based on config.mode:
      - 'single'     → singleSR pipeline (single-visit report structuring)
      - 'sequential' → sequentialSR pipeline (multi-visit sequential structuring)
    """

    def __init__(self, config: StructuringConfig):
        self.config = config

    def run_single_sr(self, report_path: Optional[str] = None) -> None:
        """
        Run singleSR structuring pipeline.

        Args:
            report_path: Override path to the input report file.
                         If None, uses path configured in config.single.
        """
        cfg = self.config.single

        if cfg.work_dir is None:
            raise ValueError(
                "SingleSRConfig.work_dir must be set to your workspace directory "
                "(the directory containing dataset/, singleSR/, benchmark/ folders). "
                "Example: SingleSRConfig(work_dir='/path/to/your/workspace', ...)"
            )

        with _working_directory(cfg.work_dir):
            from .singleSR.utils import initialize_llm_client
            from .singleSR.runner import main as single_main

            args = _config_to_single_args(cfg)
            if report_path is not None:
                _mode_path_map = {
                    'rexval': 'rexval_report_path',
                    'silver_eval': 'silver_eval_report_path',
                    'maira': 'maira2_report_path',
                    'maira_cascade': 'maira2_cascade_report_path',
                    'medversa': 'medversa_report_path',
                    'rgrg': 'rgrg_report_path',
                    'cvt2distilgpt2': 'cvt2distilgpt2_report_path',
                    'rexerr': 'rexerr_report_path',
                    'libra': 'libra_path',
                    'chexagent': 'chexagent_path',
                }
                attr = _mode_path_map.get(cfg.mode)
                if attr:
                    setattr(args, attr, report_path)

            client, tokenizer = initialize_llm_client(
                args.deployment_name, args.api_key, port=cfg.port
            )
            single_main(args, client, tokenizer)

            # After singleSR, copy pred_SR_df.csv to {output_dir}/structured_output.csv
            # so the package interface can find it at a consistent path.
            if not args.dynamic_retrieval:
                if args.multi:
                    eval_path = f'./singleSR/eval/{args.mode}/M{args.n_retrieval}_{args.candidate_type}_{args.deployment_name}/{args.output_format}/{args.unit}/{args.candidate_usage}'
                else:
                    eval_path = f'./singleSR/eval/{args.mode}/{args.n_retrieval}_{args.candidate_type}_{args.deployment_name}/{args.output_format}/{args.unit}/{args.candidate_usage}'
            else:
                eval_path = f'./singleSR/eval/{args.mode}/dynamic_{args.candidate_type}_{args.deployment_name}/{args.output_format}/{args.unit}/{args.candidate_usage}'

            pred_sr_df_path = os.path.join(eval_path, 'pred_SR_df.csv')
            if os.path.exists(pred_sr_df_path):
                os.makedirs(args.output_dir, exist_ok=True)
                dest = os.path.join(args.output_dir, 'structured_output.csv')
                shutil.copy2(pred_sr_df_path, dest)

    def run_sequential_sr(self, input_df: Optional[pd.DataFrame] = None) -> str:
        """
        Run sequentialSR structuring pipeline.

        Args:
            input_df: Override input DataFrame.
                      If None, the runner loads from config.sequential.input_path.

        Returns:
            Actual output directory path where results are saved.
        """
        cfg = self.config.sequential

        if cfg.work_dir is None:
            raise ValueError(
                "SequentialSRConfig.work_dir must be set to your workspace directory "
                "(the directory containing dataset/, sequentialSR/ folders). "
                "Example: SequentialSRConfig(work_dir='/path/to/your/workspace', ...)"
            )

        with _working_directory(cfg.work_dir):
            from .sequentialSR.llm_utils import initialize_llm_client
            from .sequentialSR import runner as seq_runner

            args = _config_to_sequential_args(cfg)

            # Resolve output path (mirrors sequentialSR/run.py logic)
            base_out = cfg.output_path
            base_batch = cfg.batch_path
            shot = 'few_shot' if cfg.few_shot else 'zero_shot'
            dataset_name = os.path.basename(cfg.input_path).split('.')[0]
            if cfg.all_eval:
                args.output_path = f'{base_out}/{cfg.LLM_name}/all_eval/{shot}/{dataset_name}'
                args.batch_path = f'{base_batch}/{cfg.LLM_name}/all_eval/{shot}/{dataset_name}'
            else:
                n = len(cfg.subset)
                args.output_path = f'{base_out}/{cfg.LLM_name}/subset{n}/{shot}/{dataset_name}'
                args.batch_path = f'{base_batch}/{cfg.LLM_name}/subset{n}/{shot}/{dataset_name}'
            os.makedirs(args.output_path, exist_ok=True)

            client, tokenizer = initialize_llm_client(
                cfg.LLM_name, api_key=cfg.api_key, port=cfg.port
            )

            if input_df is not None:
                seq_runner.run_with_dataframe(args, client, input_df)
            else:
                seq_runner.run(args, client)

            return args.output_path

    def structure_reports(self,
                          reports: Union[str, pd.DataFrame],
                          report_column: str = "report") -> str:
        """
        Structure reports using the configured mode (single or sequential).

        Runs the LLM-based structuring pipeline and saves results to disk.
        Returns the output directory path where results are saved.

        Args:
            reports: Path string to report CSV, or DataFrame of reports.
            report_column: When reports is a DataFrame, only this column
                           (plus study_id if present) is passed to the pipeline.

        Returns:
            Output directory path where structured results are saved.
        """
        if self.config.mode == 'single':
            if isinstance(reports, str):
                path = reports
            else:
                import tempfile
                if report_column not in reports.columns:
                    raise ValueError(
                        f"Column '{report_column}' not found in DataFrame. "
                        f"Available columns: {list(reports.columns)}. "
                        "Pass raw text reports with a report text column, "
                        "or pass a file path string instead."
                    )
                # Extract only the relevant columns before writing to temp file
                cols = [report_column]
                if 'study_id' in reports.columns:
                    cols = ['study_id'] + cols
                subset = reports[cols]
                tmp_fd, tmp_path = tempfile.mkstemp(suffix='.csv')
                try:
                    os.close(tmp_fd)
                    subset.to_csv(tmp_path, index=False)
                    self.run_single_sr(report_path=tmp_path)
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                return self.config.single.output_dir
            self.run_single_sr(report_path=path)
            return self.config.single.output_dir

        elif self.config.mode == 'sequential':
            df = reports if isinstance(reports, pd.DataFrame) else None
            actual_output_path = self.run_sequential_sr(input_df=df)
            return actual_output_path or self.config.sequential.output_path

        else:
            raise ValueError(f"Unknown structuring mode: '{self.config.mode}'. Use 'single' or 'sequential'.")

    def save_structured_reports(self, structured_df: pd.DataFrame,
                                output_path: str) -> None:
        """Save structured reports to CSV file."""
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        structured_df.to_csv(output_path, index=False)
        logger.info("Structured reports saved to %s", output_path)

    def load_structured_reports(self, input_path: str) -> pd.DataFrame:
        """Load structured reports from CSV file."""
        return pd.read_csv(input_path)
