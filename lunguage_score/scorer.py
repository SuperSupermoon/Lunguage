"""
Main LunguageScorer class for comprehensive report evaluation
"""

import logging
import os

import pandas as pd
from typing import Dict, Any, List, Optional, Union
from .config import Config
from .structurer import ReportStructurer
from .metric_evaluator import LunguageMetricEvaluator

logger = logging.getLogger(__name__)


class LunguageScorer:
    """
    Main class for comprehensive medical report evaluation
    
    This class provides a unified interface for:
    1. Structuring raw medical reports (optional)
    2. Calculating LunguageScore evaluation metric
    3. Managing the entire evaluation workflow
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize LunguageScorer
        
        Args:
            config: Configuration object. If None, default config will be used.
        """
        if config is None:
            config = Config()
        
        self.config = config
        self.structurer = ReportStructurer(config.structuring)
        self.metric_evaluator = LunguageMetricEvaluator(config.metrics)
        
    def evaluate(self, 
                predicted_reports: Union[List[str], pd.DataFrame],
                ground_truth_reports: Union[List[str], pd.DataFrame],
                structure_reports: bool = False,  # Default to False since structuring is optional
                report_column: str = "report",
                study_id_column: str = "study_id",
                save_structured: bool = False,
                structured_output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Main evaluation method
        
        Args:
            predicted_reports: Predicted reports (raw text or structured)
            ground_truth_reports: Ground truth reports (raw text or structured)
            structure_reports: Whether to structure raw reports first (optional)
            report_column: Column name containing reports if DataFrame is provided
            study_id_column: Column name for study IDs
            save_structured: Whether to save structured reports
            structured_output_path: Path to save structured reports
            
        Returns:
            Dictionary containing LunguageScore evaluation results
        """
        logger.info("Starting LunguageScore evaluation...")

        # Step 1: Structure reports if needed
        if structure_reports:
            # Stage 1: run LLM-based structuring pipeline — results are saved to disk
            logger.info("Structuring predicted reports...")
            pred_output_dir = self.structurer.structure_reports(predicted_reports, report_column)

            logger.info("Structuring ground truth reports...")
            gt_output_dir = self.structurer.structure_reports(ground_truth_reports, report_column)

            # Stage 1 writes results to files; load them for Stage 2
            pred_csv = os.path.join(pred_output_dir, "structured_output.csv")
            gt_csv   = os.path.join(gt_output_dir,   "structured_output.csv")

            if not os.path.exists(pred_csv) or not os.path.exists(gt_csv):
                raise FileNotFoundError(
                    f"Stage 1 (structuring) output not found.\n"
                    f"Expected predicted:     {pred_csv}\n"
                    f"Expected ground truth:  {gt_csv}\n"
                    "Run structuring first and ensure 'structured_output.csv' "
                    "is present in the output directory, then call "
                    "evaluate() with structure_reports=False."
                )
            predicted_structured   = pd.read_csv(pred_csv)
            ground_truth_structured = pd.read_csv(gt_csv)

            if save_structured:
                out = structured_output_path or self.config.metrics.output_dir
                os.makedirs(out, exist_ok=True)
                predicted_structured.to_csv(
                    os.path.join(out, "predicted_structured.csv"), index=False)
                ground_truth_structured.to_csv(
                    os.path.join(out, "ground_truth_structured.csv"), index=False)
        else:
            # Use provided structured reports directly
            predicted_structured    = predicted_reports
            ground_truth_structured = ground_truth_reports
        
        # Step 2: Calculate LunguageScore
        logger.info("Calculating LunguageScore...")
        results = self.metric_evaluator.calculate_lunguage_score(
            predicted_structured, ground_truth_structured, study_id_column
        )
        
        # Step 3: Save results
        self.metric_evaluator.save_results()
        
        # Step 4: Print summary
        self.metric_evaluator.print_summary()
        
        return results
    
    def structure_only(self,
                      reports: Union[List[str], pd.DataFrame],
                      report_column: str = "report",
                      save_path: Optional[str] = None) -> str:
        """
        Run the structuring pipeline only, without calculating metrics.

        Args:
            reports: Path string to report CSV, or DataFrame of reports.
            report_column: Column containing raw report text (when DataFrame).
            save_path: If provided, copy the structured output CSV to this path.

        Returns:
            Output directory path where structuring results are saved.
        """
        logger.info("Structuring reports...")
        output_dir = self.structurer.structure_reports(reports, report_column)

        if save_path:
            src = os.path.join(output_dir, "structured_output.csv")
            if os.path.exists(src):
                pd.read_csv(src).to_csv(save_path, index=False)
                logger.info("Structured results copied to %s", save_path)
            else:
                logger.warning("structured_output.csv not found in %s", output_dir)

        return output_dir
    
    def calculate_lunguage_score_only(self,
                                    predicted_structured: pd.DataFrame,
                                    ground_truth_structured: pd.DataFrame,
                                    study_id_column: str = "study_id") -> Dict[str, Any]:
        """
        Calculate LunguageScore only (assumes reports are already structured)
        
        Args:
            predicted_structured: Structured predicted reports
            ground_truth_structured: Structured ground truth reports
            study_id_column: Column name for study IDs
            
        Returns:
            Dictionary containing LunguageScore results
        """
        logger.info("Calculating LunguageScore on structured reports...")
        results = self.metric_evaluator.calculate_lunguage_score(
            predicted_structured, ground_truth_structured, study_id_column
        )
        
        self.metric_evaluator.save_results()
        self.metric_evaluator.print_summary()
        
        return results
    
    def load_and_evaluate(self,
                         predicted_path: str,
                         ground_truth_path: str,
                         structure_reports: bool = False,
                         report_column: str = "report",
                         study_id_column: str = "study_id") -> Dict[str, Any]:
        """
        Load reports from files and evaluate
        
        Args:
            predicted_path: Path to predicted reports file
            ground_truth_path: Path to ground truth reports file
            structure_reports: Whether reports need structuring
            report_column: Column name containing reports
            study_id_column: Column name for study IDs
            
        Returns:
            Dictionary containing evaluation results
        """
        # Load reports
        predicted_reports = pd.read_csv(predicted_path)
        ground_truth_reports = pd.read_csv(ground_truth_path)
        
        return self.evaluate(
            predicted_reports, ground_truth_reports,
            structure_reports, report_column, study_id_column
        )
    
    def get_config(self) -> Config:
        """Get current configuration"""
        return self.config
    
    def update_config(self, config: Config) -> None:
        """Update configuration"""
        self.config = config
        self.structurer = ReportStructurer(config.structuring)
        self.metric_evaluator = LunguageMetricEvaluator(config.metrics)
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary of last evaluation results"""
        return self.metric_evaluator.get_summary()
