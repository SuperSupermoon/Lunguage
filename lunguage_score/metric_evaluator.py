"""
LunguageScore metric evaluation functionality
"""

import json
import logging
import os

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from .config import MetricConfig
from .lunguagescore import MetricEvaluator, DataConverter, _load_metric_weights

logger = logging.getLogger(__name__)


class LunguageMetricEvaluator:
    """
    Class for calculating LunguageScore on structured reports
    """
    
    def __init__(self, config: MetricConfig):
        self.config = config
        self.results = {}
        
        # Load metric weights
        metric_weights = _load_metric_weights()
        
        temporal_weight = sum(metric_weights['TEMPORAL_WEIGHTS'].values())
        temporal_weights = {k: v / temporal_weight for k, v in metric_weights['TEMPORAL_WEIGHTS'].items()}
        rel_weight = sum(metric_weights['REL_WEIGHTS'].values())
        rel_weights = {k: v / rel_weight for k, v in metric_weights['REL_WEIGHTS'].items()}
        
        self.STRUCTURE_WEIGHTS = [temporal_weights, rel_weights]
        
        # Initialize evaluator
        plot_dir = os.path.join(config.output_dir, "matrix_images")
        self.evaluator = MetricEvaluator(
            self.STRUCTURE_WEIGHTS,
            config.mode,
            config.subject_matching_mode,
            save_plots=config.save_plots,
            plot_dir=plot_dir,
        )
        
    def calculate_lunguage_score(self, 
                               predicted_reports: pd.DataFrame,
                               ground_truth_reports: pd.DataFrame,
                               study_id_column: str = "study_id") -> Dict[str, Any]:
        """
        Calculate LunguageScore
        
        Args:
            predicted_reports: DataFrame with predicted structured reports
            ground_truth_reports: DataFrame with ground truth structured reports
            study_id_column: Column name for study IDs
            
        Returns:
            Dictionary containing LunguageScore results
        """
        logger.info("Calculating LunguageScore...")

        # Convert data to structured format
        data_converter = DataConverter(self.STRUCTURE_WEIGHTS, self.config.mode)
        gt_data_dict = data_converter.convert_to_structured_dict(ground_truth_reports)
        pred_data_dict = data_converter.convert_to_structured_dict(predicted_reports)

        # Calculate scores
        structure_scores, prec_scores, recall_scores = self.evaluator.evaluate_dataset(
            gt_data_dict, pred_data_dict
        )

        avg_structure_score = np.mean(list(structure_scores.values())) if structure_scores else 0.0
        avg_precision = np.mean([v for v in prec_scores.values() if isinstance(v, (int, float))]) if prec_scores else 0.0
        avg_recall = np.mean([v for v in recall_scores.values() if isinstance(v, (int, float))]) if recall_scores else 0.0

        result = {
            'structure_scores': structure_scores,
            'precision_scores': prec_scores,
            'recall_scores': recall_scores,
            'avg_structure_score': avg_structure_score,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
        }
        self.results = {'lunguage_score': result}
        return result
    
    def save_results(self, output_path: Optional[str] = None) -> None:
        """Save metric results to file."""
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, "lunguage_score_results.json")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        serializable_results = {}
        for metric_name, metric_results in self.results.items():
            if isinstance(metric_results, dict):
                serializable_results[metric_name] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in metric_results.items()
                }
            else:
                serializable_results[metric_name] = metric_results

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info("Results saved to %s", output_path)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of calculated metrics"""
        summary = {}
        
        for metric_name, metric_results in self.results.items():
            if isinstance(metric_results, dict) and 'avg_structure_score' in metric_results:
                summary[metric_name] = {
                    'structure_score': metric_results['avg_structure_score'],
                    'precision': metric_results['avg_precision'],
                    'recall': metric_results['avg_recall']
                }
            elif isinstance(metric_results, dict) and 'error' in metric_results:
                summary[metric_name] = {'error': metric_results['error']}
        
        return summary
    
    def print_summary(self) -> None:
        """Print summary of calculated metrics."""
        summary = self.get_summary()
        lines = ["\n" + "=" * 50, "LUNGUAGESCORE SUMMARY", "=" * 50]
        for metric_name, results in summary.items():
            lines.append(f"\n{metric_name.upper()}:")
            if 'error' in results:
                lines.append(f"  Error: {results['error']}")
            else:
                lines.append(f"  Structure Score: {results['structure_score']:.4f}")
                lines.append(f"  Precision:       {results['precision']:.4f}")
                lines.append(f"  Recall:          {results['recall']:.4f}")
        lines.append("=" * 50)
        print("\n".join(lines))

