"""
Configuration management for LunguageScore
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

CLAUDE_SONNET_MODEL = "claude-sonnet-4-6"

SINGLE_SR_MODES = frozenset([
    "gold_eval", "rexval", "maira", "maira_cascade", "medversa",
    "rgrg", "chexagent", "medgemma", "lingshu", "libra", "cvt2distilgpt2",
    "silver_eval",
])

METRIC_MODES = frozenset([
    "rexval", "single", "single_maira", "sequential_maira", "gold_eval",
    "chexagent", "medversa", "rgrg", "medgemma", "lingshu",
    "maira_cascade", "libra", "cvt2distilgpt2",
])

STRUCTURING_MODES = frozenset(["single", "sequential"])


@dataclass
class SingleSRConfig:
    """Configuration for single report structuring"""
    deployment_name: str = "gpt-4.1"
    api_key: str = "local_LLM"          # 'local_LLM' or actual API key
    mode: str = "gold_eval"              # gold_eval, rexval, maira, maira_cascade, ...
    candidate_type: str = "vocab_ent_rcg"
    unit: str = "section"               # report, section, sent
    n_retrieval: int = 5
    output_format: str = "SROSRO"
    multi: bool = False
    dynamic_retrieval: bool = False
    diverse_retrieval: bool = True
    fast_retrieval: bool = True
    candidate_usage: float = 1.0
    candidate_discontinuous: bool = False
    jaccard: bool = True
    toy_set: bool = False
    run_model: bool = True
    gold_path: str = "./dataset/Lunguage.csv"
    vocab_path: str = "./dataset/Lunguage_vocab.csv"
    output_dir: str = "./singleSR/data"
    entity_types: List[str] = field(default_factory=lambda: ['COF', 'NCD', 'PATIENT INFO.', 'PF', 'CF', 'OTH'])
    relation_types: List[str] = field(default_factory=lambda: ['Location', 'Associate', 'Evidence'])
    attribute_types: List[str] = field(default_factory=lambda: [
        'Morphology', 'Distribution', 'Measurement', 'Severity',
        'Comparison', 'Onset', 'No Change', 'Improved', 'Worsened',
        'Placement', 'Past Hx', 'Other Source', 'Assessment Limitations'
    ])
    # Server / workspace settings
    work_dir: Optional[str] = None      # Working directory (required for local runs)
    port: int = 8100                    # vLLM server port (used when api_key='local_LLM')

    def __post_init__(self):
        if self.mode not in SINGLE_SR_MODES:
            raise ValueError(
                f"Invalid SingleSRConfig.mode: '{self.mode}'. "
                f"Must be one of: {sorted(SINGLE_SR_MODES)}"
            )


@dataclass
class SequentialSRConfig:
    """Configuration for sequential report structuring"""
    LLM_name: str = "gpt-4.1"
    api_key: str = "local_LLM"
    input_path: str = "./dataset/Lunguage.csv"
    output_path: str = "./sequentialSR/results"
    batch_path: str = "./sequentialSR/batch_files"
    all_eval: bool = False
    few_shot: bool = False
    model_run: bool = False
    process_missing: bool = False
    eval_section: Optional[str] = None
    subset: List[str] = field(default_factory=list)
    # Server / workspace settings
    work_dir: Optional[str] = None      # Working directory (required for local runs)
    port: int = 8100                    # vLLM server port (used when api_key='local_LLM')
    gold_path: Optional[str] = None     # Path to Lunguage.csv (for sequence mapping); falls back to ./dataset/Lunguage.csv


@dataclass
class StructuringConfig:
    """Configuration for report structuring — selects mode and delegates to singleSR/sequentialSR"""
    mode: str = "single"                # 'single' or 'sequential'
    single: SingleSRConfig = field(default_factory=SingleSRConfig)
    sequential: SequentialSRConfig = field(default_factory=SequentialSRConfig)

    def __post_init__(self):
        if self.mode not in STRUCTURING_MODES:
            raise ValueError(
                f"Invalid StructuringConfig.mode: '{self.mode}'. "
                f"Must be one of: {sorted(STRUCTURING_MODES)}"
            )


@dataclass
class MetricConfig:
    """Configuration for LunguageScore calculation"""
    semantic_model: str = "FremyCompany/BioLORD-2023"
    output_dir: str = "./results"
    mode: str = "rexval"  # rexval, single_maira, sequential_maira, etc.
    subject_matching_mode: str = "semantic"  # semantic, string
    save_plots: bool = False  # Save similarity matrix visualizations to output_dir/matrix_images/

    def __post_init__(self):
        if self.mode not in METRIC_MODES:
            raise ValueError(
                f"Invalid MetricConfig.mode: '{self.mode}'. "
                f"Must be one of: {sorted(METRIC_MODES)}"
            )
        if self.subject_matching_mode not in ("semantic", "string"):
            raise ValueError(
                f"Invalid MetricConfig.subject_matching_mode: '{self.subject_matching_mode}'. "
                "Must be 'semantic' or 'string'."
            )


@dataclass
class Config:
    """Main configuration class"""
    structuring: StructuringConfig = field(default_factory=StructuringConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        s = config_dict.get('structuring', {})
        single_cfg = SingleSRConfig(**s.get('single', {}))
        seq_cfg = SequentialSRConfig(**s.get('sequential', {}))
        structuring_config = StructuringConfig(
            mode=s.get('mode', 'single'),
            single=single_cfg,
            sequential=seq_cfg,
        )
        metric_config = MetricConfig(**config_dict.get('metrics', {}))
        return cls(structuring=structuring_config, metrics=metric_config)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file"""
        import dataclasses
        config_dict = {
            'structuring': {
                'mode': self.structuring.mode,
                'single': dataclasses.asdict(self.structuring.single),
                'sequential': dataclasses.asdict(self.structuring.sequential),
            },
            'metrics': dataclasses.asdict(self.metrics),
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def get_default_config() -> Config:
    """Get default configuration"""
    return Config()


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or return default"""
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)
    return get_default_config()
