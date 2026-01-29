"""
Configuration settings for diabetes prediction models.
"""

import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    
    # Model architecture
    input_shape: tuple = (100,)  # Will be updated based on data
    dropout_rate: float = 0.3
    alpha: float = 0.75  # Focal loss alpha
    gamma: float = 2.0   # Focal loss gamma
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10
    learning_rate: float = 0.001
    
    # Cross-validation
    n_splits: int = 5
    random_state: int = 42


@dataclass
class DataConfig:
    """Data processing configuration."""
    
    # File paths
    data_path: str = "data/diabd_cohort.csv"
    target_column: str = "diabetes_status"
    
    # Preprocessing
    test_size: float = 0.2
    smote_ratio: float = 0.5
    scale_features: bool = True
    handle_categorical: bool = True
    
    # Feature selection
    max_features: Optional[int] = None
    feature_importance_threshold: float = 0.01


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    
    # Metrics to calculate
    metrics: List[str] = None
    
    # Plotting
    save_plots: bool = True
    plot_dir: str = "plots"
    
    # Results
    save_results: bool = True
    results_dir: str = "results"
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']


@dataclass
class ProjectConfig:
    """Overall project configuration."""
    
    # Project info
    project_name: str = "EDRP-DiaBD"
    version: str = "1.0.0"
    description: str = "Explainable Ensemble Modeling for Diabetes Risk Prediction"
    
    # Paths
    root_dir: str = os.getcwd()
    src_dir: str = "src"
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"
    
    # Sub-configurations
    model: ModelConfig = ModelConfig()
    data: DataConfig = DataConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    
    def __post_init__(self):
        # Create absolute paths
        self.src_dir = os.path.join(self.root_dir, self.src_dir)
        self.data_dir = os.path.join(self.root_dir, self.data_dir)
        self.models_dir = os.path.join(self.root_dir, self.models_dir)
        self.logs_dir = os.path.join(self.root_dir, self.logs_dir)
        
        # Update data path
        self.data.data_path = os.path.join(self.root_dir, self.data.data_path)


# Global configuration instance
config = ProjectConfig()
