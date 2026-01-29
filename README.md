# EDRP-DiaBD: Explainable Ensemble Modeling for Diabetes Risk Prediction

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)](https://github.com/IH-Arik/EDRP)

## Overview

This repository contains the implementation of **Explainable Ensemble Modeling for Diabetes Risk Prediction** using the real-world DiaBD cohort. The project focuses on developing robust, interpretable machine learning models for early diabetes risk stratification.

## Key Features

- **1D CNN Architecture**: Specialized convolutional neural network for medical data
- **Focal Loss Implementation**: Handles class imbalance in medical datasets
- **SMOTE-ENN Hybrid Sampling**: Advanced data balancing techniques
- **Comprehensive Evaluation**: Multiple metrics with visualization tools
- **Professional Structure**: Industry-standard code organization
- **Reproducible Research**: Configurable experiments with logging

## Project Structure

```
EDRP-DiaBD/
├── src/                          # Source code
│   ├── models/                   # Model architectures
│   │   ├── cnn_model.py         # 1D CNN implementation
│   │   └── loss_functions.py    # Custom loss functions
│   ├── data/                     # Data processing
│   │   └── preprocessing.py     # Data preprocessing pipeline
│   ├── evaluation/               # Model evaluation
│   │   └── metrics.py          # Evaluation metrics and plots
│   └── utils/                   # Utilities
│       ├── config.py           # Configuration management
│       └── logger.py           # Logging utilities
├── notebooks/                    # Research notebooks
│   └── edrp-diabd.ipynb        # Main analysis notebook
├── tests/                        # Unit tests
├── docs/                         # Documentation
├── configs/                      # Configuration files
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/IH-Arik/EDRP.git
cd EDRP

# Install dependencies
pip install -r requirements.txt

# Install the package (optional)
pip install -e .
```

### Basic Usage

```python
from src.models import create_1d_cnn_model
from src.data import DataPreprocessor
from src.evaluation import ModelEvaluator

# Initialize components
preprocessor = DataPreprocessor()
evaluator = ModelEvaluator()

# Load and preprocess data
X, y = preprocessor.fit_transform(df, target_column='diabetes_status')

# Create model
model = create_1d_cnn_model(input_shape=X.shape[1:])

# Train and evaluate
# (See notebooks for complete examples)
```

## Model Architecture

The core model uses a **1D Convolutional Neural Network** with the following architecture:

```
Input → Conv1D(64) → BatchNorm → Dropout → 
Conv1D(32) → BatchNorm → Dropout → 
GlobalAvgPool → Dense(16) → BatchNorm → Dropout → 
Output(Sigmoid)
```

### Key Features:

- **Focal Loss**: Addresses class imbalance with α=0.75, γ=2.0
- **Batch Normalization**: Stabilizes training
- **Dropout Regularization**: Prevents overfitting (rate=0.3)
- **Global Average Pooling**: Reduces parameters

## Data Processing Pipeline

1. **Categorical Encoding**: Label encoding for categorical variables
2. **Feature Scaling**: StandardScaler for numerical features
3. **Class Balancing**: SMOTE-ENN hybrid sampling
4. **Cross-Validation**: Stratified K-Fold (5-fold)

## Evaluation Metrics

- **Primary Metrics**: Accuracy, Precision, Recall, F1-Score
- **Advanced Metrics**: ROC-AUC, PR-AUC
- **Visualization**: Confusion Matrix, ROC Curves, PR Curves
- **Cross-Validation**: Comprehensive statistical summary

## Configuration

The project uses a centralized configuration system:

```python
from src.utils import config

# Access configuration
print(config.model.dropout_rate)  # 0.3
print(config.data.test_size)      # 0.2
print(config.evaluation.metrics)  # ['accuracy', 'precision', ...]
```

## Research Notebooks

The main research analysis is available in `notebooks/edrp-diabd.ipynb`, containing:

- Complete data analysis pipeline
- Model training and evaluation
- Results visualization
- Statistical analysis

## Development

### Running Tests

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Code Style

```bash
# Format code
black src/

# Lint code
flake8 src/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{edrp_diabd_2024,
  title={EDRP-DiaBD: Explainable Ensemble Modeling for Diabetes Risk Prediction},
  author={Research Team},
  year={2024},
  url={https://github.com/IH-Arik/EDRP}
}
```

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- DiaBD Cohort data providers
- Medical research community
- Open-source contributors

## Contact

For questions or collaborations:
- Email: research@example.com
- GitHub Issues: [Create an issue](https://github.com/IH-Arik/EDRP/issues)

---

**Note**: This repository contains the implementation code for research purposes. The research paper describing the methodology and results is currently under review for publication.
