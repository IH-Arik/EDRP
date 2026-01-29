"""
Basic usage example for EDRP-DiaBD package.
"""

import utils.session
import numpy as np
import pandas as pd
from src.models import create_1d_cnn_model
from src.data import DataPreprocessor
from src.evaluation import ModelEvaluator
from src.utils import config, logger


def main():
    """
    Basic usage example demonstrating the EDRP-DiaBD pipeline.
    """
    logger.info("Starting EDRP-DiaBD basic usage example")
    
    # Generate sample data (replace with actual data loading)
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Create synthetic dataset
    X = np.random.randn(n_samples, n_features)
    y = np.random.binomial(1, 0.3, n_samples)  # 30% positive class
    
    # Convert to DataFrame for preprocessing
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['diabetes_status'] = y
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(random_state=config.model.random_state)
    
    # Preprocess data
    logger.info("Preprocessing data...")
    X_processed, y_processed = preprocessor.fit_transform(df, target_column='diabetes_status')
    
    # Apply SMOTE-ENN for class balancing
    X_balanced, y_balanced = preprocessor.apply_smote_enn(X_processed, y_processed)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(
        X_balanced, y_balanced, test_size=config.data.test_size
    )
    
    logger.info(f"Training set shape: {X_train.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    
    # Create model
    logger.info("Creating 1D CNN model...")
    model = create_1d_cnn_model(
        input_shape=X_train.shape[1:],
        dropout_rate=config.model.dropout_rate,
        alpha=config.model.alpha,
        gamma=config.model.gamma
    )
    
    # Print model summary
    model.summary()
    
    # Train model (simplified for example)
    logger.info("Training model...")
    callbacks = create_model_callbacks(patience=config.model.patience)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=config.model.epochs,
        batch_size=config.model.batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Make predictions
    logger.info("Making predictions...")
    y_pred_proba = model.predict(X_test).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = ModelEvaluator()
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics(y_test, y_pred, y_pred_proba)
    logger.info("Model Performance:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Generate classification report
    report = evaluator.generate_classification_report(y_test, y_pred)
    logger.info("Classification Report:\n" + report)
    
    # Create plots (if save_plots is enabled)
    if config.evaluation.save_plots:
        import os
        os.makedirs(config.evaluation.plot_dir, exist_ok=True)
        
        # Confusion Matrix
        cm_fig = evaluator.plot_confusion_matrix(y_test, y_pred)
        cm_fig.savefig(f"{config.evaluation.plot_dir}/confusion_matrix.png")
        
        # ROC Curve
        roc_fig = evaluator.plot_roc_curve(y_test, y_pred_proba)
        roc_fig.savefig(f"{config.evaluation.plot_dir}/roc_curve.png")
        
        # Precision-Recall Curve
        pr_fig = evaluator.plot_precision_recall_curve(y_test, y_pred_proba)
        pr_fig.savefig(f"{config.evaluation.plot_dir}/pr_curve.png")
        
        logger.info(f"Plots saved to {config.evaluation.plot_dir}/")
    
    logger.info("Basic usage example completed successfully!")


if __name__ == "__main__":
    main()
