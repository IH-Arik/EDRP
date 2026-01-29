"""
Evaluation metrics for diabetes prediction models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, roc_curve
)


class ModelEvaluator:
    """
    Comprehensive model evaluation utilities.
    """
    
    def __init__(self):
        self.results = {}
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_pred_proba (array): Predicted probabilities (optional)
        
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
            # Calculate precision-recall curve metrics
            precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['pr_auc'] = np.trapz(precision, recall)
        
        return metrics
    
    def generate_classification_report(self, y_true, y_pred, target_names=None):
        """
        Generate detailed classification report.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            target_names (list): Names of target classes
        
        Returns:
            str: Classification report
        """
        return classification_report(y_true, y_pred, target_names=target_names)
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, normalize=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            labels (list): Class labels
            normalize (str): Normalization method ('true', 'pred', 'all')
        
        Returns:
            matplotlib.figure.Figure: Confusion matrix plot
        """
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        return fig
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """
        Plot ROC curve.
        
        Args:
            y_true (array): True labels
            y_pred_proba (array): Predicted probabilities
        
        Returns:
            matplotlib.figure.Figure: ROC curve plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='blue', lw=2, 
                label=f'ROC Curve (AUC = {auc_score:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba):
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true (array): True labels
            y_pred_proba (array): Predicted probabilities
        
        Returns:
            matplotlib.figure.Figure: Precision-Recall curve plot
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = np.trapz(precision, recall)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color='blue', lw=2,
                label=f'PR Curve (AUC = {pr_auc:.3f})')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def cross_validate_results(self, cv_results):
        """
        Summarize cross-validation results.
        
        Args:
            cv_results (list): List of CV fold results
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        df = pd.DataFrame(cv_results)
        
        summary = pd.DataFrame({
            'mean': df.mean(),
            'std': df.std(),
            'min': df.min(),
            'max': df.max(),
            'median': df.median()
        })
        
        return summary
