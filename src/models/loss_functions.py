"""
Custom loss functions for diabetes prediction models.
"""

import tensorflow as tf
from tensorflow.keras import backend as K


def focal_loss(alpha=0.75, gamma=2.0):
    """
    Focal Loss for imbalanced classification.
    
    Args:
        alpha (float): Weight for positive class (default: 0.75)
        gamma (float): Focusing parameter (default: 2.0)
    
    Returns:
        Callable: Focal loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        # Cast y_true to float32 to match y_pred
        y_true = tf.cast(y_true, tf.float32)
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        
        # Focal loss calculation
        pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = K.pow(1 - pt, gamma)
        
        # Binary cross entropy
        bce = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        
        # Apply focal weight and alpha
        loss = alpha * focal_weight * bce
        
        return K.mean(loss)
    
    return focal_loss_fixed
