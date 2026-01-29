"""
1D CNN model architecture for diabetes prediction.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from .loss_functions import focal_loss


def create_1d_cnn_model(input_shape, dropout_rate=0.3, alpha=0.75, gamma=2.0):
    """
    Create a 1D CNN model for diabetes prediction.
    
    Args:
        input_shape (tuple): Shape of input data (features,)
        dropout_rate (float): Dropout rate for regularization
        alpha (float): Alpha parameter for focal loss
        gamma (float): Gamma parameter for focal loss
    
    Returns:
        keras.Model: Compiled CNN model
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Reshape for Conv1D (add channel dimension)
        layers.Reshape((input_shape[0], 1)),
        
        # 1D Convolutional layers
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling1D(),
        layers.Dense(16, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model with focal loss
    model.compile(
        optimizer='adam',
        loss=focal_loss(alpha=alpha, gamma=gamma),
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model


def create_model_callbacks(patience=10, restore_best=True):
    """
    Create Keras callbacks for training.
    
    Args:
        patience (int): Early stopping patience
        restore_best (bool): Whether to restore best weights
    
    Returns:
        list: List of Keras callbacks
    """
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=restore_best,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    return callbacks_list
