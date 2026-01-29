"""
Data preprocessing utilities for diabetes prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import EditedNearestNeighbours


class DataPreprocessor:
    """
    Data preprocessing pipeline for diabetes prediction.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the preprocessor.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.target_column = None
        
    def fit_transform(self, df, target_column):
        """
        Fit preprocessing pipeline and transform data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_column (str): Name of target column
        
        Returns:
            tuple: (X, y) processed features and target
        """
        self.target_column = target_column
        df_processed = df.copy()
        
        # Handle categorical variables
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if col != target_column:
                self.label_encoders[col] = LabelEncoder()
                df_processed[col] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
        
        # Separate features and target
        X = df_processed.drop(columns=[target_column])
        y = df_processed[target_column]
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Input dataframe
        
        Returns:
            np.ndarray: Processed features
        """
        df_processed = df.copy()
        
        # Handle categorical variables
        for col, encoder in self.label_encoders.items():
            if col in df_processed.columns:
                df_processed[col] = encoder.transform(df_processed[col].astype(str))
        
        # Select and scale features
        X = df_processed[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def apply_smote_enn(self, X, y, smote_ratio=0.5):
        """
        Apply SMOTE-ENN hybrid sampling for imbalanced data.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target
            smote_ratio (float): SMOTE sampling ratio
        
        Returns:
            tuple: (X_resampled, y_resampled)
        """
        # Apply BorderlineSMOTE
        smote = BorderlineSMOTE(
            sampling_strategy=smote_ratio,
            random_state=self.random_state,
            k_neighbors=5
        )
        X_smote, y_smote = smote.fit_resample(X, y)
        
        # Apply Edited Nearest Neighbours
        enn = EditedNearestNeighbours(
            sampling_strategy='auto',
            n_neighbors=3,
            kind_sel='mode'
        )
        X_resampled, y_resampled = enn.fit_resample(X_smote, y_smote)
        
        return X_resampled, y_resampled
    
    def create_cv_splits(self, X, y, n_splits=5):
        """
        Create cross-validation splits.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target
            n_splits (int): Number of CV folds
        
        Returns:
            generator: CV split indices
        """
        skf = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        
        return skf.split(X, y)
    
    def train_test_split(self, X, y, test_size=0.2, stratify=True):
        """
        Split data into train and test sets.
        
        Args:
            X (np.ndarray): Features
            y (np.ndarray): Target
            test_size (float): Proportion of test set
            stratify (bool): Whether to stratify split
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        split_params = {
            'test_size': test_size,
            'random_state': self.random_state
        }
        
        if stratify:
            split_params['stratify'] = y
        
        return train_test_split(X, y, **split_params)
