"""
Machine Learning Models for Strat-e-GPT
Implements various ML models for race strategy prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
from typing import Dict, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RaceStrategyPredictor:
    """Main class for race strategy prediction using ML models"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = None
        
    def prepare_features(self, data: pd.DataFrame, target_col: str, 
                        categorical_cols: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for ML models"""
        self.target_column = target_col
        
        # Separate features and target
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        # Handle categorical variables
        if categorical_cols:
            for col in categorical_cols:
                if col in X.columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
        
        # Store feature columns
        self.feature_columns = list(X.columns)
        
        # Convert to numpy arrays
        X = X.values
        y = y.values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_models(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2) -> Dict[str, Dict]:
        """Train multiple ML models and return performance metrics"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Define models
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            # Store model and results
            self.models[name] = model
            results[name] = {
                'model': model,
                'mse': mse,
                'r2': r2,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
            
            logger.info(f"{name} - R²: {r2:.4f}, RMSE: {np.sqrt(mse):.4f}")
        
        return results
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        predictions = self.models[model_name].predict(X_scaled)
        
        return predictions
    
    def save_model(self, model_name: str, filepath: str):
        """Save a trained model to disk"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_data = {
            'model': self.models[model_name],
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a saved model from disk"""
        model_data = joblib.load(filepath)
        
        self.models['loaded_model'] = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        
        logger.info(f"Model loaded from {filepath}")
