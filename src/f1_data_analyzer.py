"""
F1 Data Analyzer for Strat-e-GPT
Specialized module for analyzing Formula 1 racing data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class F1DataAnalyzer:
    """Analyzes F1 racing data from multiple sources"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.archive_data = {}
        self.datathon_data = {}
        self.merged_data = None
        
    def load_archive_data(self) -> Dict[str, pd.DataFrame]:
        """Load historical F1 archive data"""
        archive_dir = self.data_dir / "archive"
        
        if not archive_dir.exists():
            logger.warning("Archive data directory not found")
            return {}
        
        # Load key tables
        tables = [
            'drivers', 'constructors', 'races', 'results', 
            'qualifying', 'pit_stops', 'lap_times',
            'driver_standings', 'constructor_standings'
        ]
        
        for table in tables:
            file_path = archive_dir / f"{table}.csv"
            if file_path.exists():
                try:
                    self.archive_data[table] = pd.read_csv(file_path)
                    logger.info(f"Loaded {table}: {self.archive_data[table].shape}")
                except Exception as e:
                    logger.error(f"Error loading {table}: {e}")
        
        return self.archive_data
    
    def load_datathon_data(self) -> Dict[str, pd.DataFrame]:
        """Load F1nalyze datathon data"""
        datathon_dir = self.data_dir / "f1nalyze-datathon-ieeecsmuj"
        
        if not datathon_dir.exists():
            logger.warning("Datathon data directory not found")
            return {}
        
        # Load datathon files
        files = ['train', 'test', 'validation', 'sample_submission']
        
        for file_name in files:
            file_path = datathon_dir / f"{file_name}.csv"
            if file_path.exists():
                try:
                    self.datathon_data[file_name] = pd.read_csv(file_path)
                    logger.info(f"Loaded {file_name}: {self.datathon_data[file_name].shape}")
                except Exception as e:
                    logger.error(f"Error loading {file_name}: {e}")
        
        return self.datathon_data
    
    def analyze_archive_data(self) -> Dict:
        """Analyze historical F1 data structure and content"""
        if not self.archive_data:
            self.load_archive_data()
        
        analysis = {}
        
        for table_name, df in self.archive_data.items():
            analysis[table_name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'sample_data': df.head(3).to_dict('records')
            }
        
        return analysis
    
    def analyze_datathon_data(self) -> Dict:
        """Analyze datathon data structure and content"""
        if not self.datathon_data:
            self.load_datathon_data()
        
        analysis = {}
        
        for table_name, df in self.datathon_data.items():
            analysis[table_name] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'sample_data': df.head(3).to_dict('records')
            }
        
        return analysis
    
    def create_racing_features(self, target_column: str = "position") -> Tuple[pd.DataFrame, pd.Series]:
        """Create features from historical F1 data for ML prediction"""
        if not self.archive_data:
            self.load_archive_data()
        
        # Start with results data
        if 'results' not in self.archive_data:
            raise ValueError("Results data not available")
        
        results = self.archive_data['results'].copy()
        
        # Merge with related data
        if 'drivers' in self.archive_data:
            results = results.merge(
                self.archive_data['drivers'][['driverId', 'nationality']], 
                on='driverId', how='left'
            )
        
        if 'constructors' in self.archive_data:
            results = results.merge(
                self.archive_data['constructors'][['constructorId', 'nationality']], 
                on='constructorId', how='left'
            )
        
        if 'races' in self.archive_data:
            results = results.merge(
                self.archive_data['races'][['raceId', 'year', 'round', 'circuitId']], 
                on='raceId', how='left'
            )
        
        # Create features
        features = results.copy()
        
        # Handle missing values
        features = features.fillna({
            'position': 999,  # DNF/DNS
            'grid': 999,
            'points': 0,
            'laps': 0
        })
        
        # Convert position to numeric, handling DNF/DNS
        features['position_numeric'] = pd.to_numeric(features['position'], errors='coerce')
        features['position_numeric'] = features['position_numeric'].fillna(999)
        
        # Create target variable
        if target_column == "position":
            y = features['position_numeric']
        elif target_column == "points":
            y = features['points']
        else:
            y = features[target_column]
        
        # Select feature columns
        feature_cols = [
            'grid', 'laps', 'fastestLap', 'fastestLapSpeed',
            'year', 'round'
        ]
        
        # Add categorical features
        categorical_cols = ['nationality_x', 'nationality_y']  # driver and constructor nationality
        
        # Prepare final features
        X = features[feature_cols + categorical_cols].copy()
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].fillna('Unknown')
                X[col] = pd.Categorical(X[col]).codes
        
        # Remove rows with missing target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Created features: {X.shape}, target: {y.shape}")
        
        return X, y
    
    def get_data_summary(self) -> Dict:
        """Get comprehensive summary of all available data"""
        summary = {
            'archive_data': self.analyze_archive_data(),
            'datathon_data': self.analyze_datathon_data(),
            'total_archive_tables': len(self.archive_data),
            'total_datathon_files': len(self.datathon_data)
        }
        
        return summary
    
    def suggest_ml_approach(self) -> Dict:
        """Suggest ML approach based on available data"""
        suggestions = {
            'archive_data': {
                'approach': 'Traditional ML with engineered features',
                'target_variables': ['position', 'points', 'laps'],
                'feature_engineering': [
                    'Driver performance history',
                    'Constructor performance',
                    'Track-specific features',
                    'Seasonal trends'
                ],
                'models': ['Random Forest', 'Gradient Boosting', 'XGBoost']
            },
            'datathon_data': {
                'approach': 'Modern ML with raw features',
                'target_variables': ['position', 'result_driver_standing'],
                'feature_engineering': [
                    'Raw telemetry data',
                    'Time series features',
                    'Driver-constructor combinations'
                ],
                'models': ['Neural Networks', 'Ensemble Methods', 'Time Series Models']
            }
        }
        
        return suggestions
