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
                    # Fix 1: Handle '\N' null values during import
                    # Fix 2: Use low_memory=False to avoid DtypeWarnings
                    df = pd.read_csv(file_path, na_values='\\N', low_memory=False)
                    
                    # Fix 3: Clean numeric columns to resolve mixed data types
                    df = self._clean_numeric_columns(df, table)
                    
                    self.archive_data[table] = df
                    logger.info(f"Loaded {table}: {self.archive_data[table].shape}")
                except Exception as e:
                    logger.error(f"Error loading {table}: {e}")
        
        return self.archive_data
    
    def _clean_numeric_columns(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Clean numeric columns to resolve mixed data types and DtypeWarnings"""
        df_clean = df.copy()
        
        # Common numeric columns in F1 data
        numeric_columns = [
            'position', 'grid', 'points', 'laps', 'fastestLap', 'fastestLapSpeed',
            'raceId', 'driverId', 'constructorId', 'circuitId', 'year', 'round',
            'rank', 'positionText', 'positionOrder', 'points', 'wins'
        ]
        
        # Clean each numeric column
        for col in numeric_columns:
            if col in df_clean.columns:
                try:
                    # Skip if already numeric
                    if pd.api.types.is_numeric_dtype(df_clean[col]):
                        continue
                    
                    # Convert to numeric, coercing errors to NaN
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                    
                    # Log cleaning results
                    original_dtype = df[col].dtype
                    new_dtype = df_clean[col].dtype
                    if original_dtype != new_dtype:
                        logger.info(f"Cleaned {table_name}.{col}: {original_dtype} → {new_dtype}")
                        
                except Exception as e:
                    logger.warning(f"Could not clean column {table_name}.{col}: {e}")
                    # Keep original column if cleaning fails
                    continue
        
        return df_clean
    
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
                    # Fix 1: Handle '\N' null values during import
                    # Fix 2: Use low_memory=False to avoid DtypeWarnings
                    df = pd.read_csv(file_path, na_values='\\N', low_memory=False)
                    
                    # Fix 3: Clean numeric columns to resolve mixed data types
                    df = self._clean_numeric_columns(df, file_name)
                    
                    self.datathon_data[file_name] = df
                    logger.info(f"Loaded {file_name}: {self.datathon_data[file_name].shape}")
                except Exception as e:
                    logger.error(f"Error loading {file_name}: {e}")
        
        return self.datathon_data
    
    def validate_data_quality(self) -> Dict[str, Dict]:
        """Validate data quality after loading to catch any remaining issues"""
        validation_results = {}
        
        # Validate archive data
        for table_name, df in self.archive_data.items():
            validation_results[table_name] = self._validate_table(df, table_name)
        
        # Validate datathon data
        for file_name, df in self.datathon_data.items():
            validation_results[file_name] = self._validate_table(df, file_name)
        
        return validation_results
    
    def _validate_table(self, df: pd.DataFrame, table_name: str) -> Dict:
        """Validate individual table for common data quality issues"""
        validation = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'issues': []
        }
        
        # Check for common F1 data issues
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed content in object columns
                unique_values = df[col].dropna().unique()
                if len(unique_values) < 100:  # Only check if reasonable number of unique values
                    numeric_count = sum(1 for val in unique_values if str(val).replace('.', '').replace('-', '').isdigit())
                    if numeric_count > 0 and numeric_count < len(unique_values):
                        validation['issues'].append(f"Column '{col}' has mixed numeric/text content")
        
        return validation
    
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
        
        # Handle missing values - data is already cleaned by _clean_numeric_columns
        features = features.fillna({
            'position': 999,  # DNF/DNS
            'grid': 999,
            'points': 0,
            'laps': 0
        })
        
        # Position is already numeric from cleaning, just handle NaN values
        features['position_numeric'] = features['position'].fillna(999)
        
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
        
        # Handle missing values in numeric features
        for col in feature_cols:
            if col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    # For numeric columns, fill NaN with median
                    median_val = X[col].median()
                    X[col] = X[col].fillna(median_val)
                    logger.info(f"Filled NaN in {col} with median: {median_val}")
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in X.columns:
                X[col] = X[col].fillna('Unknown')
                X[col] = pd.Categorical(X[col]).codes
        
        # Remove rows with missing target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Final check: ensure no NaN values remain
        if X.isnull().any().any():
            logger.warning("NaN values still present after cleaning. Removing affected rows...")
            X = X.dropna()
            y = y[X.index]  # Align target with cleaned features
        
        logger.info(f"Final feature matrix: {X.shape}, no NaN values: {not X.isnull().any().any()}")
        
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
