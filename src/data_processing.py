"""
Data processing module for Strat-e-GPT
Handles data loading, cleaning, and preprocessing for racing datasets
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RacingDataProcessor:
    """Processes racing data for machine learning models"""
    
    def __init__(self):
        self.data = None
        self.cleaned_data = None
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load racing data from various file formats"""
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                self.data = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            logger.info(f"Successfully loaded data with shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess the racing data"""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        # Create a copy for cleaning
        self.cleaned_data = self.data.copy()
        
        # Remove duplicates
        self.cleaned_data = self.cleaned_data.drop_duplicates()
        
        # Handle missing values
        self.cleaned_data = self.cleaned_data.fillna(method='ffill')
        
        # Remove rows with too many missing values
        threshold = len(self.cleaned_data.columns) * 0.5
        self.cleaned_data = self.cleaned_data.dropna(thresh=threshold)
        
        logger.info(f"Data cleaned. Shape: {self.cleaned_data.shape}")
        return self.cleaned_data
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the data"""
        if self.cleaned_data is None:
            raise ValueError("No cleaned data available. Please clean data first.")
        
        summary = {
            'shape': self.cleaned_data.shape,
            'columns': list(self.cleaned_data.columns),
            'dtypes': self.cleaned_data.dtypes.to_dict(),
            'missing_values': self.cleaned_data.isnull().sum().to_dict(),
            'numeric_columns': list(self.cleaned_data.select_dtypes(include=[np.number]).columns)
        }
        
        return summary
