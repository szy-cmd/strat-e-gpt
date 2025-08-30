"""
Visualization module for Strat-e-GPT
Creates charts and plots for racing data analysis and model results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class RacingDataVisualizer:
    """Creates visualizations for racing data and ML model results"""
    
    def __init__(self):
        self.figures = {}
    
    def plot_data_distribution(self, data: pd.DataFrame, columns: List[str] = None, 
                              figsize: tuple = (15, 10)) -> plt.Figure:
        """Plot distribution of numerical columns"""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, col in enumerate(columns):
            if i < len(axes):
                axes[i].hist(data[col].dropna(), bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        # Hide empty subplots
        for i in range(len(columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_correlation_matrix(self, data: pd.DataFrame, figsize: tuple = (10, 8)) -> plt.Figure:
        """Plot correlation matrix of numerical columns"""
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, ax=ax)
        
        ax.set_title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        return fig
    
    def plot_model_performance(self, results: Dict, figsize: tuple = (12, 8)) -> plt.Figure:
        """Plot model performance comparison"""
        models = list(results.keys())
        metrics = ['r2', 'rmse', 'mae']
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in models]
            
            if metric == 'r2':
                # Higher is better for R²
                colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in values]
            else:
                # Lower is better for RMSE and MAE
                colors = ['green' if v < np.mean(values) else 'orange' if v < np.mean(values) * 1.2 else 'red' for v in values]
            
            bars = axes[i].bar(models, values, color=colors, alpha=0.7)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  model_name: str = "Model") -> plt.Figure:
        """Plot predicted vs actual values"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        ax1.scatter(y_true, y_pred, alpha=0.6)
        ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title(f'{model_name}: Predicted vs Actual')
        
        # Residuals plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title(f'{model_name}: Residuals Plot')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self, data: pd.DataFrame, 
                                   target_col: str = None) -> go.Figure:
        """Create an interactive Plotly dashboard"""
        if target_col is None:
            target_col = data.select_dtypes(include=[np.number]).columns[0]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Distribution', 'Target Distribution', 
                          'Feature vs Target', 'Correlation Heatmap'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # Feature distribution
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            feature_col = numeric_cols[0] if numeric_cols[0] != target_col else numeric_cols[1] if len(numeric_cols) > 1 else None
            if feature_col:
                fig.add_trace(
                    go.Histogram(x=data[feature_col], name=feature_col),
                    row=1, col=1
                )
        
        # Target distribution
        fig.add_trace(
            go.Histogram(x=data[target_col], name=target_col),
            row=1, col=2
        )
        
        # Feature vs Target scatter
        if feature_col:
            fig.add_trace(
                go.Scatter(x=data[feature_col], y=data[target_col], 
                          mode='markers', name=f'{feature_col} vs {target_col}'),
                row=2, col=1
            )
        
        # Correlation heatmap
        corr_matrix = data.select_dtypes(include=[np.number]).corr()
        fig.add_trace(
            go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, 
                      y=corr_matrix.columns, colorscale='RdBu'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, title_text="Racing Data Analysis Dashboard")
        return fig
    
    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = 300):
        """Save a matplotlib figure to file"""
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        logger.info(f"Figure saved as {filename}")
    
    def save_plotly_figure(self, fig: go.Figure, filename: str):
        """Save a Plotly figure to HTML file"""
        fig.write_html(filename)
        logger.info(f"Interactive figure saved as {filename}")
