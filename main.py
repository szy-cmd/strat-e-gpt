"""
Main application for Strat-e-GPT
Race Strategy Prediction using Machine Learning
"""

import os
import sys
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_processing import RacingDataProcessor
from src.ml_models import RaceStrategyPredictor
from src.visualization import RacingDataVisualizer
from src.f1_data_analyzer import F1DataAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main function to demonstrate Strat-e-GPT capabilities"""
    logger.info("Starting Strat-e-GPT - Race Strategy Prediction System")
    
    # Initialize components
    data_processor = RacingDataProcessor()
    ml_predictor = RaceStrategyPredictor()
    visualizer = RacingDataVisualizer()
    f1_analyzer = F1DataAnalyzer()
    
    # Example workflow
    logger.info("Initializing F1 data analysis pipeline...")
    
    # Check if data directory exists
    data_dir = Path("data")
    if not data_dir.exists():
        logger.warning("Data directory not found. Creating example structure...")
        data_dir.mkdir(exist_ok=True)
        create_sample_data(data_dir)
    
    # Analyze F1 data
    try:
        logger.info("Analyzing F1 racing data...")
        
        # Load and analyze F1 data
        f1_analyzer.load_archive_data()
        f1_analyzer.load_datathon_data()
        
        # Get comprehensive data summary
        data_summary = f1_analyzer.get_data_summary()
        logger.info(f"Data Summary:")
        logger.info(f"  Archive tables: {data_summary['total_archive_tables']}")
        logger.info(f"  Datathon files: {data_summary['total_datathon_files']}")
        
        # Get ML approach suggestions
        suggestions = f1_analyzer.suggest_ml_approach()
        logger.info(f"ML Approach Suggestions:")
        for data_type, suggestion in suggestions.items():
            logger.info(f"  {data_type}: {suggestion['approach']}")
            logger.info(f"    Target variables: {suggestion['target_variables']}")
            logger.info(f"    Recommended models: {suggestion['models']}")
        
        # Try to create features from archive data
        try:
            logger.info("Creating racing features from historical data...")
            X, y = f1_analyzer.create_racing_features(target_column="position")
            
            if X.shape[0] > 0 and y.shape[0] > 0:
                # Train models
                logger.info("Training machine learning models...")
                results = ml_predictor.train_models(X, y)
                
                # Visualize results
                logger.info("Creating visualizations...")
                
                # Model performance comparison
                perf_fig = visualizer.plot_model_performance(results)
                visualizer.save_figure(perf_fig, "outputs/f1_model_performance.png")
                
                # Create feature importance plot if Random Forest is available
                if 'random_forest' in ml_predictor.models:
                    rf_model = ml_predictor.models['random_forest']
                    if hasattr(rf_model, 'feature_importances_'):
                        feature_importance = rf_model.feature_importances_
                        feature_names = ml_predictor.feature_columns
                        
                        # Create feature importance plot
                        importance_df = pd.DataFrame({
                            'feature': feature_names,
                            'importance': feature_importance
                        }).sort_values('importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        importance_df.plot(x='feature', y='importance', kind='barh', ax=ax)
                        ax.set_title('F1 Racing Feature Importance')
                        ax.set_xlabel('Feature Importance')
                        plt.tight_layout()
                        visualizer.save_figure(fig, "outputs/f1_feature_importance.png")
                
                logger.info("F1 analysis complete! Check the outputs/ directory for results.")
                
            else:
                logger.warning("No valid features created from historical data")
                
        except Exception as e:
            logger.error(f"Error creating features from historical data: {e}")
            logger.info("Falling back to sample data...")
            
            # Fallback to sample data
            data_file = data_dir / "sample_racing_data.csv"
            if data_file.exists():
                data = data_processor.load_data(str(data_file))
                cleaned_data = data_processor.clean_data()
                X, y = ml_predictor.prepare_features(cleaned_data, "finish_position")
                results = ml_predictor.train_models(X, y)
                logger.info("Sample data analysis complete!")
            else:
                logger.warning("No sample data available for fallback")
                
    except Exception as e:
        logger.error(f"Error during F1 analysis: {e}")
        logger.info("Please ensure you have the required dependencies installed and data available.")
            
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        logger.info("Please ensure you have the required dependencies installed and data available.")

def create_sample_data(data_dir: Path):
    """Create sample racing data for demonstration"""
    import pandas as pd
    import numpy as np
    
    # Create sample racing data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'driver_id': range(1, n_samples + 1),
        'qualifying_position': np.random.randint(1, 21, n_samples),
        'starting_position': np.random.randint(1, 21, n_samples),
        'lap_times': np.random.normal(85, 5, n_samples),  # Average lap time around 85 seconds
        'fuel_load': np.random.uniform(80, 120, n_samples),  # Fuel in kg
        'tire_wear': np.random.uniform(0, 100, n_samples),  # Tire wear percentage
        'weather_condition': np.random.choice(['dry', 'wet', 'intermediate'], n_samples),
        'track_temperature': np.random.uniform(15, 35, n_samples),  # Celsius
        'pit_stops': np.random.randint(1, 5, n_samples),
        'finish_position': np.random.randint(1, 21, n_samples)
    }
    
    # Add some correlations to make it more realistic
    df = pd.DataFrame(data)
    
    # Better drivers tend to have better qualifying and starting positions
    df['qualifying_position'] = np.clip(df['qualifying_position'] + np.random.normal(0, 2, n_samples), 1, 20)
    df['starting_position'] = np.clip(df['qualifying_position'] + np.random.normal(0, 3, n_samples), 1, 20)
    
    # Better positions correlate with fewer pit stops and better finish
    df['pit_stops'] = np.clip(df['pit_stops'] - (df['qualifying_position'] - 10) // 5, 1, 5)
    df['finish_position'] = np.clip(df['starting_position'] + np.random.normal(0, 4, n_samples), 1, 20)
    
    # Save sample data
    sample_file = data_dir / "sample_racing_data.csv"
    df.to_csv(sample_file, index=False)
    logger.info(f"Sample data created at {sample_file}")

if __name__ == "__main__":
    # Create outputs directory
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    main()
