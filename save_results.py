#!/usr/bin/env python3
"""
Save Results Script for Strat-e-GPT
Saves analysis results to JSON files for the viewer to load
"""

import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def save_results_to_json():
    """Save analysis results to JSON files"""
    try:
        from f1_data_analyzer import F1DataAnalyzer
        from ml_models import RaceStrategyPredictor
        
        print("Loading F1 data and running analysis...")
        
        # Initialize components
        f1_analyzer = F1DataAnalyzer()
        ml_predictor = RaceStrategyPredictor()
        
        # Load data
        archive_data = f1_analyzer.load_archive_data()
        datathon_data = f1_analyzer.load_datathon_data()
        
        # Get data summary
        data_summary = f1_analyzer.get_data_summary()
        
        # Get ML suggestions
        ml_suggestions = f1_analyzer.suggest_ml_approach()
        
        # Create features and train models
        X, y = f1_analyzer.create_racing_features(target_column="position")
        results = ml_predictor.train_models(X, y)
        
        # Prepare results for saving
        results_to_save = {
            'performance_results': {},
            'feature_importance': [],
            'data_summary': {
                'total_archive_tables': data_summary['total_archive_tables'],
                'total_datathon_files': data_summary['total_datathon_files'],
                'ml_suggestions': ml_suggestions
            },
            'raw_results': {
                'features_shape': list(X.shape),
                'target_shape': list(y.shape),
                'models_trained': list(results.keys())
            }
        }
        
        # Extract performance metrics
        for model_name, model_results in results.items():
            results_to_save['performance_results'][model_name] = {
                'r2': float(model_results['r2']),
                'rmse': float(model_results['rmse']),
                'mae': float(model_results['mae'])
            }
        
        # Extract feature importance from Random Forest
        if 'random_forest' in ml_predictor.models:
            rf_model = ml_predictor.models['random_forest']
            if hasattr(rf_model, 'feature_importances_'):
                feature_names = ['grid', 'laps', 'fastestLap', 'fastestLapSpeed', 
                               'year', 'round', 'driver_nationality', 'constructor_nationality']
                feature_importance = rf_model.feature_importances_
                
                # Sort by importance
                importance_pairs = list(zip(feature_names, feature_importance))
                importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                results_to_save['feature_importance'] = [
                    {'feature': name, 'importance': float(imp)} 
                    for name, imp in importance_pairs
                ]
        
        # Save to JSON file
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        results_file = outputs_dir / "analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Results saved to {results_file}")
        
        # Also save individual files for easier access
        with open(outputs_dir / "performance_results.json", 'w') as f:
            json.dump(results_to_save['performance_results'], f, indent=2)
        
        with open(outputs_dir / "feature_importance.json", 'w') as f:
            json.dump(results_to_save['feature_importance'], f, indent=2)
        
        with open(outputs_dir / "data_summary.json", 'w') as f:
            json.dump(results_to_save['data_summary'], f, indent=2)
        
        print("Individual result files saved successfully!")
        return True
        
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

if __name__ == "__main__":
    success = save_results_to_json()
    if success:
        print("✅ Results saved successfully! You can now run the viewer.")
    else:
        print("❌ Failed to save results. Check the error messages above.")
