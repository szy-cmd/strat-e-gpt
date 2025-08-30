#!/usr/bin/env python3
"""
Command-Line Results Viewer for Strat-e-GPT
Alternative to Tkinter viewer for systems without GUI support
"""

import json
from pathlib import Path
import sys

def view_results_cli():
    """Display results in command line format"""
    print("🏁 Strat-e-GPT Results Viewer (CLI)")
    print("=" * 50)
    
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        print("❌ No outputs directory found. Run the main program first.")
        return
    
    # Check for JSON results
    results_file = outputs_dir / "analysis_results.json"
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            display_results(results)
        except Exception as e:
            print(f"❌ Error loading results: {e}")
    else:
        print("❌ No analysis results found. Run save_results.py first.")
        print("💡 Available files:")
        for file_path in outputs_dir.glob("*"):
            if file_path.is_file():
                size_kb = file_path.stat().st_size / 1024
                print(f"  📁 {file_path.name} ({size_kb:.1f} KB)")

def display_results(results):
    """Display results in formatted text"""
    
    # Model Performance
    print("\n📊 MODEL PERFORMANCE")
    print("-" * 30)
    if 'performance_results' in results:
        for model_name, metrics in results['performance_results'].items():
            print(f"  {model_name.replace('_', ' ').title()}:")
            print(f"    R² Score: {metrics.get('r2', 0):.4f}")
            print(f"    RMSE: {metrics.get('rmse', 0):.2f}")
            print(f"    MAE: {metrics.get('mae', 0):.2f}")
            print()
    
    # Feature Importance
    print("🎯 FEATURE IMPORTANCE")
    print("-" * 30)
    if 'feature_importance' in results:
        for i, item in enumerate(results['feature_importance'], 1):
            feature = item.get('feature', 'Unknown')
            importance = item.get('importance', 0)
            print(f"  {i:2d}. {feature}: {importance:.4f}")
    
    # Data Summary
    print("\n📈 DATA SUMMARY")
    print("-" * 30)
    if 'data_summary' in results:
        summary = results['data_summary']
        print(f"  Archive Tables: {summary.get('total_archive_tables', 0)}")
        print(f"  Datathon Files: {summary.get('total_datathon_files', 0)}")
        
        if 'ml_suggestions' in summary:
            print("\n  ML Approach Suggestions:")
            for data_type, suggestion in summary['ml_suggestions'].items():
                print(f"    • {data_type}: {suggestion.get('approach', 'N/A')}")
                print(f"      Models: {', '.join(suggestion.get('models', []))}")
    
    # Raw Results
    print("\n🔍 RAW RESULTS")
    print("-" * 30)
    if 'raw_results' in results:
        raw = results['raw_results']
        print(f"  Features Shape: {raw.get('features_shape', [])}")
        print(f"  Target Shape: {raw.get('target_shape', [])}")
        print(f"  Models Trained: {', '.join(raw.get('models_trained', []))}")
    
    # Available Images
    print("\n🖼️  GENERATED IMAGES")
    print("-" * 30)
    image_files = list(outputs_dir.glob("*.png"))
    if image_files:
        for img_file in image_files:
            size_kb = img_file.stat().st_size / 1024
            print(f"  📷 {img_file.name} ({size_kb:.1f} KB)")
    else:
        print("  No image files found")

def main():
    """Main function"""
    try:
        view_results_cli()
    except KeyboardInterrupt:
        print("\n\n👋 Viewer closed by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
