#!/usr/bin/env python3
"""
Quick Start Script for Strat-e-GPT
Simple way to test the system with minimal setup
"""

import sys
from pathlib import Path

def quick_test():
    """Quick test of the system without full setup"""
    print("Strat-e-GPT Quick Test")
    print("=" * 40)
    
    # Check Python version
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("Please run this from the strat-e-gpt directory")
        return False
    
    # Check data directory
    data_dir = Path("data")
    if data_dir.exists():
        print("Data directory found")
        
        # Check for F1 data
        archive_dir = data_dir / "archive"
        datathon_dir = data_dir / "f1nalyze-datathon-ieeecsmuj"
        
        if archive_dir.exists():
            archive_files = list(archive_dir.glob("*.csv"))
            print(f"Found {len(archive_files)} archive data files")
        
        if datathon_dir.exists():
            datathon_files = list(datathon_dir.glob("*.csv"))
            print(f"Found {len(datathon_files)} datathon data files")
    else:
        print("Data directory not found - will create sample data")
    
    # Try to import key modules
    try:
        sys.path.append(str(Path.cwd() / "src"))
        from f1_data_analyzer import F1DataAnalyzer
        print("F1 Data Analyzer module imported successfully")
    except ImportError as e:
        print(f"Error importing F1 Data Analyzer: {e}")
        print("Try running: pip install -r requirements.txt")
        return False
    
    # Quick data analysis
    try:
        print("\nQuick Data Analysis...")
        analyzer = F1DataAnalyzer()
        
        # Load data
        archive_data = analyzer.load_archive_data()
        datathon_data = analyzer.load_datathon_data()
        
        print(f"Loaded {len(archive_data)} archive tables")
        print(f"Loaded {len(datathon_data)} datathon files")
        
        if archive_data:
            # Show sample of results data
            if 'results' in archive_data:
                results = archive_data['results']
                print(f"Results data: {results.shape[0]} rows, {results.shape[1]} columns")
                print(f" Year range: {results['raceId'].min()} - {results['raceId'].max()}")
        
        print("\n Quick test completed successfully!")
        print(" Ready to run: python main.py")
        
    except Exception as e:
        print(f" Error during quick test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = quick_test()
    if not success:
        print("\n For help, see USAGE_GUIDE.md or run: python setup_env.py")
        sys.exit(1)
