#!/usr/bin/env python3
"""
Environment setup script for Strat-e-GPT
Automatically sets up the project environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f" {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f" {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f" {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f" Python {version.major}.{version.minor} detected. Python 3.8+ is required.")
        return False
    print(f"Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def create_directories():
    """Create necessary project directories"""
    directories = ['data', 'outputs', 'models', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f" Created directory: {directory}/")

def setup_virtual_environment():
    """Set up Python virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print(" Virtual environment already exists")
        return True
    
    print(" Creating virtual environment...")
    if run_command("python -m venv venv", "Creating virtual environment"):
        print(" Virtual environment created successfully")
        return True
    return False

def install_dependencies():
    """Install project dependencies"""
    # Determine the correct pip command based on OS
    if platform.system() == "Windows":
        pip_cmd = "venv\\Scripts\\pip"
    else:
        pip_cmd = "venv/bin/pip"
    
    # Upgrade pip first
    run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip")
    
    # Install requirements
    if run_command(f"{pip_cmd} install -r requirements.txt", "Installing dependencies"):
        print(" Dependencies installed successfully")
        return True
    return False

def create_sample_data():
    """Create sample data if it doesn't exist"""
    data_file = Path("data/sample_racing_data.csv")
    if not data_file.exists():
        print(" Creating sample racing data...")
        try:
            # Import and run the sample data creation
            sys.path.append(str(Path.cwd() / "src"))
            from main import create_sample_data
            create_sample_data(Path("data"))
            print(" Sample data created successfully")
        except Exception as e:
            print(f" Could not create sample data: {e}")
            print("You can run the main application to create sample data later")

def main():
    """Main setup function"""
    print(" Welcome to Strat-e-GPT Setup!")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print(" Setup cannot continue. Please upgrade Python to 3.8+")
        return False
    
    # Create directories
    create_directories()
    
    # Set up virtual environment
    if not setup_virtual_environment():
        print(" Failed to create virtual environment")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print(" Failed to install dependencies")
        return False
    
    # Create sample data
    create_sample_data()
    
    print("\n" + "=" * 50)
    print(" Setup completed successfully!")
    print("\n Next steps:")
    print("1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Run the application: python main.py")
    print("3. Open Jupyter notebooks: jupyter notebook")
    print("\n Happy racing strategy analysis!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
