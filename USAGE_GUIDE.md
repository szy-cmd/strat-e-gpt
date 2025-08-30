# 🏁 Strat-e-GPT Usage Guide

## 🚀 **Complete Step-by-Step Instructions**

This guide will walk you through setting up and using Strat-e-GPT with your F1 racing data.

---

## 📋 **Prerequisites**

- **Python 3.8+** installed on your system
- **Git** for version control
- **At least 3GB free disk space** for data and models

---

## 🛠️ **Step 1: Environment Setup**

### **Option A: Automated Setup (Recommended)**
```bash
# Navigate to project directory
cd strat-e-gpt

# Run automated setup script
python setup_env.py
```

### **Option B: Manual Setup**
```bash
# Navigate to project directory
cd strat-e-gpt

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir data outputs models logs
```

---

## 🏎️ **Step 2: Understanding Your Data**

Your project contains **two types of F1 racing data**:

### **1. Historical F1 Archive Data** (`data/archive/`)
- **`results.csv`** - Race results, positions, points, lap times
- **`drivers.csv`** - Driver information, nationalities
- **`constructors.csv`** - Team information
- **`races.csv`** - Race metadata, years, circuits
- **`qualifying.csv`** - Qualifying session results
- **`pit_stops.csv`** - Pit stop timing data
- **`lap_times.csv`** - Individual lap times
- **`driver_standings.csv`** - Championship standings

### **2. Modern F1 Datathon Data** (`data/f1nalyze-datathon-ieeecsmuj/`)
- **`train.csv`** (1.0GB) - Training data for ML models
- **`test.csv`** (145MB) - Test data for predictions
- **`validation.csv` (141MB) - Validation data
- **`sample_submission.csv`** - Submission format

---

## 🚀 **Step 3: Running the Application**

### **Basic Usage**
```bash
# Make sure virtual environment is activated
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# Run the main application
python main.py
```

**What happens:**
1. ✅ Loads and analyzes your F1 data
2. 🔧 Creates ML features from historical data
3. 🤖 Trains multiple machine learning models
4. 📊 Generates performance visualizations
5. 💾 Saves results to `outputs/` directory

### **Expected Output:**
```
🏁 Starting Strat-e-GPT - Race Strategy Prediction System
📊 Initializing F1 data analysis pipeline...
🏎️ Analyzing F1 racing data...
📊 Data Summary:
  Archive tables: 9
  Datathon files: 4
🤖 ML Approach Suggestions:
  archive_data: Traditional ML with engineered features
    Target variables: ['position', 'points', 'laps']
    Recommended models: ['Random Forest', 'Gradient Boosting', 'XGBoost']
🔧 Creating racing features from historical data...
🤖 Training machine learning models...
📈 Creating visualizations...
✅ F1 analysis complete! Check the outputs/ directory for results.
```

---

## 📊 **Step 4: Exploring Results**

After running the application, check the `outputs/` directory:

### **Generated Files:**
- **`f1_model_performance.png`** - Model performance comparison
- **`f1_feature_importance.png`** - Feature importance analysis

### **Understanding the Results:**
1. **Model Performance**: Compare R² scores, RMSE, and MAE across different algorithms
2. **Feature Importance**: See which factors most influence race outcomes
3. **Predictions**: Understand how well the models predict finishing positions

---

## 🔬 **Step 5: Advanced Analysis with Jupyter Notebooks**

### **Launch Jupyter**
```bash
# Make sure you're in the project directory with venv activated
jupyter notebook
```

### **Available Notebooks:**
1. **`01_data_exploration.ipynb`** - Explore and understand your F1 data
2. **`02_ml_training.ipynb`** - Advanced ML model training and evaluation

### **Key Analysis Areas:**
- **Driver Performance Analysis**: Compare drivers across seasons
- **Constructor Performance**: Analyze team performance trends
- **Track Analysis**: Circuit-specific performance patterns
- **Seasonal Trends**: Year-over-year performance changes
- **Feature Engineering**: Create new predictive features

---

## 🎯 **Step 6: Customizing for Your Needs**

### **Modifying Target Variables**
Edit `src/f1_data_analyzer.py` to change what you're predicting:
```python
# Change from position prediction to points prediction
X, y = f1_analyzer.create_racing_features(target_column="points")
```

### **Adding New Features**
Extend the feature engineering in `create_racing_features()`:
```python
# Add new features like driver experience, constructor history, etc.
features['driver_experience'] = features['year'] - driver_debut_year
features['constructor_points'] = constructor_historical_points
```

### **Tuning ML Models**
Modify `src/ml_models.py` to adjust model parameters:
```python
# Customize Random Forest parameters
'random_forest': RandomForestRegressor(
    n_estimators=200,  # More trees
    max_depth=25,       # Deeper trees
    random_state=42
)
```

---

## 📈 **Step 7: Real-World Applications**

### **Race Strategy Prediction**
- **Qualifying Strategy**: Predict optimal qualifying performance
- **Race Pace**: Forecast race pace based on historical data
- **Pit Stop Timing**: Optimize pit stop strategies
- **Tire Management**: Predict tire wear and performance

### **Driver Development**
- **Performance Tracking**: Monitor driver improvement over time
- **Comparison Analysis**: Compare drivers across different conditions
- **Career Planning**: Identify optimal career paths

### **Team Strategy**
- **Constructor Performance**: Analyze team development
- **Resource Allocation**: Optimize team resources
- **Season Planning**: Plan for championship campaigns

---

## 🐛 **Troubleshooting**

### **Common Issues:**

1. **"Module not found" errors**
   ```bash
   # Make sure virtual environment is activated
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   
   # Reinstall requirements
   pip install -r requirements.txt
   ```

2. **Memory errors with large datasets**
   ```bash
   # Use smaller sample for testing
   # Edit main.py to load only subset of data initially
   ```

3. **Data loading errors**
   ```bash
   # Check file paths in data/ directory
   # Ensure CSV files are properly formatted
   ```

4. **Visualization errors**
   ```bash
   # Install additional dependencies
   pip install matplotlib seaborn plotly
   ```

### **Getting Help:**
- Check the logs in `logs/` directory
- Review error messages in the console output
- Ensure all dependencies are properly installed

---

## 🔮 **Next Steps & Advanced Features**

### **Immediate Enhancements:**
- [ ] Add more sophisticated feature engineering
- [ ] Implement time series analysis for lap times
- [ ] Create driver-constructor performance models
- [ ] Build real-time prediction APIs

### **Advanced ML Techniques:**
- [ ] Neural networks for complex pattern recognition
- [ ] Ensemble methods for improved accuracy
- [ ] Time series forecasting models
- [ ] Anomaly detection for unusual race events

### **Data Integration:**
- [ ] Real-time telemetry data
- [ ] Weather condition data
- [ ] Tire compound information
- [ ] Fuel strategy data

---

## 📚 **Learning Resources**

### **F1 Data Understanding:**
- [F1 Data Documentation](https://ergast.com/mrd/)
- [F1 Analytics Best Practices](https://www.f1analytics.com/)
- [Racing Data Science](https://racingdatascience.com/)

### **Machine Learning:**
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas User Guide](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorials](https://matplotlib.org/tutorials/)

---

## 🏆 **Success Metrics**

### **Model Performance Targets:**
- **R² Score**: > 0.7 (Good), > 0.8 (Excellent)
- **RMSE**: < 3 positions for finishing position prediction
- **Feature Importance**: Clear identification of key factors

### **Business Value:**
- **Strategy Improvement**: 10-20% better race outcomes
- **Decision Support**: Faster, data-driven decisions
- **Performance Insights**: Deeper understanding of racing dynamics

---

## 🎉 **Congratulations!**

You're now ready to revolutionize race strategy with machine learning! 

**Remember:**
- Start with the basic analysis (`python main.py`)
- Explore data with Jupyter notebooks
- Customize models for your specific needs
- Iterate and improve based on results

**Happy racing strategy analysis! 🏁🚀**
