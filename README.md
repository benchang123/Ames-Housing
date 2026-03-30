# Ames Housing Price Analysis

A comprehensive machine learning project to predict housing prices in Ames, Iowa using multiple regression techniques and feature selection methods.

## Overview

This project analyzes the Ames Housing dataset to:
1. **Identify Key Price Drivers** - Determine which features most strongly correlate with housing prices
2. **Compare Feature Selection Methods** - Evaluate Random Forest, Gradient Boosting, and correlation-based approaches
3. **Build Predictive Models** - Compare OLS, Ridge, and LASSO regression performance

## Project Structure

```
Ames-Housing/
├── housinganalysis.py   # Main analysis script (HousingAnalyzer class)
├── ames.csv             # Housing dataset (2,930 observations, 82 variables)
├── codebook.txt         # Variable descriptions
├── plots/               # Output directory for generated visualizations
└── README.md
```

## Installation

### Requirements
- Python 3.7+
- Required packages:
  ```
  numpy
  pandas
  seaborn
  matplotlib
  scikit-learn
  statsmodels
  ```

### Setup
```bash
pip install numpy pandas seaborn matplotlib scikit-learn statsmodels
```

## Usage

### Command Line
```bash
python housinganalysis.py
```

### Run Complete Analysis
```python
from housinganalysis import HousingAnalyzer

analyzer = HousingAnalyzer()
analyzer.run_complete_analysis()
```

### Run Individual Steps
```python
analyzer = HousingAnalyzer()
analyzer.load_data()
analyzer.split_data()

# Exploratory Data Analysis
analyzer.run_eda()

# Feature Engineering & Importance
analyzer.run_feature_analysis()

# Run models with specific feature selection method
results = analyzer.run_all_models('rf', num_features=13)   # Random Forest features
results = analyzer.run_all_models('gb', num_features=20)   # Gradient Boosting features
results = analyzer.run_all_models('corr', num_features=20) # Correlation-based features
```

### Use a Local CSV
```python
analyzer = HousingAnalyzer(data_path='ames.csv')
```

## Analysis Pipeline

### 1. Exploratory Data Analysis (EDA)
- Remove columns with >25% missing values
- Identify and remove outliers (living area > 4000 sq ft)
- Analyze sale price distribution and statistics
- Visualize correlations between features and sale price

### 2. Feature Engineering
- **TotalBathrooms** - Combined full and half bathrooms (weighted)
- **Total_SF** - Total square footage (basement + living area)
- **in_rich_neighborhood** - Binary flag for top 4 neighborhoods by avg price
- Label encoding for all categorical variables

### 3. Feature Selection Methods
| Method | Description |
|--------|-------------|
| Random Forest (`rf`) | Feature importance from RF regressor |
| Gradient Boosting (`gb`) | Feature importance from GB regressor |
| Correlation (`corr`) | Pearson correlation with sale price |

Cross-validation (5-fold) determines the optimal feature count for each method.

### 4. Regression Models
| Model | Description |
|-------|-------------|
| **OLS** | Ordinary Least Squares baseline |
| **Ridge** | L2 regularization with two-stage tuned alpha |
| **LASSO** | L1 regularization with two-stage tuned alpha |

## Evaluation Metrics
- **RMSE** (Root Mean Squared Error) in original dollar units for training and test sets
- **R²** score for both training and test sets
- Log transform (`log1p`) applied to sale prices before modeling; predictions back-transformed via `expm1`

## Output

All plots are saved to `plots/` (relative to the script location) with sequential numeric prefixes, e.g. `01_sales_by_year.png`, `02_price_distribution.png`. Running partial pipelines continues numbering from where it left off.

## Data Description

The dataset contains 2,930 observations with 82 variables describing:
- **Location** - Neighborhood, zoning, lot configuration
- **Structure** - Building type, style, quality, condition
- **Size** - Square footage, rooms, bathrooms
- **Age** - Year built, remodel date
- **Features** - Garage, basement, fireplace, pool
- **Sale** - Price, type, condition

See `codebook.txt` for detailed variable descriptions.

## Key Findings

- **Overall Quality** and **Total Square Footage** are the strongest predictors
- Ridge and LASSO regularization reduce overfitting compared to OLS
- Feature selection method significantly impacts model performance
- Optimal feature count varies by method (RF: ~13, GB: ~20, correlation: ~20)

## Acknowledgments

Some ideas in this project are adapted from the [Data 100 Course](http://www.ds100.org/su20/) at UC Berkeley.

## License

This project is for educational and personal analysis purposes.
