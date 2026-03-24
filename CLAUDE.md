# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

Install dependencies:
```bash
pip install numpy pandas seaborn matplotlib scikit-learn statsmodels
```

## Running the Analysis

Run the full pipeline from the command line:
```bash
python housinganalysis.py
```

Or run selectively from Python:
```python
from housinganalysis import HousingAnalyzer

analyzer = HousingAnalyzer()
analyzer.load_data()
analyzer.split_data()

analyzer.run_eda()                              # EDA only
analyzer.run_feature_analysis()                 # Feature importance only

# Run models with a specific feature selection method and feature count
results = analyzer.run_all_models('rf', num_features=13)   # Random Forest
results = analyzer.run_all_models('gb', num_features=20)   # Gradient Boosting
results = analyzer.run_all_models('corr', num_features=20) # Correlation-based
```

To use a local CSV instead of downloading from GitHub:
```python
analyzer = HousingAnalyzer(data_path='ames.csv')
```

## Architecture

The entire codebase is a single file (`housinganalysis.py`) organized around the `HousingAnalyzer` class. The analysis runs in three sequential phases, each of which mutates class state:

1. **EDA** (`run_eda`): Drops columns with >25% missing values (applied to both `training_data` and `full_data`), removes living area outliers (>4000 sq ft) from training only, and generates correlation/distribution plots.

2. **Feature Analysis** (`run_feature_analysis`): Calls `apply_feature_engineering` which mutates `self.training_data` in place, adding `TotalBathrooms`, `Total_SF`, `in_rich_neighborhood`, and label-encoding all categoricals. It then computes and stores feature importances in `self.feature_indices` (dict of numpy index arrays) and `self.feature_names` (dict of column name lists), keyed by `'rf'`, `'gb'`, and `'corr'`. Optimal feature counts are stored in `self.optimal_features`.

3. **Modeling** (`run_all_models`): Re-processes both train and test splits independently via `process_data_for_modeling` (outlier removal, feature engineering, StandardScaler on numerics, label encoding) using the feature names determined in phase 2. Ridge and LASSO both do a two-stage GridSearchCV: coarse search then fine-grained search in a ±20% band around the best coarse alpha.

**Important data flow detail**: `run_eda` mutates `self.training_data` by dropping high-NA columns and outlier rows. `run_feature_analysis` further mutates it with engineered features and encoding. `run_all_models` does NOT use this already-encoded `self.training_data` directly — it reloads a copy and re-applies feature engineering and scaling via `process_data_for_modeling`. The test set (`self.test_data`) is only touched during `run_all_models`.

**Plot output**: All plots are saved to the `plots/` directory with a sequential numeric prefix (e.g., `01_sales_by_year.png`). The counter increments globally across the session, so running partial pipelines will pick up numbering where it left off.

## Key Data Details

- Dataset: 2,930 observations, 82 variables (`ames.csv`)
- Target variable: `SalePrice`
- `rich_neighborhoods` is derived from top-4 neighborhoods by average training-set sale price and stored on the instance — it must be computed before `run_all_models` is called, since `process_data_for_modeling` uses `self.rich_neighborhoods` to apply the neighborhood flag to the test set.
- `codebook.txt` contains descriptions of all 82 variables.
