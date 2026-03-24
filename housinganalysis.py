"""
Ames Housing Price Analysis

A comprehensive analysis of the Ames Housing dataset using multiple regression
techniques to predict housing prices and identify key price-driving features.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn import linear_model as lm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


class HousingAnalyzer:
    """A comprehensive analyzer for Ames Housing price prediction."""

    # Configuration constants
    DATA_URL = 'https://raw.githubusercontent.com/benchang123/Ames-Housing/master/ames.csv'
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    NA_THRESHOLD = 25  # Percentage threshold for dropping columns with NA
    OUTLIER_THRESHOLD = 4000  # Living area outlier threshold
    PLOTS_DIR = Path('plots')
    LOG_TARGET = True  # Apply log1p transform to SalePrice before modeling

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the HousingAnalyzer.
        
        Args:
            data_path: Optional path to local CSV file. Uses remote URL if None.
        """
        self.data_path = data_path
        self.full_data: Optional[pd.DataFrame] = None
        self.training_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None
        self.rich_neighborhoods: List[str] = []
        self.label_encoders: Dict[str, Any] = {}
        self.feature_indices: Dict[str, np.ndarray] = {}
        self.feature_names: Dict[str, List[str]] = {}
        self.optimal_features: Dict[str, int] = {}
        self.plot_counter: int = 0
        self.PLOTS_DIR.mkdir(exist_ok=True)

    def _save_plot(self, name: str) -> None:
        """Save current plot to file and close it."""
        self.plot_counter += 1
        filename = self.PLOTS_DIR / f"{self.plot_counter:02d}_{name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")

    def load_data(self) -> None:
        """Load housing data from CSV file or URL."""
        source = self.data_path if self.data_path else self.DATA_URL
        self.full_data = pd.read_csv(source)
        print(
            f"Loaded {len(self.full_data)} records with {len(self.full_data.columns)} features"
        )

    def split_data(self) -> None:
        """Split data into training and test sets."""
        if self.full_data is None:
            raise ValueError("Data must be loaded first. Call load_data()")
        self.training_data, self.test_data = train_test_split(
            self.full_data,
            random_state=self.RANDOM_STATE,
            test_size=self.TEST_SIZE)
        print(
            f"Training set: {len(self.training_data)}, Test set: {len(self.test_data)}"
        )

    # ==================== EDA Methods ====================

    def remove_high_na_columns(self) -> pd.Series:
        """Remove columns with NA percentage above threshold."""
        if self.training_data is None:
            raise ValueError("Data must be split first. Call split_data()")

        na_percent = self.training_data.isna().mean() * 100
        high_na = na_percent[na_percent > self.NA_THRESHOLD].sort_values(
            ascending=False)
        print(
            f"Removing {len(high_na)} columns with >{self.NA_THRESHOLD}% NA values:"
        )
        print(high_na)

        self.training_data.drop(columns=high_na.index, inplace=True)
        if self.full_data is not None:
            cols_to_drop = [
                c for c in high_na.index if c in self.full_data.columns
            ]
            self.full_data.drop(columns=cols_to_drop, inplace=True)
        return high_na

    def analyze_sales_by_year(self) -> None:
        """Plot number of houses sold per year."""
        self.training_data.groupby('Yr_Sold').count()['Order'].plot(kind='bar')
        plt.title('Houses Sold by Year')
        plt.xlabel('Year')
        plt.ylabel('Count')
        self._save_plot('sales_by_year')

    def analyze_living_area_vs_price(self) -> None:
        """Create joint plot of living area vs sale price."""
        sns.jointplot(x='Gr_Liv_Area', y='SalePrice', data=self.training_data)
        self._save_plot('living_area_vs_price')

    def get_price_statistics(self) -> Tuple[float, float, Tuple[float, float]]:
        """Calculate sale price statistics."""
        mean_price = np.mean(self.training_data['SalePrice'])
        std_price = np.std(self.training_data['SalePrice'])
        price_range = (mean_price - 2 * std_price, mean_price + 2 * std_price)

        print(f'Mean Sales Price: ${mean_price:,.2f}')
        print(f'STD Sales Price: ${std_price:,.2f}')
        print(f'2 SD Range: ${price_range[0]:,.2f} - ${price_range[1]:,.2f}')
        return mean_price, std_price, price_range

    def plot_price_distribution(self) -> None:
        """Plot sale price distribution."""
        plt.figure(figsize=(10, 5))
        sns.histplot(data=self.training_data, x='SalePrice')
        plt.xlabel("Sales Price")
        plt.ylabel("Frequency")
        plt.title("Distribution of Sale Prices")
        self._save_plot('price_distribution')

    @staticmethod
    def remove_outliers(
            data: pd.DataFrame, variable: str, upper: float) -> pd.DataFrame:
        """Remove outliers above a threshold."""
        return data.loc[data[variable] < upper, :].copy()

    def remove_living_area_outliers(self) -> None:
        """Remove living area outliers from training data."""
        outliers = self.training_data.loc[self.training_data['Gr_Liv_Area'] >
                                          self.OUTLIER_THRESHOLD,
                                          ['Gr_Liv_Area', 'SalePrice']]
        print(
            f"Removing {len(outliers)} outliers with Gr_Liv_Area > {self.OUTLIER_THRESHOLD}"
        )
        self.training_data = self.remove_outliers(
            self.training_data, 'Gr_Liv_Area', self.OUTLIER_THRESHOLD)

    def plot_neighborhood_distribution(self) -> None:
        """Plot neighborhood distribution."""
        self.training_data.groupby('Neighborhood').size().sort_values(
            ascending=False).plot(kind='bar')
        plt.title('Houses by Neighborhood')
        plt.xlabel('Neighborhood')
        plt.ylabel('Count')
        plt.tight_layout()
        self._save_plot('neighborhood_distribution')

    def get_numeric_columns(self, data: pd.DataFrame) -> pd.Index:
        """Get numeric column names from dataframe."""
        return data.dtypes[(data.dtypes == 'int64') |
                           (data.dtypes == 'float64')].index

    def analyze_correlations(self) -> pd.Series:
        """Analyze correlations with sale price."""
        num_cols = self.get_numeric_columns(self.training_data)
        corr_df = self.training_data.loc[:, num_cols].corr()
        sale_price_corr = corr_df['SalePrice'].drop('SalePrice').sort_values(
            ascending=False)

        plt.figure(figsize=(10, 15))
        sns.barplot(y=sale_price_corr.index, x=sale_price_corr.values)
        plt.xlabel("Correlation")
        plt.ylabel("Feature")
        plt.title("Feature Correlations with Sale Price")
        plt.tight_layout()
        self._save_plot('correlations')

        return sale_price_corr

    def plot_feature_vs_price(
            self, feature: str, add_jitter: bool = True) -> None:
        """Plot a feature against sale price with optional jitter."""
        data = self.training_data.copy()
        if add_jitter:
            noise = np.random.normal(0, 0.5, len(data))
            data[feature] = data[feature] + noise

        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=data, x=feature, y='SalePrice')
        plt.title(f'{feature} vs Sale Price')
        self._save_plot(f'feature_{feature}_vs_price')

    # ==================== Feature Engineering ====================

    @staticmethod
    def add_total_bathrooms(data: pd.DataFrame) -> pd.DataFrame:
        """
        Add total bathrooms feature combining full and half baths.
        
        Args:
            data: DataFrame with bathroom columns
            
        Returns:
            DataFrame with TotalBathrooms column added
        """
        result = data.copy()
        bath_vars = [
            'Bsmt_Full_Bath', 'Full_Bath', 'Bsmt_Half_Bath', 'Half_Bath'
        ]
        weights = pd.Series([1, 1, 0.5, 0.5], index=bath_vars)
        result['TotalBathrooms'] = result[bath_vars].fillna(0) @ weights
        return result

    @staticmethod
    def add_total_sf(data: pd.DataFrame) -> pd.DataFrame:
        """Add total square footage feature."""
        result = data.copy()
        result['Total_SF'] = result['Total_Bsmt_SF'] + result['Gr_Liv_Area']
        return result

    @staticmethod
    def find_rich_neighborhoods(data: pd.DataFrame, n: int = 3) -> List[str]:
        """Find top n neighborhoods by average sale price."""
        return data.groupby('Neighborhood')['SalePrice'].mean().sort_values(
            ascending=False).iloc[:n].index.tolist()

    @staticmethod
    def add_rich_neighborhood_flag(
            data: pd.DataFrame, neighborhoods: List[str]) -> pd.DataFrame:
        """Add binary flag for rich neighborhoods."""
        result = data.copy()
        result['in_rich_neighborhood'] = result['Neighborhood'].isin(
            neighborhoods).astype(int)
        return result

    def plot_rich_neighborhoods(self, n: int = 20) -> None:
        """Plot top neighborhoods by average sale price."""
        rich_hoods = self.training_data.groupby(
            'Neighborhood')['SalePrice'].mean().sort_values(
                ascending=False).iloc[:n]

        plt.figure(figsize=(8, 10))
        sns.barplot(y=rich_hoods.index, x=rich_hoods.values)
        plt.title(f'Top {n} Neighborhoods by Avg Sale Price')
        plt.xlabel('Average Sale Price')
        plt.tight_layout()
        self._save_plot('rich_neighborhoods')

    @staticmethod
    def encode_categorical(
            data: pd.DataFrame,
            fitted_encoders: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Label encode categorical columns.

        Args:
            data: DataFrame to encode
            fitted_encoders: Dict of pre-fitted LabelEncoders keyed by column
                name. When None, fits new encoders (training mode).

        Returns:
            Tuple of (encoded DataFrame, dict of fitted encoders)
        """
        result = data.copy()
        categorical_cols = result.select_dtypes(include=['object']).columns
        encoders: Dict[str, Any] = {} if fitted_encoders is None else fitted_encoders

        for col in categorical_cols:
            if fitted_encoders is None:
                encoder = preprocessing.LabelEncoder()
                result[col] = encoder.fit_transform(result[col].astype(str))
                encoders[col] = encoder
            else:
                encoder = fitted_encoders[col]
                # Handle unseen categories by mapping to the last known class
                known_classes = set(encoder.classes_)
                result[col] = result[col].astype(str).apply(
                    lambda x: x if x in known_classes else encoder.classes_[0]
                )
                result[col] = encoder.transform(result[col])
        return result, encoders

    def apply_feature_engineering(self) -> None:
        """Apply all feature engineering steps to training data."""
        self.training_data = self.add_total_bathrooms(self.training_data)
        self.training_data = self.add_total_sf(self.training_data)
        self.rich_neighborhoods = self.find_rich_neighborhoods(
            self.training_data, 4)
        self.training_data = self.add_rich_neighborhood_flag(
            self.training_data, self.rich_neighborhoods)
        self.training_data, self.label_encoders = self.encode_categorical(self.training_data)
        print("Feature engineering completed")

    # ==================== Feature Importance ====================

    def prepare_features_target(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix X and target vector y."""
        X = self.training_data.drop(columns=['SalePrice'])
        y = self.training_data['SalePrice']

        num_cols = self.get_numeric_columns(X)
        for col in num_cols:
            X[col] = X[col].fillna(X[col].mean())

        X = X.ffill().bfill()
        return X, y

    def compute_feature_importance_rf(
            self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute feature importance using Random Forest."""
        clf = RandomForestRegressor(
            n_estimators=100, random_state=self.RANDOM_STATE, n_jobs=-1)
        clf.fit(X, y)

        importances = clf.feature_importances_
        idx = np.argsort(importances)[::-1]

        plt.figure(figsize=(15, 10))
        sns.barplot(x=np.arange(len(idx)), y=importances[idx], color='black')
        plt.xticks(range(len(idx)), [X.columns[i] for i in idx], rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importance (Random Forest)')
        plt.tight_layout()
        self._save_plot('feature_importance_rf')

        self.feature_indices['rf'] = idx
        self.feature_names['rf'] = [X.columns[i] for i in idx]
        return idx

    def compute_feature_importance_gb(
            self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Compute feature importance using Gradient Boosting."""
        clf = GradientBoostingRegressor(
            n_estimators=100, random_state=self.RANDOM_STATE, verbose=0)
        clf.fit(X, y)

        importances = clf.feature_importances_
        idx = np.argsort(importances)[::-1]

        plt.figure(figsize=(15, 10))
        sns.barplot(x=np.arange(len(idx)), y=importances[idx], color='black')
        plt.xticks(range(len(idx)), [X.columns[i] for i in idx], rotation=90)
        plt.xlabel('Feature')
        plt.ylabel('Importance')
        plt.title('Feature Importance (Gradient Boosting)')
        plt.tight_layout()
        self._save_plot('feature_importance_gb')

        self.feature_indices['gb'] = idx
        self.feature_names['gb'] = [X.columns[i] for i in idx]
        return idx

    def find_optimal_features(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            feature_idx: np.ndarray,
            method_name: str,
            max_features: int = 79,
            cv: int = 5) -> int:
        """Find optimal number of features using cross-validation."""
        train_errors, cv_errors = [], []

        for i in range(1, min(max_features, len(feature_idx))):
            X_subset = X.iloc[:, feature_idx[:i]]
            model = lm.LinearRegression()

            cv_results = cross_validate(
                model,
                X_subset,
                y,
                cv=cv,
                scoring=('r2', 'neg_root_mean_squared_error'),
                return_train_score=True)

            train_errors.append(
                -np.mean(cv_results['train_neg_root_mean_squared_error']))
            cv_errors.append(
                -np.mean(cv_results['test_neg_root_mean_squared_error']))

        optimal = np.argmin(cv_errors) + 1

        plt.figure(figsize=(10, 7))
        plt.plot(
            range(1,
                  len(train_errors) + 1),
            train_errors,
            label="Training Error")
        plt.plot(range(1, len(cv_errors) + 1), cv_errors, label="CV Error")
        plt.axvline(x=optimal, color='green', linestyle='--', label=f'Optimal: {optimal}')
        plt.legend()
        plt.xlabel("Number of Features")
        plt.ylabel("RMSE")
        plt.title(f'Feature Selection ({method_name})')
        self._save_plot(
            f'feature_selection_{method_name.lower().replace(" ", "_")}')
        self.optimal_features[method_name] = optimal
        print(f"Optimal features for {method_name}: {optimal}")
        return optimal

    def compute_correlation_feature_indices(self) -> np.ndarray:
        """Compute feature indices based on correlation with sale price."""
        num_cols = self.get_numeric_columns(self.training_data)
        corr_df = self.training_data.loc[:, num_cols].corr()
        sale_price_corr = corr_df['SalePrice'].drop('SalePrice').sort_values(
            ascending=False)

        idx = []
        feature_names = []
        for feature in sale_price_corr.index:
            idx.append(self.training_data.columns.get_loc(feature))
            feature_names.append(feature)

        sale_price_idx = self.training_data.columns.get_loc('SalePrice')
        idx.insert(0, sale_price_idx)
        feature_names.insert(0, 'SalePrice')

        self.feature_indices['corr'] = np.array(idx)
        self.feature_names['corr'] = feature_names
        return np.array(idx)

    def plot_multicollinearity(self, n_features: int = 20) -> None:
        """Plot correlation heatmap for top features."""
        num_cols = self.get_numeric_columns(self.training_data)
        corr_df = self.training_data.loc[:, num_cols].corr()
        sale_price_corr = corr_df['SalePrice'].drop('SalePrice').sort_values(
            ascending=False)
        top_features = sale_price_corr.iloc[:n_features].index

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            self.training_data[top_features].corr(),
            annot=True,
            fmt='.2f',
            cmap='coolwarm')
        plt.title(f'Multicollinearity Check - Top {n_features} Features')
        plt.tight_layout()
        self._save_plot('multicollinearity')

    # ==================== Modeling ====================

    def process_data_for_modeling(
            self,
            data: pd.DataFrame,
            feature_names: List[str],
            fitted_scaler: Optional[Any] = None,
            fitted_encoders: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, Any, Optional[Dict[str, Any]]]:
        """
        Process data for modeling: remove outliers, engineer features, scale, and encode.

        In training mode (fitted_scaler=None, fitted_encoders=None) the scaler and
        encoders are fitted on the supplied data and returned so they can be reused
        on the test set, avoiding data leakage.

        Args:
            data: Raw dataframe
            feature_names: List of feature column names to select
            fitted_scaler: Pre-fitted StandardScaler (None = training mode)
            fitted_encoders: Pre-fitted LabelEncoder dict (None = training mode)

        Returns:
            Tuple of (X features, y target, scaler, encoders).
            In test mode the returned scaler and encoders are None.
        """
        data = self.remove_outliers(data, 'Gr_Liv_Area', self.OUTLIER_THRESHOLD)
        data = self.add_total_bathrooms(data)
        data = self.add_total_sf(data)
        data = self.add_rich_neighborhood_flag(data, self.rich_neighborhoods)

        # Ensure SalePrice is included
        if 'SalePrice' not in feature_names:
            feature_names = feature_names + ['SalePrice']

        # Select only columns that exist in the data
        available_features = [f for f in feature_names if f in data.columns]
        data = data[available_features]

        X = data.drop(['SalePrice'], axis=1)
        y = data['SalePrice']

        if self.LOG_TARGET:
            y = np.log1p(y)

        X = X.copy()
        if fitted_scaler is None:
            # Training mode: encode first so num_cols matches what the scaler will see.
            # If training_data was already encoded by apply_feature_engineering there are
            # no object columns left; fall back to the saved label_encoders in that case.
            X, encoders = self.encode_categorical(X, fitted_encoders=None)
            if not encoders and self.label_encoders:
                encoders = self.label_encoders
            num_cols = self.get_numeric_columns(X)
            X[num_cols] = X[num_cols].astype(float)
            scaler = preprocessing.StandardScaler()
            X[num_cols] = scaler.fit_transform(X[num_cols])
        else:
            # Test mode: encode first using the fitted encoders, then scale.
            effective_encoders = fitted_encoders if fitted_encoders else self.label_encoders
            X, _ = self.encode_categorical(X, fitted_encoders=effective_encoders)
            num_cols = self.get_numeric_columns(X)
            X[num_cols] = X[num_cols].astype(float)
            X[num_cols] = fitted_scaler.transform(X[num_cols])
            scaler = None
            encoders = None

        X = X.ffill().bfill()

        return X, y, scaler, encoders

    def run_ols(
            self, X_train: pd.DataFrame, y_train: pd.Series,
            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Run Ordinary Least Squares regression.

        Returns:
            Dictionary with training and test RMSE and R²
        """
        model = lm.LinearRegression()
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Back-transform if log target so RMSE is in original dollar units
        if self.LOG_TARGET:
            y_pred_train_orig = np.expm1(y_pred_train)
            y_train_orig = np.expm1(y_train)
            y_pred_test_orig = np.expm1(y_pred_test)
            y_test_orig = np.expm1(y_test)
        else:
            y_pred_train_orig = y_pred_train
            y_train_orig = y_train
            y_pred_test_orig = y_pred_test
            y_test_orig = y_test

        train_rmse = np.sqrt(metrics.mean_squared_error(y_train_orig, y_pred_train_orig))
        test_rmse = np.sqrt(metrics.mean_squared_error(y_test_orig, y_pred_test_orig))
        train_r2 = r2_score(y_train_orig, y_pred_train_orig)
        test_r2 = r2_score(y_test_orig, y_pred_test_orig)

        print(
            f'OLS - Training RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}'
        )
        print(f'OLS - Training R²: {train_r2:.4f}, Test R²: {test_r2:.4f}')

        price_min = min(y_train_orig.min(), y_test_orig.min())
        price_max = max(y_train_orig.max(), y_test_orig.max())

        # Prediction vs Actual plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].scatter(y_pred_train_orig, y_train_orig, alpha=0.5, label="Training")
        axes[0].scatter(y_pred_test_orig, y_test_orig, alpha=0.5, label="Test")
        axes[0].plot([price_min, price_max], [price_min, price_max], 'r-', label="Perfect Prediction")
        axes[0].set_xlabel('Predicted Sales Price')
        axes[0].set_ylabel('Actual Sales Price')
        axes[0].set_title('OLS: Predicted vs Actual')
        axes[0].legend()

        # Residual plot
        train_residuals = y_train_orig - y_pred_train_orig
        test_residuals = y_test_orig - y_pred_test_orig
        axes[1].scatter(
            y_pred_train_orig, train_residuals, alpha=0.5, label="Training")
        axes[1].scatter(y_pred_test_orig, test_residuals, alpha=0.5, label="Test")
        axes[1].axhline(y=0, color='r', linestyle='-')
        axes[1].set_xlabel('Predicted Sales Price')
        axes[1].set_ylabel('Residual')
        axes[1].set_title('OLS: Residual Plot')
        axes[1].legend()

        plt.tight_layout()
        self._save_plot('ols_results')

        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
        }

    def run_ridge(
            self, X_train: pd.DataFrame, y_train: pd.Series,
            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Run Ridge regression with hyperparameter tuning.
        
        Returns:
            Dictionary with best alpha, training and test RMSE
        """
        # Initial grid search
        param_grid = {'alpha': [0.01, 0.1, 1., 5., 10., 25., 50., 100.]}
        grid_search = GridSearchCV(
            lm.Ridge(),
            cv=5,
            param_grid=param_grid,
            scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        initial_alpha = grid_search.best_params_['alpha']
        print(f'Ridge - Initial Best Alpha: {initial_alpha}')

        # Fine-tuned grid search
        fine_grid = {
            'alpha':
                list(
                    np.linspace(initial_alpha * 0.8, initial_alpha * 1.2, 200))
        }
        fine_search = GridSearchCV(
            lm.Ridge(),
            cv=5,
            param_grid=fine_grid,
            scoring='neg_mean_squared_error')
        fine_search.fit(X_train, y_train)
        best_alpha = fine_search.best_params_['alpha']
        print(f'Ridge - Tuned Best Alpha: {best_alpha:.4f}')

        y_pred_train = fine_search.predict(X_train)
        y_pred_test = fine_search.predict(X_test)

        # Back-transform if log target so RMSE is in original dollar units
        if self.LOG_TARGET:
            y_pred_train_orig = np.expm1(y_pred_train)
            y_train_orig = np.expm1(y_train)
            y_pred_test_orig = np.expm1(y_pred_test)
            y_test_orig = np.expm1(y_test)
        else:
            y_pred_train_orig = y_pred_train
            y_train_orig = y_train
            y_pred_test_orig = y_pred_test
            y_test_orig = y_test

        train_rmse = np.sqrt(metrics.mean_squared_error(y_train_orig, y_pred_train_orig))
        test_rmse = np.sqrt(metrics.mean_squared_error(y_test_orig, y_pred_test_orig))
        train_r2 = r2_score(y_train_orig, y_pred_train_orig)
        test_r2 = r2_score(y_test_orig, y_pred_test_orig)
        print(
            f'Ridge - Training RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}'
        )
        print(f'Ridge - Training R²: {train_r2:.4f}, Test R²: {test_r2:.4f}')

        price_min = min(y_train_orig.min(), y_test_orig.min())
        price_max = max(y_train_orig.max(), y_test_orig.max())

        # Plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Prediction vs Actual
        axes[0].scatter(y_pred_train_orig, y_train_orig, alpha=0.5, label="Training")
        axes[0].scatter(y_pred_test_orig, y_test_orig, alpha=0.5, label="Test")
        axes[0].plot([price_min, price_max], [price_min, price_max], 'r-')
        axes[0].set_xlabel('Predicted Sales Price')
        axes[0].set_ylabel('Actual Sales Price')
        axes[0].set_title('Ridge: Predicted vs Actual')
        axes[0].legend()

        # Residuals
        axes[1].scatter(
            y_pred_train_orig, y_train_orig - y_pred_train_orig, alpha=0.5, label="Training")
        axes[1].scatter(
            y_pred_test_orig, y_test_orig - y_pred_test_orig, alpha=0.5, label="Test")
        axes[1].axhline(y=0, color='r', linestyle='-')
        axes[1].set_xlabel('Predicted Sales Price')
        axes[1].set_ylabel('Residual')
        axes[1].set_title('Ridge: Residual Plot')
        axes[1].legend()

        # Feature Importance
        coefs = pd.DataFrame(
            {
                'coef': fine_search.best_estimator_.coef_,
                'abs_coef': np.abs(fine_search.best_estimator_.coef_)
            },
            index=X_train.columns)
        top_coefs = coefs.nlargest(10, 'abs_coef')
        sns.barplot(x=top_coefs['abs_coef'], y=top_coefs.index, ax=axes[2])
        axes[2].set_title('Ridge: Top 10 Features')
        axes[2].set_xlabel('Absolute Coefficient')

        plt.tight_layout()
        self._save_plot('ridge_results')

        return {
            'best_alpha': best_alpha,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
        }

    def run_lasso(
            self, X_train: pd.DataFrame, y_train: pd.Series,
            X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Run LASSO regression with hyperparameter tuning.
        
        Returns:
            Dictionary with best alpha, training and test RMSE
        """
        # Initial grid search
        param_grid = {
            'alpha': [0.01, 0.1, 1., 5., 10., 25., 50., 100., 500., 1000.]
        }
        grid_search = GridSearchCV(
            lm.Lasso(),
            cv=5,
            param_grid=param_grid,
            scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        initial_alpha = grid_search.best_params_['alpha']
        print(f'LASSO - Initial Best Alpha: {initial_alpha}')

        # Fine-tuned grid search
        fine_grid = {
            'alpha':
                list(
                    np.linspace(initial_alpha * 0.8, initial_alpha * 1.2, 1000))
        }
        fine_search = GridSearchCV(
            lm.Lasso(),
            cv=5,
            param_grid=fine_grid,
            scoring='neg_mean_squared_error')
        fine_search.fit(X_train, y_train)
        best_alpha = fine_search.best_params_['alpha']
        print(f'LASSO - Tuned Best Alpha: {best_alpha:.4f}')

        y_pred_train = fine_search.predict(X_train)
        y_pred_test = fine_search.predict(X_test)

        # Back-transform if log target so RMSE is in original dollar units
        if self.LOG_TARGET:
            y_pred_train_orig = np.expm1(y_pred_train)
            y_train_orig = np.expm1(y_train)
            y_pred_test_orig = np.expm1(y_pred_test)
            y_test_orig = np.expm1(y_test)
        else:
            y_pred_train_orig = y_pred_train
            y_train_orig = y_train
            y_pred_test_orig = y_pred_test
            y_test_orig = y_test

        train_rmse = np.sqrt(metrics.mean_squared_error(y_train_orig, y_pred_train_orig))
        test_rmse = np.sqrt(metrics.mean_squared_error(y_test_orig, y_pred_test_orig))
        train_r2 = r2_score(y_train_orig, y_pred_train_orig)
        test_r2 = r2_score(y_test_orig, y_pred_test_orig)
        print(
            f'LASSO - Training RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}'
        )
        print(f'LASSO - Training R²: {train_r2:.4f}, Test R²: {test_r2:.4f}')

        price_min = min(y_train_orig.min(), y_test_orig.min())
        price_max = max(y_train_orig.max(), y_test_orig.max())

        # Plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Prediction vs Actual
        axes[0].scatter(y_pred_train_orig, y_train_orig, alpha=0.5, label="Training")
        axes[0].scatter(y_pred_test_orig, y_test_orig, alpha=0.5, label="Test")
        axes[0].plot([price_min, price_max], [price_min, price_max], 'r-')
        axes[0].set_xlabel('Predicted Sales Price')
        axes[0].set_ylabel('Actual Sales Price')
        axes[0].set_title('LASSO: Predicted vs Actual')
        axes[0].legend()

        # Residuals
        axes[1].scatter(
            y_pred_train_orig, y_train_orig - y_pred_train_orig, alpha=0.5, label="Training")
        axes[1].scatter(
            y_pred_test_orig, y_test_orig - y_pred_test_orig, alpha=0.5, label="Test")
        axes[1].axhline(y=0, color='r', linestyle='-')
        axes[1].set_xlabel('Predicted Sales Price')
        axes[1].set_ylabel('Residual')
        axes[1].set_title('LASSO: Residual Plot')
        axes[1].legend()

        # Feature Importance
        coefs = pd.DataFrame(
            {
                'coef': fine_search.best_estimator_.coef_,
                'abs_coef': np.abs(fine_search.best_estimator_.coef_)
            },
            index=X_train.columns)
        top_coefs = coefs.nlargest(10, 'abs_coef')
        sns.barplot(x=top_coefs['abs_coef'], y=top_coefs.index, ax=axes[2])
        axes[2].set_title('LASSO: Top 10 Features')
        axes[2].set_xlabel('Absolute Coefficient')

        plt.tight_layout()
        self._save_plot('lasso_results')

        return {
            'best_alpha': best_alpha,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
        }

    def run_all_models(self, method: str, num_features: int) -> Dict[str, Dict]:
        """
        Run all regression models (OLS, Ridge, LASSO) for a given feature selection method.
        
        Args:
            method: Feature selection method ('rf', 'gb', or 'corr')
            num_features: Number of top features to use
            
        Returns:
            Dictionary with results from all models
        """
        if method not in self.feature_names:
            raise ValueError(
                f"Feature names for '{method}' not computed. Run feature importance first."
            )

        # Use feature names instead of indices for consistent column selection
        selected_features = self.feature_names[method][:num_features]

        # Reload and prepare data — fit scaler/encoders on train only, then
        # apply the same fitted objects to test to avoid leakage.
        train_data = self.training_data.copy()
        test_data = self.test_data.copy()

        X_train, y_train, fitted_scaler, fitted_encoders = self.process_data_for_modeling(
            train_data, selected_features)
        X_test, y_test, _, _ = self.process_data_for_modeling(
            test_data, selected_features,
            fitted_scaler=fitted_scaler,
            fitted_encoders=fitted_encoders)

        # OLS Summary
        X_train_const = sm.add_constant(X_train)
        ols_model = sm.OLS(y_train, X_train_const)
        print(ols_model.fit().summary())

        print(f"\n{'='*60}")
        print(
            f"Running models with {method.upper()} feature selection ({num_features} features)"
        )
        print(f"{'='*60}\n")

        results = {
            'ols': self.run_ols(X_train, y_train, X_test, y_test),
            'ridge': self.run_ridge(X_train, y_train, X_test, y_test),
            'lasso': self.run_lasso(X_train, y_train, X_test, y_test)
        }

        # Model comparison summary table
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        header = f"{'Model':<12} {'Train RMSE':>12} {'Test RMSE':>12} {'Test R²':>10}"
        print(header)
        print('-' * len(header))
        for model_name, res in [('OLS', results['ols']),
                                 ('Ridge', results['ridge']),
                                 ('LASSO', results['lasso'])]:
            print(
                f"{model_name:<12} {res['train_rmse']:>12.2f} {res['test_rmse']:>12.2f} {res['test_r2']:>10.4f}"
            )
        print(f"{'='*60}\n")

        return results

    # ==================== Main Pipeline ====================

    def run_eda(self) -> None:
        """Run complete EDA pipeline."""
        print("\n" + "=" * 60)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 60 + "\n")

        self.remove_high_na_columns()
        self.analyze_sales_by_year()
        self.analyze_living_area_vs_price()
        self.get_price_statistics()
        self.plot_price_distribution()
        self.remove_living_area_outliers()
        self.plot_neighborhood_distribution()
        self.analyze_correlations()
        self.plot_feature_vs_price('Bedroom_AbvGr')
        self.plot_feature_vs_price('Overall_Qual')

    def run_feature_analysis(self) -> None:
        """Run complete feature importance analysis."""
        print("\n" + "=" * 60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60 + "\n")

        self.apply_feature_engineering()
        X, y = self.prepare_features_target()

        # Random Forest
        self.compute_feature_importance_rf(X, y)
        self.find_optimal_features(
            X, y, self.feature_indices['rf'], 'Random Forest')

        # Gradient Boosting
        self.compute_feature_importance_gb(X, y)
        self.find_optimal_features(
            X, y, self.feature_indices['gb'], 'Gradient Boosting')

        # Correlation-based
        self.compute_correlation_feature_indices()
        self.plot_multicollinearity()

    def run_complete_analysis(self) -> None:
        """Run the complete housing price analysis pipeline."""
        print("=" * 60)
        print("AMES HOUSING PRICE ANALYSIS")
        print("=" * 60)

        # Load and prepare data
        self.load_data()
        self.split_data()

        # EDA
        self.run_eda()

        # Feature Analysis
        self.run_feature_analysis()

        # Run models with different feature selection methods
        print("\n" + "=" * 60)
        print("MODEL TRAINING AND EVALUATION")
        print("=" * 60 + "\n")

        results = {}

        if 'rf' in self.feature_indices:
            num_features_rf = self.optimal_features.get('Random Forest', 13)
            results['rf'] = self.run_all_models('rf', num_features_rf)

        if 'gb' in self.feature_indices:
            num_features_gb = self.optimal_features.get('Gradient Boosting', 20)
            results['gb'] = self.run_all_models('gb', num_features_gb)

        if 'corr' in self.feature_indices:
            results['corr'] = self.run_all_models('corr', 20)

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)

        return results


def main() -> None:
    """Main entry point for the housing analysis."""
    analyzer = HousingAnalyzer()

    try:
        analyzer.run_complete_analysis()
    except Exception as e:
        print(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
