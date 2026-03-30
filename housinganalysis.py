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
    PLOTS_DIR = Path(__file__).parent / 'plots'
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

    def _save_plot(self, name: str) -> Path:
        """Save the current matplotlib figure to a numbered PNG file and close it.

        The filename is constructed from the instance-level ``plot_counter`` (zero-
        padded to two digits) and the supplied *name*, e.g. ``03_correlations.png``.
        The counter is incremented before each save so that files sort in the order
        they were produced.

        Args:
            name: Descriptive suffix used in the filename (spaces are allowed but
                underscores are conventional).

        Returns:
            The :class:`~pathlib.Path` of the saved file.

        Example:
            >>> analyzer = HousingAnalyzer()
            >>> import matplotlib.pyplot as plt
            >>> plt.figure()
            >>> plt.plot([1, 2], [3, 4])
            >>> path = analyzer._save_plot('my_chart')
            >>> path.suffix
            '.png'
        """
        self.plot_counter += 1
        filename = self.PLOTS_DIR / f"{self.plot_counter:02d}_{name}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {filename}")
        return filename

    def load_data(self) -> None:
        """Load housing data from a CSV file path or the remote GitHub URL.

        When ``self.data_path`` is set the local file is read; otherwise the
        class-level ``DATA_URL`` constant is used.  The resulting DataFrame is
        stored on ``self.full_data``.

        Raises:
            FileNotFoundError: If ``data_path`` is set but the file does not exist.
            ValueError: If the CSV cannot be parsed as a valid DataFrame.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data()
            Loaded 2930 records with 82 features
        """
        source = self.data_path if self.data_path else self.DATA_URL
        self.full_data = pd.read_csv(source)
        print(
            f"Loaded {len(self.full_data)} records with {len(self.full_data.columns)} features"
        )

    def split_data(self) -> None:
        """Split ``self.full_data`` into stratified training and held-out test sets.

        Uses ``RANDOM_STATE`` and ``TEST_SIZE`` class constants so results are
        reproducible.  The resulting subsets are assigned to ``self.training_data``
        and ``self.test_data`` respectively.

        Raises:
            ValueError: If :meth:`load_data` has not been called yet (i.e.
                ``self.full_data`` is ``None``).

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data()
            >>> analyzer.split_data()
            Training set: 2344, Test set: 586
        """
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
        """Drop columns whose missing-value rate exceeds ``NA_THRESHOLD`` percent.

        The same columns are removed from both ``self.training_data`` and
        ``self.full_data`` (if present) so that the two DataFrames stay in sync.
        Columns are identified using the *training* set only to avoid leaking test
        information.

        Returns:
            A :class:`~pandas.Series` mapping each dropped column name to its NA
            percentage, sorted descending.

        Raises:
            ValueError: If :meth:`split_data` has not been called yet (i.e.
                ``self.training_data`` is ``None``).

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data()
            >>> analyzer.split_data()
            >>> dropped = analyzer.remove_high_na_columns()
            >>> list(dropped.index)  # doctest: +SKIP
            ['Pool_QC', 'Misc_Feature', 'Alley', ...]
        """
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
        """Plot the number of houses sold in each calendar year as a bar chart.

        Groups ``self.training_data`` by the ``Yr_Sold`` column and counts
        observations per year.  The resulting figure is saved via
        :meth:`_save_plot`.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data()
            >>> analyzer.split_data()
            >>> analyzer.analyze_sales_by_year()  # doctest: +SKIP
        """
        self.training_data.groupby('Yr_Sold').count()['Order'].plot(kind='bar')
        plt.title('Houses Sold by Year')
        plt.xlabel('Year')
        plt.ylabel('Count')
        self._save_plot('sales_by_year')

    def analyze_living_area_vs_price(self) -> None:
        """Create a seaborn joint plot of above-grade living area vs. sale price.

        Displays a scatter plot with marginal histograms, saved via
        :meth:`_save_plot`.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data()
            >>> analyzer.split_data()
            >>> analyzer.analyze_living_area_vs_price()  # doctest: +SKIP
        """
        sns.jointplot(x='Gr_Liv_Area', y='SalePrice', data=self.training_data)
        self._save_plot('living_area_vs_price')

    def get_price_statistics(self) -> Tuple[float, float, Tuple[float, float]]:
        """Compute descriptive statistics for the ``SalePrice`` column.

        Calculates the mean, standard deviation, and the symmetric ±2 SD range
        from ``self.training_data`` and prints a formatted summary.

        Returns:
            A three-element tuple ``(mean_price, std_price, price_range)`` where
            ``price_range`` is itself a ``(lower, upper)`` float tuple representing
            the ±2 standard-deviation band around the mean.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data()
            >>> analyzer.split_data()
            >>> mean, std, (lo, hi) = analyzer.get_price_statistics()
            >>> mean > 0
            True
        """
        mean_price = np.mean(self.training_data['SalePrice'])
        std_price = np.std(self.training_data['SalePrice'])
        price_range = (mean_price - 2 * std_price, mean_price + 2 * std_price)

        print(f'Mean Sales Price: ${mean_price:,.2f}')
        print(f'STD Sales Price: ${std_price:,.2f}')
        print(f'2 SD Range: ${price_range[0]:,.2f} - ${price_range[1]:,.2f}')
        return mean_price, std_price, price_range

    def plot_price_distribution(self) -> None:
        """Plot the empirical distribution of sale prices as a histogram.

        Uses seaborn's ``histplot`` on ``self.training_data['SalePrice']`` and
        saves the figure via :meth:`_save_plot`.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data()
            >>> analyzer.split_data()
            >>> analyzer.plot_price_distribution()  # doctest: +SKIP
        """
        plt.figure(figsize=(10, 5))
        sns.histplot(data=self.training_data, x='SalePrice')
        plt.xlabel("Sales Price")
        plt.ylabel("Frequency")
        plt.title("Distribution of Sale Prices")
        self._save_plot('price_distribution')

    @staticmethod
    def remove_outliers(
            data: pd.DataFrame, variable: str, upper: float) -> pd.DataFrame:
        """Return a copy of *data* with rows where *variable* >= *upper* removed.

        Args:
            data: Source DataFrame; must contain *variable* as a column.
            variable: Name of the numeric column used for the outlier filter.
            upper: Exclusive upper bound — rows with ``data[variable] >= upper``
                are dropped.

        Returns:
            A filtered copy of *data* (original is not modified).

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'x': [100, 5000, 200], 'y': [1, 2, 3]})
            >>> HousingAnalyzer.remove_outliers(df, 'x', 4000)
               x  y
            0  100  1
            2  200  3
        """
        return data.loc[data[variable] < upper, :].copy()

    def remove_living_area_outliers(self) -> None:
        """Remove rows from ``self.training_data`` where ``Gr_Liv_Area`` exceeds the threshold.

        Uses ``OUTLIER_THRESHOLD`` (default 4000 sq ft) as the exclusive upper bound.
        Prints the number of rows removed along with their living area and sale price.
        Only the training split is modified; the test set is left untouched.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data()
            >>> analyzer.split_data()
            >>> analyzer.remove_living_area_outliers()  # doctest: +SKIP
            Removing 2 outliers with Gr_Liv_Area > 4000
        """
        outliers = self.training_data.loc[self.training_data['Gr_Liv_Area'] >
                                          self.OUTLIER_THRESHOLD,
                                          ['Gr_Liv_Area', 'SalePrice']]
        print(
            f"Removing {len(outliers)} outliers with Gr_Liv_Area > {self.OUTLIER_THRESHOLD}"
        )
        self.training_data = self.remove_outliers(
            self.training_data, 'Gr_Liv_Area', self.OUTLIER_THRESHOLD)

    def plot_neighborhood_distribution(self) -> None:
        """Plot the count of training-set houses per neighborhood as a bar chart.

        Neighborhoods are sorted by frequency in descending order.  The figure is
        saved via :meth:`_save_plot`.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data()
            >>> analyzer.split_data()
            >>> analyzer.plot_neighborhood_distribution()  # doctest: +SKIP
        """
        self.training_data.groupby('Neighborhood').size().sort_values(
            ascending=False).plot(kind='bar')
        plt.title('Houses by Neighborhood')
        plt.xlabel('Neighborhood')
        plt.ylabel('Count')
        plt.tight_layout()
        self._save_plot('neighborhood_distribution')

    def get_numeric_columns(self, data: pd.DataFrame) -> pd.Index:
        """Return the column names in *data* that have an ``int64`` or ``float64`` dtype.

        Args:
            data: DataFrame to inspect.

        Returns:
            A :class:`~pandas.Index` of column names whose dtype is either
            ``int64`` or ``float64``.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y'], 'c': [1.5, 2.5]})
            >>> analyzer = HousingAnalyzer()
            >>> list(analyzer.get_numeric_columns(df))
            ['a', 'c']
        """
        return data.dtypes[(data.dtypes == 'int64') |
                           (data.dtypes == 'float64')].index

    def analyze_correlations(self) -> pd.Series:
        """Compute and plot Pearson correlations between numeric features and ``SalePrice``.

        Only numeric (int64 / float64) columns are considered.  A horizontal bar chart
        of all feature correlations is saved via :meth:`_save_plot`.

        Returns:
            A :class:`~pandas.Series` of Pearson correlation coefficients with
            ``SalePrice``, sorted in descending order, with ``SalePrice`` itself
            excluded.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data()
            >>> analyzer.split_data()
            >>> corr = analyzer.analyze_correlations()
            >>> corr.index[0]  # highest-correlated feature  # doctest: +SKIP
            'Overall_Qual'
        """
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
        """Scatter-plot a single feature against ``SalePrice``, with optional jitter.

        Jitter (Gaussian noise with ``std=0.5``) is added to the feature axis to
        reveal overlapping points when the column contains discrete values.

        Args:
            feature: Column name in ``self.training_data`` to plot on the x-axis.
            add_jitter: When ``True`` (default), add Gaussian noise to the feature
                values before plotting to separate overlapping points.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data()
            >>> analyzer.split_data()
            >>> analyzer.plot_feature_vs_price('Overall_Qual')  # doctest: +SKIP
        """
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
        """Add a ``TotalBathrooms`` column that combines full and half bathrooms.

        Full bathrooms (basement and above-grade) count as 1.0 each; half
        bathrooms count as 0.5.  Missing values in any bathroom column are
        treated as 0 before summing.

        Args:
            data: DataFrame containing ``Bsmt_Full_Bath``, ``Full_Bath``,
                ``Bsmt_Half_Bath``, and ``Half_Bath`` columns.

        Returns:
            A copy of *data* with the ``TotalBathrooms`` column appended.

        Example:
            >>> import pandas as pd
            >>> row = {'Bsmt_Full_Bath': 1, 'Full_Bath': 1,
            ...        'Bsmt_Half_Bath': 0, 'Half_Bath': 1}
            >>> df = pd.DataFrame([row])
            >>> HousingAnalyzer.add_total_bathrooms(df)['TotalBathrooms'].iloc[0]
            2.5
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
        """Add a ``Total_SF`` column equal to basement plus above-grade living area.

        Args:
            data: DataFrame containing ``Total_Bsmt_SF`` and ``Gr_Liv_Area``
                columns.

        Returns:
            A copy of *data* with the ``Total_SF`` column appended.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'Total_Bsmt_SF': [800], 'Gr_Liv_Area': [1200]})
            >>> HousingAnalyzer.add_total_sf(df)['Total_SF'].iloc[0]
            2000
        """
        result = data.copy()
        result['Total_SF'] = result['Total_Bsmt_SF'] + result['Gr_Liv_Area']
        return result

    @staticmethod
    def find_rich_neighborhoods(data: pd.DataFrame, n: int = 3) -> List[str]:
        """Return the names of the top *n* neighborhoods by mean ``SalePrice``.

        Args:
            data: DataFrame containing ``Neighborhood`` and ``SalePrice`` columns.
            n: Number of top neighborhoods to return (default ``3``).

        Returns:
            A list of *n* neighborhood name strings sorted by descending mean
            sale price.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'Neighborhood': ['A', 'A', 'B', 'B', 'C'],
            ...     'SalePrice':    [300, 400, 100, 150, 500],
            ... })
            >>> HousingAnalyzer.find_rich_neighborhoods(df, n=2)
            ['C', 'A']
        """
        return data.groupby('Neighborhood')['SalePrice'].mean().sort_values(
            ascending=False).iloc[:n].index.tolist()

    @staticmethod
    def add_rich_neighborhood_flag(
            data: pd.DataFrame, neighborhoods: List[str]) -> pd.DataFrame:
        """Add a binary ``in_rich_neighborhood`` indicator column.

        Args:
            data: DataFrame containing a ``Neighborhood`` column.
            neighborhoods: List of neighborhood names considered "rich"
                (typically produced by :meth:`find_rich_neighborhoods`).

        Returns:
            A copy of *data* with ``in_rich_neighborhood`` appended — ``1`` when
            the row's neighborhood is in *neighborhoods*, ``0`` otherwise.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'Neighborhood': ['NridgHt', 'OldTown', 'NridgHt']})
            >>> result = HousingAnalyzer.add_rich_neighborhood_flag(df, ['NridgHt'])
            >>> list(result['in_rich_neighborhood'])
            [1, 0, 1]
        """
        result = data.copy()
        result['in_rich_neighborhood'] = result['Neighborhood'].isin(
            neighborhoods).astype(int)
        return result

    def plot_rich_neighborhoods(self, n: int = 20) -> None:
        """Plot the top *n* neighborhoods ranked by average sale price.

        Args:
            n: Number of neighborhoods to include in the chart (default ``20``).

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data()
            >>> analyzer.split_data()
            >>> analyzer.plot_rich_neighborhoods(n=10)  # doctest: +SKIP
        """
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
        """Label-encode every ``object``-dtype column in *data*.

        In **training mode** (``fitted_encoders=None``) a fresh
        :class:`~sklearn.preprocessing.LabelEncoder` is fitted for each
        categorical column and the fitted encoders are returned so they can be
        reused on the test set.

        In **inference/test mode** (``fitted_encoders`` supplied) the existing
        encoders are used to transform the column values.  Any value that was not
        seen during training is mapped to the first known class
        (``encoder.classes_[0]``) to avoid ``ValueError``.

        Args:
            data: DataFrame whose ``object``-typed columns will be encoded in
                place on a copy.
            fitted_encoders: Dict mapping column name to a pre-fitted
                :class:`~sklearn.preprocessing.LabelEncoder`.  Pass ``None`` to
                fit new encoders (training mode).

        Returns:
            A tuple ``(encoded_df, encoders)`` where *encoded_df* is a copy of
            *data* with all categorical columns replaced by integer codes, and
            *encoders* is the dict of fitted
            :class:`~sklearn.preprocessing.LabelEncoder` instances (same object
            as *fitted_encoders* when in test mode).

        Raises:
            KeyError: If *fitted_encoders* is provided but does not contain an
                encoder for a categorical column present in *data*.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'color': ['red', 'blue', 'red']})
            >>> encoded, encoders = HousingAnalyzer.encode_categorical(df)
            >>> list(encoded['color'])
            [1, 0, 1]
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
        """Apply the full feature engineering pipeline to ``self.training_data`` in place.

        The following transformations are applied sequentially:

        1. Add ``TotalBathrooms`` via :meth:`add_total_bathrooms`.
        2. Add ``Total_SF`` via :meth:`add_total_sf`.
        3. Derive ``self.rich_neighborhoods`` (top-4 by avg price) via
           :meth:`find_rich_neighborhoods` and add ``in_rich_neighborhood`` flag.
        4. Label-encode all ``object``-dtype columns via :meth:`encode_categorical`;
           the fitted encoders are stored on ``self.label_encoders``.

        Note:
            This method mutates ``self.training_data`` and sets
            ``self.rich_neighborhoods`` and ``self.label_encoders``.  It must be
            called *after* :meth:`remove_high_na_columns` and *before*
            :meth:`run_all_models`.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data()
            >>> analyzer.split_data()
            >>> analyzer.apply_feature_engineering()
            Feature engineering completed
        """
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
        """Separate ``self.training_data`` into a feature matrix *X* and target *y*.

        Numeric columns in *X* have their missing values filled with the column
        mean; any remaining NaNs are then filled forward then backward.

        Returns:
            A tuple ``(X, y)`` where *X* is a :class:`~pandas.DataFrame` of all
            columns except ``SalePrice`` and *y* is the ``SalePrice``
            :class:`~pandas.Series`.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data()
            >>> analyzer.split_data()
            >>> analyzer.apply_feature_engineering()
            >>> X, y = analyzer.prepare_features_target()
            >>> 'SalePrice' not in X.columns
            True
        """
        X = self.training_data.drop(columns=['SalePrice'])
        y = self.training_data['SalePrice']

        num_cols = self.get_numeric_columns(X)
        for col in num_cols:
            X[col] = X[col].fillna(X[col].mean())

        X = X.ffill().bfill()
        return X, y

    def compute_feature_importance_rf(
            self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """Fit a Random Forest and compute feature importances.

        Trains a 100-tree :class:`~sklearn.ensemble.RandomForestRegressor` on
        (*X*, *y*), records the resulting importances, and saves a bar-chart via
        :meth:`_save_plot`.  The sorted index array and corresponding column name
        list are stored in ``self.feature_indices['rf']`` and
        ``self.feature_names['rf']`` respectively.

        Args:
            X: Feature matrix (should already be fully numeric / encoded).
            y: Target vector (``SalePrice``).

        Returns:
            A 1-D :class:`~numpy.ndarray` of column indices sorted by descending
            feature importance.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data(); analyzer.split_data()
            >>> analyzer.apply_feature_engineering()
            >>> X, y = analyzer.prepare_features_target()
            >>> idx = analyzer.compute_feature_importance_rf(X, y)
            >>> idx.shape[0] == X.shape[1]
            True
        """
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
        """Fit a Gradient Boosting regressor and compute feature importances.

        Trains a 100-estimator
        :class:`~sklearn.ensemble.GradientBoostingRegressor` on (*X*, *y*),
        records the resulting importances, and saves a bar-chart via
        :meth:`_save_plot`.  Results are stored in ``self.feature_indices['gb']``
        and ``self.feature_names['gb']``.

        Args:
            X: Feature matrix (should already be fully numeric / encoded).
            y: Target vector (``SalePrice``).

        Returns:
            A 1-D :class:`~numpy.ndarray` of column indices sorted by descending
            feature importance.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data(); analyzer.split_data()
            >>> analyzer.apply_feature_engineering()
            >>> X, y = analyzer.prepare_features_target()
            >>> idx = analyzer.compute_feature_importance_gb(X, y)
            >>> idx.shape[0] == X.shape[1]
            True
        """
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
        """Identify the feature count that minimises cross-validated RMSE.

        Iterates from 1 to ``min(max_features, len(feature_idx))`` features
        (selected in *feature_idx* order), fits a plain
        :class:`~sklearn.linear_model.LinearRegression` with *cv*-fold cross-
        validation at each size, and returns the count that achieves the lowest
        mean CV RMSE.  A learning-curve figure is saved via :meth:`_save_plot`.
        The result is stored in ``self.optimal_features[method_name]``.

        Args:
            X: Feature matrix; columns are referenced by position via
                *feature_idx*.
            y: Target vector.
            feature_idx: 1-D array of column indices sorted by descending
                importance (output of :meth:`compute_feature_importance_rf` or
                similar).
            method_name: Human-readable label used in plot title and as the key
                in ``self.optimal_features``.
            max_features: Hard upper limit on the number of features to evaluate
                (default ``79``).
            cv: Number of cross-validation folds (default ``5``).

        Returns:
            The optimal number of features as an integer.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data(); analyzer.split_data()
            >>> analyzer.apply_feature_engineering()
            >>> X, y = analyzer.prepare_features_target()
            >>> idx = analyzer.compute_feature_importance_rf(X, y)
            >>> n = analyzer.find_optimal_features(X, y, idx, 'RF', max_features=10)
            >>> 1 <= n <= 10
            True
        """
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
        """Rank features by absolute Pearson correlation with ``SalePrice``.

        Computes the full correlation matrix for all numeric columns in
        ``self.training_data`` and orders features by their correlation with
        ``SalePrice`` (descending).  The resulting index array (with ``SalePrice``
        prepended at position 0) is stored in ``self.feature_indices['corr']``
        and ``self.feature_names['corr']``.

        Returns:
            A 1-D :class:`~numpy.ndarray` of integer column indices into
            ``self.training_data``, ordered highest-to-lowest correlation with
            ``SalePrice`` (``SalePrice`` itself at index 0).

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data(); analyzer.split_data()
            >>> analyzer.apply_feature_engineering()
            >>> idx = analyzer.compute_correlation_feature_indices()
            >>> analyzer.feature_names['corr'][0]
            'SalePrice'
        """
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
        """Plot a correlation heatmap for the top *n_features* numeric features.

        Features are ranked by their correlation with ``SalePrice``.  The heatmap
        uses the ``coolwarm`` palette and annotates each cell with the rounded
        coefficient.  The figure is saved via :meth:`_save_plot`.

        Args:
            n_features: Number of top-correlated features to include in the
                heatmap (default ``20``).

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data(); analyzer.split_data()
            >>> analyzer.apply_feature_engineering()
            >>> analyzer.plot_multicollinearity(n_features=10)  # doctest: +SKIP
        """
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
            data: Raw DataFrame containing all feature and target columns.
            feature_names: Ordered list of column names to keep (``SalePrice``
                will be appended automatically if missing).  Columns not present
                in *data* are silently ignored.
            fitted_scaler: Pre-fitted :class:`~sklearn.preprocessing.StandardScaler`
                for numeric columns.  Pass ``None`` to fit a new scaler on
                *data* (training mode).
            fitted_encoders: Dict of pre-fitted
                :class:`~sklearn.preprocessing.LabelEncoder` instances keyed by
                column name.  Pass ``None`` to fit new encoders (training mode).

        Returns:
            A four-element tuple ``(X, y, scaler, encoders)`` where:

            - *X* is the processed feature :class:`~pandas.DataFrame`.
            - *y* is the (optionally log-transformed) ``SalePrice``
              :class:`~pandas.Series`.
            - *scaler* is the fitted
              :class:`~sklearn.preprocessing.StandardScaler` (``None`` in test
              mode).
            - *encoders* is the dict of fitted encoders (``None`` in test mode).

        Raises:
            KeyError: If ``SalePrice`` is absent from *data* after feature
                selection.
            ValueError: If *data* is empty after outlier removal.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data(); analyzer.split_data()
            >>> analyzer.apply_feature_engineering()
            >>> feats = ['Overall_Qual', 'Gr_Liv_Area', 'SalePrice']
            >>> X, y, scaler, enc = analyzer.process_data_for_modeling(
            ...     analyzer.training_data.copy(), feats)
            >>> scaler is not None
            True
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
        """Fit an Ordinary Least Squares model and report performance metrics.

        When ``LOG_TARGET`` is ``True`` the predictions and labels are
        back-transformed via ``expm1`` before computing RMSE and R² so that
        all reported metrics are in original dollar units.  Two diagnostic plots
        (predicted vs. actual and residuals) are saved via :meth:`_save_plot`.

        Args:
            X_train: Scaled, encoded training feature matrix.
            y_train: Training target vector (log-transformed when
                ``LOG_TARGET`` is ``True``).
            X_test: Scaled, encoded test feature matrix.
            y_test: Test target vector (log-transformed when ``LOG_TARGET``
                is ``True``).

        Returns:
            A dict with keys ``'train_rmse'``, ``'test_rmse'``,
            ``'train_r2'``, and ``'test_r2'``, all in original dollar units.

        Example:
            >>> import pandas as pd, numpy as np
            >>> X = pd.DataFrame({'a': np.random.randn(100)})
            >>> y = pd.Series(np.random.randn(100))
            >>> analyzer = HousingAnalyzer()
            >>> analyzer.LOG_TARGET = False
            >>> res = analyzer.run_ols(X, y, X, y)
            >>> set(res.keys()) == {'train_rmse', 'test_rmse', 'train_r2', 'test_r2'}
            True
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
        """Fit a Ridge regression model with two-stage hyperparameter search.

        Runs a coarse ``GridSearchCV`` over eight candidate alpha values, then a
        fine-grained search over 200 values in the ±20 % band around the best
        coarse alpha.  Three diagnostic plots (predicted vs. actual, residuals,
        top-10 feature coefficients) are saved via :meth:`_save_plot`.

        Args:
            X_train: Scaled, encoded training feature matrix.
            y_train: Training target vector (log-transformed when
                ``LOG_TARGET`` is ``True``).
            X_test: Scaled, encoded test feature matrix.
            y_test: Test target vector (log-transformed when ``LOG_TARGET``
                is ``True``).

        Returns:
            A dict with keys ``'best_alpha'``, ``'train_rmse'``,
            ``'test_rmse'``, ``'train_r2'``, and ``'test_r2'``.

        Example:
            >>> import pandas as pd, numpy as np
            >>> X = pd.DataFrame({'a': np.random.randn(50), 'b': np.random.randn(50)})
            >>> y = pd.Series(np.random.randn(50))
            >>> analyzer = HousingAnalyzer()
            >>> analyzer.LOG_TARGET = False
            >>> res = analyzer.run_ridge(X, y, X, y)
            >>> 'best_alpha' in res
            True
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
        """Fit a LASSO regression model with two-stage hyperparameter search.

        Runs a coarse ``GridSearchCV`` over ten candidate alpha values, then a
        fine-grained search over 1000 values in the ±20 % band around the best
        coarse alpha.  Three diagnostic plots (predicted vs. actual, residuals,
        top-10 feature coefficients) are saved via :meth:`_save_plot`.

        Args:
            X_train: Scaled, encoded training feature matrix.
            y_train: Training target vector (log-transformed when
                ``LOG_TARGET`` is ``True``).
            X_test: Scaled, encoded test feature matrix.
            y_test: Test target vector (log-transformed when ``LOG_TARGET``
                is ``True``).

        Returns:
            A dict with keys ``'best_alpha'``, ``'train_rmse'``,
            ``'test_rmse'``, ``'train_r2'``, and ``'test_r2'``.

        Example:
            >>> import pandas as pd, numpy as np
            >>> X = pd.DataFrame({'a': np.random.randn(50), 'b': np.random.randn(50)})
            >>> y = pd.Series(np.random.randn(50))
            >>> analyzer = HousingAnalyzer()
            >>> analyzer.LOG_TARGET = False
            >>> res = analyzer.run_lasso(X, y, X, y)
            >>> 'best_alpha' in res
            True
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
        """Train and evaluate OLS, Ridge, and LASSO on a selected feature subset.

        Selects the top *num_features* features according to the pre-computed
        importance ranking stored under *method*, re-processes both train and test
        splits via :meth:`process_data_for_modeling` (fitting the scaler and
        encoders on the training split only to avoid leakage), then runs all three
        regression models.  A comparison summary table is printed to stdout.

        Args:
            method: Feature selection method key — one of ``'rf'`` (Random
                Forest), ``'gb'`` (Gradient Boosting), or ``'corr'``
                (correlation-based).
            num_features: Number of top-ranked features to use.

        Returns:
            A dict with keys ``'ols'``, ``'ridge'``, and ``'lasso'``, each
            mapping to the result dict returned by the corresponding ``run_*``
            method.

        Raises:
            ValueError: If *method* is not a key in ``self.feature_names``
                (i.e., feature importance for that method has not been computed
                yet).

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data(); analyzer.split_data()
            >>> analyzer.run_eda(); analyzer.run_feature_analysis()
            >>> results = analyzer.run_all_models('rf', num_features=13)
            >>> set(results.keys()) == {'ols', 'ridge', 'lasso'}
            True
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
        """Execute the full Exploratory Data Analysis pipeline.

        Runs the following steps in order, mutating ``self.training_data``:

        1. :meth:`remove_high_na_columns` — drop sparse columns.
        2. :meth:`analyze_sales_by_year` — bar chart of annual sales.
        3. :meth:`analyze_living_area_vs_price` — joint plot.
        4. :meth:`get_price_statistics` — print descriptive stats.
        5. :meth:`plot_price_distribution` — histogram.
        6. :meth:`remove_living_area_outliers` — drop extreme observations.
        7. :meth:`plot_neighborhood_distribution` — bar chart by neighborhood.
        8. :meth:`analyze_correlations` — correlation bar chart.
        9. :meth:`plot_feature_vs_price` for ``Bedroom_AbvGr`` and
           ``Overall_Qual``.

        Note:
            Must be called after :meth:`split_data`.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data(); analyzer.split_data()
            >>> analyzer.run_eda()  # doctest: +SKIP
        """
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
        """Execute the full feature-importance analysis pipeline.

        Calls the following steps in order:

        1. :meth:`apply_feature_engineering` — adds engineered columns and
           label-encodes categoricals on ``self.training_data``.
        2. :meth:`prepare_features_target` — splits into *X* and *y*.
        3. Random Forest importance via :meth:`compute_feature_importance_rf`
           followed by :meth:`find_optimal_features`.
        4. Gradient Boosting importance via :meth:`compute_feature_importance_gb`
           followed by :meth:`find_optimal_features`.
        5. Correlation-based ranking via
           :meth:`compute_correlation_feature_indices`.
        6. Multicollinearity heatmap via :meth:`plot_multicollinearity`.

        After this method completes, ``self.feature_indices``,
        ``self.feature_names``, and ``self.optimal_features`` are all populated
        and ready for use in :meth:`run_all_models`.

        Note:
            Must be called after :meth:`run_eda`.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> analyzer.load_data(); analyzer.split_data(); analyzer.run_eda()
            >>> analyzer.run_feature_analysis()  # doctest: +SKIP
        """
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

    def run_complete_analysis(self) -> Optional[Dict[str, Dict]]:
        """Run the end-to-end Ames Housing analysis pipeline.

        Executes all phases in order:

        1. :meth:`load_data` and :meth:`split_data`.
        2. :meth:`run_eda`.
        3. :meth:`run_feature_analysis`.
        4. :meth:`run_all_models` for each available feature selection method
           (``'rf'``, ``'gb'``, ``'corr'``), using optimal feature counts where
           available.

        Returns:
            A dict mapping each feature-selection method key (``'rf'``,
            ``'gb'``, ``'corr'``) to its :meth:`run_all_models` result dict, or
            ``None`` if no feature indices were computed.

        Example:
            >>> analyzer = HousingAnalyzer(data_path='ames.csv')
            >>> results = analyzer.run_complete_analysis()  # doctest: +SKIP
            >>> 'rf' in results
            True
        """
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
    """CLI entry point: instantiate :class:`HousingAnalyzer` and run the full pipeline.

    Calls :meth:`~HousingAnalyzer.run_complete_analysis` and re-raises any
    exception after printing a human-readable message.

    Example:
        Run from the command line::

            python housinganalysis.py
    """
    analyzer = HousingAnalyzer()

    try:
        analyzer.run_complete_analysis()
    except Exception as e:
        print(f"Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
