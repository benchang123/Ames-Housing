"""
Unit tests for HousingAnalyzer.

Run with:
    pytest tests/test_housinganalysis.py -v

All tests are self-contained: expensive I/O (pd.read_csv over HTTP) and
matplotlib rendering are mocked so the suite runs quickly without network
access and without leaving stray plot files on disk.
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make sure the repo root is importable regardless of how pytest is invoked.
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from housinganalysis import HousingAnalyzer  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_minimal_df(n: int = 40) -> pd.DataFrame:
    """Return a small but structurally valid Ames-like DataFrame.

    Contains enough columns to exercise every method under test without
    requiring the real dataset.
    """
    rng = np.random.default_rng(0)
    neighborhoods = ["NridgHt", "OldTown", "CollgCr", "Somerst", "Edwards"]
    df = pd.DataFrame(
        {
            "Order": range(1, n + 1),
            "Yr_Sold": rng.choice([2006, 2007, 2008, 2009, 2010], n),
            "Neighborhood": rng.choice(neighborhoods, n),
            "Gr_Liv_Area": rng.integers(800, 3000, n).astype(float),
            "Total_Bsmt_SF": rng.integers(400, 1500, n).astype(float),
            "Bsmt_Full_Bath": rng.integers(0, 2, n).astype(float),
            "Full_Bath": rng.integers(1, 3, n).astype(float),
            "Bsmt_Half_Bath": rng.integers(0, 1, n).astype(float),
            "Half_Bath": rng.integers(0, 2, n).astype(float),
            "Bedroom_AbvGr": rng.integers(1, 5, n).astype(float),
            "Overall_Qual": rng.integers(1, 10, n).astype(float),
            "SalePrice": rng.integers(100_000, 400_000, n).astype(float),
            # A categorical column to exercise encoding paths
            "House_Style": rng.choice(["1Story", "2Story", "1.5Fin"], n),
            # A column that will be kept (low NA)
            "Garage_Type": rng.choice(["Attchd", "Detchd", "None"], n),
        }
    )
    return df


def _make_high_na_df(n: int = 40) -> pd.DataFrame:
    """Extend the minimal DataFrame with a column that has >25 % NA values."""
    df = _make_minimal_df(n)
    # >25 % of rows are NaN in Pool_QC
    pool_qc = np.array(["Ex"] * n, dtype=object)
    pool_qc[: int(n * 0.9)] = None  # 90 % NaN
    df["Pool_QC"] = pool_qc
    return df


@pytest.fixture()
def analyzer_loaded(tmp_path: Path) -> HousingAnalyzer:
    """Return an analyzer with full_data already loaded from a local CSV."""
    df = _make_minimal_df(n=60)
    csv_path = tmp_path / "ames_test.csv"
    df.to_csv(csv_path, index=False)

    analyzer = HousingAnalyzer(data_path=str(csv_path))
    # Patch PLOTS_DIR so no files are written to the real repo directory
    analyzer.PLOTS_DIR = tmp_path / "plots"
    analyzer.PLOTS_DIR.mkdir()
    analyzer.load_data()
    return analyzer


@pytest.fixture()
def analyzer_split(analyzer_loaded: HousingAnalyzer) -> HousingAnalyzer:
    """Return an analyzer that has been loaded AND split."""
    analyzer_loaded.split_data()
    return analyzer_loaded


@pytest.fixture()
def analyzer_eda(analyzer_split: HousingAnalyzer) -> HousingAnalyzer:
    """Return an analyzer that has completed the EDA phase."""
    # Patch plt to avoid creating plot windows and writing files
    with patch("housinganalysis.plt"):
        with patch("housinganalysis.sns"):
            analyzer_split.remove_high_na_columns()
            analyzer_split.remove_living_area_outliers()
    return analyzer_split


@pytest.fixture()
def analyzer_engineered(analyzer_eda: HousingAnalyzer) -> HousingAnalyzer:
    """Return an analyzer that has had feature engineering applied."""
    with patch("housinganalysis.plt"):
        with patch("housinganalysis.sns"):
            analyzer_eda.apply_feature_engineering()
    return analyzer_eda


# ---------------------------------------------------------------------------
# Tests: load_data
# ---------------------------------------------------------------------------

class TestLoadData:
    def test_load_from_local_csv(self, tmp_path: Path) -> None:
        df = _make_minimal_df(n=20)
        csv_path = tmp_path / "sample.csv"
        df.to_csv(csv_path, index=False)

        analyzer = HousingAnalyzer(data_path=str(csv_path))
        analyzer.load_data()

        assert analyzer.full_data is not None
        assert len(analyzer.full_data) == 20
        assert "SalePrice" in analyzer.full_data.columns

    def test_load_from_url_uses_read_csv(self, tmp_path: Path) -> None:
        """When data_path is None, pd.read_csv should be called with DATA_URL."""
        fake_df = _make_minimal_df(n=10)

        with patch("housinganalysis.pd.read_csv", return_value=fake_df) as mock_csv:
            analyzer = HousingAnalyzer()
            analyzer.PLOTS_DIR = tmp_path / "plots"
            analyzer.PLOTS_DIR.mkdir()
            analyzer.load_data()

        mock_csv.assert_called_once_with(HousingAnalyzer.DATA_URL)
        assert analyzer.full_data is not None
        assert len(analyzer.full_data) == 10

    def test_full_data_column_count(self, tmp_path: Path) -> None:
        df = _make_minimal_df(n=15)
        csv_path = tmp_path / "c.csv"
        df.to_csv(csv_path, index=False)

        analyzer = HousingAnalyzer(data_path=str(csv_path))
        analyzer.load_data()

        assert analyzer.full_data.shape[1] == df.shape[1]

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        analyzer = HousingAnalyzer(data_path=str(tmp_path / "nonexistent.csv"))
        with pytest.raises(FileNotFoundError):
            analyzer.load_data()


# ---------------------------------------------------------------------------
# Tests: split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_raises_if_not_loaded(self, tmp_path: Path) -> None:
        analyzer = HousingAnalyzer()
        analyzer.PLOTS_DIR = tmp_path / "plots"
        analyzer.PLOTS_DIR.mkdir()
        with pytest.raises(ValueError, match="load_data"):
            analyzer.split_data()

    def test_sizes_match_test_size(self, analyzer_loaded: HousingAnalyzer) -> None:
        analyzer_loaded.split_data()
        total = len(analyzer_loaded.full_data)
        test_n = len(analyzer_loaded.test_data)
        train_n = len(analyzer_loaded.training_data)

        assert train_n + test_n == total
        expected_test = round(total * HousingAnalyzer.TEST_SIZE)
        # Allow off-by-one due to rounding
        assert abs(test_n - expected_test) <= 1

    def test_reproducibility(self, analyzer_loaded: HousingAnalyzer) -> None:
        """Two successive splits must produce the same index sets."""
        analyzer_loaded.split_data()
        train_idx_1 = set(analyzer_loaded.training_data.index)

        # Reload and split again
        analyzer_loaded.load_data()
        analyzer_loaded.split_data()
        train_idx_2 = set(analyzer_loaded.training_data.index)

        assert train_idx_1 == train_idx_2

    def test_no_overlap_between_train_test(
        self, analyzer_loaded: HousingAnalyzer
    ) -> None:
        analyzer_loaded.split_data()
        train_idx = set(analyzer_loaded.training_data.index)
        test_idx = set(analyzer_loaded.test_data.index)
        assert train_idx.isdisjoint(test_idx)


# ---------------------------------------------------------------------------
# Tests: remove_high_na_columns
# ---------------------------------------------------------------------------

class TestRemoveHighNaColumns:
    def test_raises_if_not_split(self, tmp_path: Path) -> None:
        analyzer = HousingAnalyzer()
        analyzer.PLOTS_DIR = tmp_path / "plots"
        analyzer.PLOTS_DIR.mkdir()
        with pytest.raises(ValueError, match="split_data"):
            analyzer.remove_high_na_columns()

    def test_drops_high_na_column(self, tmp_path: Path) -> None:
        df = _make_high_na_df(n=60)
        csv_path = tmp_path / "h.csv"
        df.to_csv(csv_path, index=False)

        analyzer = HousingAnalyzer(data_path=str(csv_path))
        analyzer.PLOTS_DIR = tmp_path / "plots"
        analyzer.PLOTS_DIR.mkdir()
        analyzer.load_data()
        analyzer.split_data()

        dropped = analyzer.remove_high_na_columns()

        assert "Pool_QC" in dropped.index
        assert "Pool_QC" not in analyzer.training_data.columns

    def test_low_na_columns_retained(self, analyzer_split: HousingAnalyzer) -> None:
        original_cols = set(analyzer_split.training_data.columns)
        # SalePrice has no NaN values so it must always survive the filter
        assert "SalePrice" in original_cols
        dropped = analyzer_split.remove_high_na_columns()
        remaining = set(analyzer_split.training_data.columns)
        # All remaining columns must have been present before the call
        assert remaining.issubset(original_cols)
        # Columns that were dropped must no longer appear in training_data
        for col in dropped.index:
            assert col not in remaining
        # SalePrice must never be dropped (it has no missing values)
        assert "SalePrice" in remaining

    def test_returns_series(self, analyzer_split: HousingAnalyzer) -> None:
        result = analyzer_split.remove_high_na_columns()
        assert isinstance(result, pd.Series)

    def test_full_data_also_updated(self, tmp_path: Path) -> None:
        df = _make_high_na_df(n=80)
        csv_path = tmp_path / "h2.csv"
        df.to_csv(csv_path, index=False)

        analyzer = HousingAnalyzer(data_path=str(csv_path))
        analyzer.PLOTS_DIR = tmp_path / "plots"
        analyzer.PLOTS_DIR.mkdir()
        analyzer.load_data()
        analyzer.split_data()
        analyzer.remove_high_na_columns()

        assert "Pool_QC" not in analyzer.full_data.columns


# ---------------------------------------------------------------------------
# Tests: apply_feature_engineering
# ---------------------------------------------------------------------------

class TestApplyFeatureEngineering:
    def test_adds_total_bathrooms(
        self, analyzer_eda: HousingAnalyzer
    ) -> None:
        with patch("housinganalysis.plt"):
            with patch("housinganalysis.sns"):
                analyzer_eda.apply_feature_engineering()
        assert "TotalBathrooms" in analyzer_eda.training_data.columns

    def test_adds_total_sf(self, analyzer_eda: HousingAnalyzer) -> None:
        with patch("housinganalysis.plt"):
            with patch("housinganalysis.sns"):
                analyzer_eda.apply_feature_engineering()
        assert "Total_SF" in analyzer_eda.training_data.columns

    def test_adds_rich_neighborhood_flag(
        self, analyzer_eda: HousingAnalyzer
    ) -> None:
        with patch("housinganalysis.plt"):
            with patch("housinganalysis.sns"):
                analyzer_eda.apply_feature_engineering()
        assert "in_rich_neighborhood" in analyzer_eda.training_data.columns
        vals = analyzer_eda.training_data["in_rich_neighborhood"].unique()
        assert set(vals).issubset({0, 1})

    def test_rich_neighborhoods_populated(
        self, analyzer_eda: HousingAnalyzer
    ) -> None:
        with patch("housinganalysis.plt"):
            with patch("housinganalysis.sns"):
                analyzer_eda.apply_feature_engineering()
        assert len(analyzer_eda.rich_neighborhoods) == 4

    def test_label_encoders_populated(
        self, analyzer_eda: HousingAnalyzer
    ) -> None:
        with patch("housinganalysis.plt"):
            with patch("housinganalysis.sns"):
                analyzer_eda.apply_feature_engineering()
        assert len(analyzer_eda.label_encoders) > 0

    def test_no_object_dtype_after_encoding(
        self, analyzer_eda: HousingAnalyzer
    ) -> None:
        with patch("housinganalysis.plt"):
            with patch("housinganalysis.sns"):
                analyzer_eda.apply_feature_engineering()
        object_cols = analyzer_eda.training_data.select_dtypes(
            include=["object"]
        ).columns.tolist()
        assert object_cols == [], f"Unexpected object cols: {object_cols}"


# ---------------------------------------------------------------------------
# Tests: encode_categorical (static)
# ---------------------------------------------------------------------------

class TestEncodeCategorical:
    def _sample_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "color": ["red", "blue", "green", "red"],
                "size": ["S", "M", "L", "M"],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        )

    def test_returns_tuple(self) -> None:
        df = self._sample_df()
        result = HousingAnalyzer.encode_categorical(df)
        assert isinstance(result, tuple) and len(result) == 2

    def test_object_cols_become_integer(self) -> None:
        df = self._sample_df()
        encoded, _ = HousingAnalyzer.encode_categorical(df)
        assert encoded["color"].dtype in (np.int64, np.int32, int)
        assert encoded["size"].dtype in (np.int64, np.int32, int)

    def test_numeric_cols_unchanged(self) -> None:
        df = self._sample_df()
        encoded, _ = HousingAnalyzer.encode_categorical(df)
        pd.testing.assert_series_equal(encoded["value"], df["value"])

    def test_encoder_dict_keys_are_categorical_cols(self) -> None:
        df = self._sample_df()
        _, encoders = HousingAnalyzer.encode_categorical(df)
        assert set(encoders.keys()) == {"color", "size"}

    def test_test_mode_uses_fitted_encoders(self) -> None:
        train = pd.DataFrame({"cat": ["a", "b", "c"]})
        _, fitted = HousingAnalyzer.encode_categorical(train)

        test = pd.DataFrame({"cat": ["a", "c", "b"]})
        encoded, _ = HousingAnalyzer.encode_categorical(test, fitted_encoders=fitted)
        # Values must be within [0, 2]
        assert encoded["cat"].between(0, 2).all()

    def test_unseen_category_mapped_to_first_class(self) -> None:
        train = pd.DataFrame({"cat": ["a", "b"]})
        _, fitted = HousingAnalyzer.encode_categorical(train)

        test = pd.DataFrame({"cat": ["z"]})  # unseen value
        encoded, _ = HousingAnalyzer.encode_categorical(test, fitted_encoders=fitted)
        # Should not raise and should map to the first known class (index 0)
        assert encoded["cat"].iloc[0] == 0


# ---------------------------------------------------------------------------
# Tests: process_data_for_modeling
# ---------------------------------------------------------------------------

class TestProcessDataForModeling:
    def test_returns_four_element_tuple(
        self, analyzer_engineered: HousingAnalyzer
    ) -> None:
        feats = ["Overall_Qual", "Gr_Liv_Area", "SalePrice"]
        result = analyzer_engineered.process_data_for_modeling(
            analyzer_engineered.training_data.copy(), feats
        )
        assert len(result) == 4

    def test_saleprice_not_in_X(
        self, analyzer_engineered: HousingAnalyzer
    ) -> None:
        feats = ["Overall_Qual", "Gr_Liv_Area", "SalePrice"]
        X, y, _, _ = analyzer_engineered.process_data_for_modeling(
            analyzer_engineered.training_data.copy(), feats
        )
        assert "SalePrice" not in X.columns

    def test_y_is_log_transformed_when_flag_set(
        self, analyzer_engineered: HousingAnalyzer
    ) -> None:
        analyzer_engineered.LOG_TARGET = True
        feats = ["Overall_Qual", "Gr_Liv_Area", "SalePrice"]
        _, y, _, _ = analyzer_engineered.process_data_for_modeling(
            analyzer_engineered.training_data.copy(), feats
        )
        raw_prices = analyzer_engineered.training_data["SalePrice"]
        # log1p of prices in the 1e5 range is around 11–13
        assert y.max() < 20  # not original dollar scale

    def test_y_not_log_transformed_when_flag_unset(
        self, analyzer_engineered: HousingAnalyzer
    ) -> None:
        analyzer_engineered.LOG_TARGET = False
        feats = ["Overall_Qual", "Gr_Liv_Area", "SalePrice"]
        _, y, _, _ = analyzer_engineered.process_data_for_modeling(
            analyzer_engineered.training_data.copy(), feats
        )
        # Original prices are in the 100k–400k range
        assert y.max() > 1000

    def test_scaler_returned_in_training_mode(
        self, analyzer_engineered: HousingAnalyzer
    ) -> None:
        from sklearn.preprocessing import StandardScaler

        feats = ["Overall_Qual", "Gr_Liv_Area", "SalePrice"]
        _, _, scaler, _ = analyzer_engineered.process_data_for_modeling(
            analyzer_engineered.training_data.copy(), feats
        )
        assert isinstance(scaler, StandardScaler)

    def test_scaler_none_in_test_mode(
        self, analyzer_engineered: HousingAnalyzer
    ) -> None:
        feats = ["Overall_Qual", "Gr_Liv_Area", "SalePrice"]
        _, _, scaler_tr, enc_tr = analyzer_engineered.process_data_for_modeling(
            analyzer_engineered.training_data.copy(), feats
        )
        _, _, scaler_te, enc_te = analyzer_engineered.process_data_for_modeling(
            analyzer_engineered.training_data.copy(),
            feats,
            fitted_scaler=scaler_tr,
            fitted_encoders=enc_tr,
        )
        assert scaler_te is None
        assert enc_te is None

    def test_missing_feature_names_silently_ignored(
        self, analyzer_engineered: HousingAnalyzer
    ) -> None:
        feats = ["Overall_Qual", "NonExistentColumn", "SalePrice"]
        X, _, _, _ = analyzer_engineered.process_data_for_modeling(
            analyzer_engineered.training_data.copy(), feats
        )
        assert "NonExistentColumn" not in X.columns

    def test_outliers_removed_from_input(
        self, analyzer_engineered: HousingAnalyzer
    ) -> None:
        data = analyzer_engineered.training_data.copy()
        # Inject a fake outlier row
        outlier = data.iloc[0].copy()
        outlier["Gr_Liv_Area"] = 9999.0
        data = pd.concat([data, outlier.to_frame().T], ignore_index=True)

        feats = ["Overall_Qual", "Gr_Liv_Area", "SalePrice"]
        X, _, _, _ = analyzer_engineered.process_data_for_modeling(data, feats)
        # The outlier row (Gr_Liv_Area=9999) must have been dropped
        assert X.shape[0] < len(data)


# ---------------------------------------------------------------------------
# Tests: _save_plot
# ---------------------------------------------------------------------------

class TestSavePlot:
    def test_increments_counter(self, tmp_path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        analyzer = HousingAnalyzer()
        analyzer.PLOTS_DIR = tmp_path / "plots"
        analyzer.PLOTS_DIR.mkdir()

        assert analyzer.plot_counter == 0
        plt.figure()
        analyzer._save_plot("first")
        assert analyzer.plot_counter == 1

        plt.figure()
        analyzer._save_plot("second")
        assert analyzer.plot_counter == 2

    def test_file_created_with_correct_name(self, tmp_path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        analyzer = HousingAnalyzer()
        analyzer.PLOTS_DIR = tmp_path / "plots"
        analyzer.PLOTS_DIR.mkdir()

        plt.figure()
        path = analyzer._save_plot("my_chart")

        assert path.exists()
        assert path.name == "01_my_chart.png"

    def test_returns_path_object(self, tmp_path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        analyzer = HousingAnalyzer()
        analyzer.PLOTS_DIR = tmp_path / "plots"
        analyzer.PLOTS_DIR.mkdir()

        plt.figure()
        result = analyzer._save_plot("test")
        assert isinstance(result, Path)

    def test_sequential_numbering(self, tmp_path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        analyzer = HousingAnalyzer()
        analyzer.PLOTS_DIR = tmp_path / "plots"
        analyzer.PLOTS_DIR.mkdir()

        names = ["alpha", "beta", "gamma"]
        paths = []
        for name in names:
            plt.figure()
            paths.append(analyzer._save_plot(name))

        for i, (path, name) in enumerate(zip(paths, names), start=1):
            assert path.name == f"{i:02d}_{name}.png"

    def test_existing_figure_is_closed_after_save(self, tmp_path: Path) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        analyzer = HousingAnalyzer()
        analyzer.PLOTS_DIR = tmp_path / "plots"
        analyzer.PLOTS_DIR.mkdir()

        plt.figure()
        open_before = len(plt.get_fignums())
        analyzer._save_plot("close_test")
        open_after = len(plt.get_fignums())
        assert open_after < open_before


# ---------------------------------------------------------------------------
# Tests: static helper methods
# ---------------------------------------------------------------------------

class TestStaticHelpers:
    """Quick sanity tests for the pure static / standalone helpers."""

    def test_add_total_bathrooms_basic(self) -> None:
        row = {
            "Bsmt_Full_Bath": 1.0,
            "Full_Bath": 1.0,
            "Bsmt_Half_Bath": 0.0,
            "Half_Bath": 1.0,
        }
        df = pd.DataFrame([row])
        result = HousingAnalyzer.add_total_bathrooms(df)
        assert result["TotalBathrooms"].iloc[0] == pytest.approx(2.5)

    def test_add_total_bathrooms_missing_values(self) -> None:
        row = {
            "Bsmt_Full_Bath": np.nan,
            "Full_Bath": 2.0,
            "Bsmt_Half_Bath": np.nan,
            "Half_Bath": 0.0,
        }
        df = pd.DataFrame([row])
        result = HousingAnalyzer.add_total_bathrooms(df)
        assert result["TotalBathrooms"].iloc[0] == pytest.approx(2.0)

    def test_add_total_sf(self) -> None:
        df = pd.DataFrame({"Total_Bsmt_SF": [800.0], "Gr_Liv_Area": [1200.0]})
        result = HousingAnalyzer.add_total_sf(df)
        assert result["Total_SF"].iloc[0] == pytest.approx(2000.0)

    def test_find_rich_neighborhoods_order(self) -> None:
        df = pd.DataFrame(
            {
                "Neighborhood": ["A", "A", "B", "B", "C"],
                "SalePrice": [300.0, 400.0, 100.0, 150.0, 500.0],
            }
        )
        rich = HousingAnalyzer.find_rich_neighborhoods(df, n=2)
        assert rich[0] == "C"
        assert rich[1] == "A"

    def test_add_rich_neighborhood_flag_values(self) -> None:
        df = pd.DataFrame({"Neighborhood": ["NridgHt", "OldTown", "NridgHt"]})
        result = HousingAnalyzer.add_rich_neighborhood_flag(df, ["NridgHt"])
        assert list(result["in_rich_neighborhood"]) == [1, 0, 1]

    def test_remove_outliers_filters_correctly(self) -> None:
        df = pd.DataFrame({"x": [100.0, 5000.0, 200.0], "y": [1, 2, 3]})
        result = HousingAnalyzer.remove_outliers(df, "x", 4000.0)
        assert 5000.0 not in result["x"].values
        assert len(result) == 2

    def test_remove_outliers_does_not_mutate_input(self) -> None:
        df = pd.DataFrame({"x": [100.0, 5000.0]})
        _ = HousingAnalyzer.remove_outliers(df, "x", 200.0)
        assert len(df) == 2  # original unchanged
