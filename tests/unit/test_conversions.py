"""Unit tests for power conversion utilities."""

import pandas as pd

from src.utils.conversions import convert_power_series_to_watts


class TestPowerConversions:
    """Test power unit conversions."""

    def test_convert_watts_preserves_values(self):
        """Watt inputs should pass through unchanged as floats."""
        result = convert_power_series_to_watts(pd.Series([1000, 12.5]), "W")

        assert result.tolist() == [1000.0, 12.5]

    def test_convert_milliwatts_to_watts(self):
        """Milliwatt inputs should be divided by 1000."""
        result = convert_power_series_to_watts(pd.Series([50000]), "mW")

        assert result.iloc[0] == 50.0

    def test_convert_kilowatts_to_watts(self):
        """Kilowatt inputs should be multiplied by 1000."""
        result = convert_power_series_to_watts(pd.Series([1.5]), "kW")

        assert result.iloc[0] == 1500.0

    def test_unknown_unit_defaults_to_watts(self):
        """Unknown units currently default to W for compatibility."""
        result = convert_power_series_to_watts(pd.Series([1000]), "invalid_unit")

        assert result.iloc[0] == 1000.0

    def test_empty_series(self):
        """Empty series inputs should remain empty."""
        result = convert_power_series_to_watts(pd.Series(dtype=float), "W")

        assert len(result) == 0
