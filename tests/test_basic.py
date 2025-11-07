"""Basic tests for URDB County Rates utility."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd

from urdb_county_rates.crosswalk import UtilityCountyCrosswalk, UtilityCountyMapping
from urdb_county_rates.tariffs import TariffSelector
from urdb_county_rates.pricing import RatePricer, UsageProfile


class TestUtilityCountyCrosswalk:
    """Test the utility-county crosswalk functionality."""

    def test_create_sample_crosswalk(self):
        """Test creating a sample crosswalk."""
        crosswalk = UtilityCountyCrosswalk.create_sample_crosswalk()

        assert len(crosswalk.mappings) > 0
        assert len(crosswalk.get_all_utilities()) > 0
        assert len(crosswalk.get_all_counties()) > 0
        assert len(crosswalk.get_states()) > 0

    def test_add_mapping(self):
        """Test adding a mapping."""
        crosswalk = UtilityCountyCrosswalk()

        crosswalk.add_mapping(
            utility_id=123,
            utility_name="Test Utility",
            state="CA",
            county_fips="06001",
            county_name="Alameda County",
            weight=1.0
        )

        assert len(crosswalk.mappings) == 1
        assert 123 in crosswalk.get_all_utilities()
        assert "06001" in crosswalk.get_all_counties()

    def test_get_counties_for_utility(self):
        """Test getting counties for a utility."""
        crosswalk = UtilityCountyCrosswalk.create_sample_crosswalk()

        # Get first utility from sample data
        utility_id = list(crosswalk.get_all_utilities())[0]
        counties = crosswalk.get_counties_for_utility(utility_id)

        assert len(counties) > 0
        assert all(c.utility_id == utility_id for c in counties)

    def test_validate_fips_codes(self):
        """Test FIPS code validation."""
        crosswalk = UtilityCountyCrosswalk()

        # Add valid FIPS
        crosswalk.add_mapping(1, "Utility 1", "CA", "06001", "County 1")

        # Add invalid FIPS
        crosswalk.add_mapping(2, "Utility 2", "CA", "999", "Invalid County")

        invalid_fips = crosswalk.validate_fips_codes()
        assert len(invalid_fips) == 1
        assert "999" in invalid_fips[0]


class TestTariffSelector:
    """Test tariff selection and filtering."""

    def test_normalize_customer_class(self):
        """Test customer class normalization."""
        selector = TariffSelector()

        assert selector.normalize_customer_class("residential") == "residential"
        assert selector.normalize_customer_class("Residential") == "residential"
        assert selector.normalize_customer_class("res") == "residential"
        assert selector.normalize_customer_class("commercial") == "commercial"
        assert selector.normalize_customer_class("industrial") == "industrial"

    def test_parse_date(self):
        """Test date parsing."""
        selector = TariffSelector()

        # Test valid date formats
        date1 = selector.parse_date("2023-01-15")
        assert date1 is not None
        assert date1.year == 2023
        assert date1.month == 1
        assert date1.day == 15

        date2 = selector.parse_date("01/15/2023")
        assert date2 is not None

        # Test invalid date
        date3 = selector.parse_date("invalid-date")
        assert date3 is None

    def test_create_criteria(self):
        """Test creating selection criteria."""
        criteria = TariffSelector.create_criteria(
            customer_class="residential",
            approved_only=True,
            utility_ids=[1, 2, 3]
        )

        assert criteria.customer_class == "residential"
        assert criteria.approved_only is True
        assert criteria.utility_ids == {1, 2, 3}


class TestRatePricer:
    """Test rate pricing calculations."""

    def test_create_usage_profile(self):
        """Test creating usage profiles."""
        profile = RatePricer.create_usage_profile(monthly_kwh=1000)

        assert profile.monthly_kwh == 1000
        assert profile.peak_kw > 0
        assert profile.load_factor > 0

    def test_clean_rate_value(self):
        """Test rate value cleaning."""
        pricer = RatePricer()

        # Test numeric values
        assert pricer.clean_rate_value(0.15) == 0.15
        assert pricer.clean_rate_value("0.15") == 0.15
        assert pricer.clean_rate_value("$0.15") == 0.15

        # Test invalid values
        assert pricer.clean_rate_value(None) == 0.0
        assert pricer.clean_rate_value("invalid") == 0.0
        assert pricer.clean_rate_value(-0.5) == 0.0

    def test_default_profiles(self):
        """Test default usage profiles exist."""
        pricer = RatePricer()

        assert "residential" in pricer.DEFAULT_PROFILES
        assert "commercial" in pricer.DEFAULT_PROFILES
        assert "industrial" in pricer.DEFAULT_PROFILES

        # Check profiles have reasonable values
        res_profile = pricer.DEFAULT_PROFILES["residential"]
        assert res_profile.monthly_kwh > 0
        assert res_profile.peak_kw > 0


# Integration test
def test_full_workflow_mock():
    """Test a simplified end-to-end workflow with mocked data."""

    # Create sample crosswalk
    crosswalk = UtilityCountyCrosswalk.create_sample_crosswalk()

    # Mock tariff data would go here in a real test
    # For now, just verify components can be initialized
    selector = TariffSelector()
    pricer = RatePricer()

    # Test that components work together
    criteria = selector.create_criteria(customer_class="residential")
    assert criteria.customer_class == "residential"

    profile = pricer.create_usage_profile(1000)
    assert profile.monthly_kwh == 1000

    # Verify crosswalk has data
    assert len(crosswalk.mappings) > 0

    print("✅ All components initialized successfully")


if __name__ == "__main__":
    # Run basic tests
    test_full_workflow_mock()

    # Run individual test classes
    test_crosswalk = TestUtilityCountyCrosswalk()
    test_crosswalk.test_create_sample_crosswalk()
    test_crosswalk.test_add_mapping()

    test_selector = TestTariffSelector()
    test_selector.test_normalize_customer_class()
    test_selector.test_parse_date()

    test_pricer = TestRatePricer()
    test_pricer.test_create_usage_profile()
    test_pricer.test_clean_rate_value()

    print("✅ All basic tests passed!")