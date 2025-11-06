"""County and state-level aggregation of electricity rates with configurable weighting."""

import logging
from typing import Dict, List, Optional, Set, Union

import pandas as pd
from pydantic import BaseModel

from .crosswalk import UtilityCountyCrosswalk, UtilityCountyMapping
from .pricing import EffectiveRate

logger = logging.getLogger(__name__)


class AggregationWeights(BaseModel):
    """Model for aggregation weight configuration."""
    method: str  # 'simple', 'sales', 'meters', 'custom'
    custom_weights: Optional[Dict[int, float]] = None  # utility_id -> weight


class CountyRate(BaseModel):
    """Model for aggregated county-level rate data."""
    county_fips: str
    county_name: str
    state: str
    customer_class: str
    weighted_rate_cents_per_kwh: float
    utility_count: int
    utilities: List[int]  # List of utility IDs
    total_weight: float
    rate_range_min: float
    rate_range_max: float
    data_quality_score: float
    coverage_flags: List[str]


class StateRate(BaseModel):
    """Model for aggregated state-level rate data."""
    state: str
    customer_class: str
    weighted_rate_cents_per_kwh: float
    county_count: int
    utility_count: int
    total_weight: float
    rate_range_min: float
    rate_range_max: float
    coverage_ratio: float  # Fraction of counties with data


class AggregationError(Exception):
    """Exception for aggregation-related errors."""
    pass


class RateAggregator:
    """
    Aggregates utility-level electricity rates to county and state levels.

    Features:
    - Multiple weighting methods (simple, sales-weighted, meter-weighted)
    - Handles multiple utilities per county
    - Calculates coverage statistics and data quality metrics
    - Provides validation and outlier detection
    """

    WEIGHTING_METHODS = {
        'simple': 'Equal weight for all utilities',
        'sales': 'Weight by electricity sales volume',
        'meters': 'Weight by number of customer meters',
        'custom': 'User-provided weights'
    }

    def __init__(self,
                 crosswalk: UtilityCountyCrosswalk,
                 weighting_method: str = 'simple'):
        """
        Initialize the rate aggregator.

        Args:
            crosswalk: Utility-county mapping crosswalk
            weighting_method: Method for weighting utilities
        """
        self.crosswalk = crosswalk
        self.weighting_method = weighting_method

        if weighting_method not in self.WEIGHTING_METHODS:
            raise AggregationError(f"Unknown weighting method: {weighting_method}")

        logger.info(f"Initialized RateAggregator with {weighting_method} weighting")

    def prepare_rate_data(self, effective_rates: List[EffectiveRate]) -> pd.DataFrame:
        """
        Prepare effective rate data for aggregation.

        Args:
            effective_rates: List of calculated effective rates

        Returns:
            DataFrame with rate data ready for aggregation
        """
        if not effective_rates:
            return pd.DataFrame()

        # Convert to DataFrame
        rate_data = []
        for rate in effective_rates:
            rate_data.append({
                'utility_id': rate.utility_id,
                'utility_name': rate.utility_name,
                'customer_class': rate.customer_class,
                'tariff_label': rate.tariff_label,
                'effective_rate_cents_per_kwh': rate.effective_rate_cents_per_kwh,
                'monthly_bill_dollars': rate.monthly_bill_dollars,
                'data_quality_flags': rate.data_quality_flags
            })

        df = pd.DataFrame(rate_data)

        # Add crosswalk information
        crosswalk_df = self.crosswalk.to_dataframe()
        if not crosswalk_df.empty:
            df = df.merge(
                crosswalk_df[['utility_id', 'county_fips', 'county_name', 'state', 'weight']],
                on='utility_id',
                how='left'
            )
        else:
            logger.warning("Empty crosswalk - county aggregation will not be possible")

        return df

    def calculate_data_quality_score(self, utility_data: pd.DataFrame) -> float:
        """
        Calculate a data quality score for a group of utilities.

        Args:
            utility_data: DataFrame with utility rate data

        Returns:
            Quality score between 0 and 1
        """
        score = 1.0

        # Penalize for missing data
        if utility_data.empty:
            return 0.0

        # Penalize for data quality flags
        total_flags = sum(len(flags) for flags in utility_data['data_quality_flags'] if flags)
        flag_penalty = min(0.5, total_flags * 0.1)
        score -= flag_penalty

        # Penalize for high rate variance (potential outliers)
        rates = utility_data['effective_rate_cents_per_kwh']
        if len(rates) > 1:
            cv = rates.std() / rates.mean() if rates.mean() > 0 else 0
            variance_penalty = min(0.3, cv * 0.5)
            score -= variance_penalty

        # Bonus for multiple utilities (better coverage)
        if len(utility_data) > 1:
            score += min(0.1, len(utility_data) * 0.02)

        return max(0.0, min(1.0, score))

    def aggregate_to_counties(self,
                             effective_rates: List[EffectiveRate],
                             customer_classes: Optional[List[str]] = None) -> List[CountyRate]:
        """
        Aggregate utility rates to county level.

        Args:
            effective_rates: List of calculated effective rates
            customer_classes: List of customer classes to include

        Returns:
            List of county-level aggregated rates
        """
        df = self.prepare_rate_data(effective_rates)

        if df.empty or 'county_fips' not in df.columns:
            logger.warning("No data available for county aggregation")
            return []

        # Filter by customer classes if specified
        if customer_classes:
            df = df[df['customer_class'].isin(customer_classes)]

        county_rates = []

        # Group by county and customer class
        for (county_fips, customer_class), group in df.groupby(['county_fips', 'customer_class']):
            try:
                # Get weights
                if self.weighting_method == 'simple':
                    weights = pd.Series(1.0, index=group.index)
                else:
                    weights = group['weight']

                # Calculate weighted average rate
                rates = group['effective_rate_cents_per_kwh']
                weighted_rate = (rates * weights).sum() / weights.sum()

                # Get county metadata
                county_name = group['county_name'].iloc[0]
                state = group['state'].iloc[0]
                utilities = group['utility_id'].unique().tolist()

                # Calculate statistics
                rate_min = rates.min()
                rate_max = rates.max()
                total_weight = weights.sum()
                quality_score = self.calculate_data_quality_score(group)

                # Coverage flags
                coverage_flags = []
                if len(utilities) == 1:
                    coverage_flags.append('single_utility')
                if quality_score < 0.5:
                    coverage_flags.append('low_quality')
                if rate_max - rate_min > weighted_rate * 0.5:  # High variance
                    coverage_flags.append('high_variance')

                county_rate = CountyRate(
                    county_fips=county_fips,
                    county_name=county_name,
                    state=state,
                    customer_class=customer_class,
                    weighted_rate_cents_per_kwh=weighted_rate,
                    utility_count=len(utilities),
                    utilities=utilities,
                    total_weight=total_weight,
                    rate_range_min=rate_min,
                    rate_range_max=rate_max,
                    data_quality_score=quality_score,
                    coverage_flags=coverage_flags
                )

                county_rates.append(county_rate)

            except Exception as e:
                logger.error(f"Failed to aggregate county {county_fips}: {e}")
                continue

        logger.info(f"Aggregated rates for {len(county_rates)} counties")
        return county_rates

    def aggregate_to_states(self,
                           county_rates: List[CountyRate],
                           customer_classes: Optional[List[str]] = None) -> List[StateRate]:
        """
        Aggregate county rates to state level.

        Args:
            county_rates: List of county-level rates
            customer_classes: List of customer classes to include

        Returns:
            List of state-level aggregated rates
        """
        if not county_rates:
            return []

        # Convert to DataFrame for easier grouping
        county_data = []
        for county_rate in county_rates:
            if customer_classes and county_rate.customer_class not in customer_classes:
                continue

            county_data.append({
                'state': county_rate.state,
                'customer_class': county_rate.customer_class,
                'county_fips': county_rate.county_fips,
                'weighted_rate_cents_per_kwh': county_rate.weighted_rate_cents_per_kwh,
                'total_weight': county_rate.total_weight,
                'utility_count': county_rate.utility_count,
                'data_quality_score': county_rate.data_quality_score,
                'rate_range_min': county_rate.rate_range_min,
                'rate_range_max': county_rate.rate_range_max
            })

        if not county_data:
            return []

        df = pd.DataFrame(county_data)
        state_rates = []

        # Group by state and customer class
        for (state, customer_class), group in df.groupby(['state', 'customer_class']):
            try:
                # Weight by county weights for state average
                weights = group['total_weight']
                rates = group['weighted_rate_cents_per_kwh']

                # Calculate weighted average
                state_rate = (rates * weights).sum() / weights.sum() if weights.sum() > 0 else rates.mean()

                # Calculate statistics
                county_count = len(group)
                total_utilities = group['utility_count'].sum()
                total_weight = weights.sum()
                rate_min = rates.min()
                rate_max = rates.max()

                # Coverage ratio (could be enhanced with total county count per state)
                coverage_ratio = 1.0  # Placeholder - would need external data for total counties

                state_rate_obj = StateRate(
                    state=state,
                    customer_class=customer_class,
                    weighted_rate_cents_per_kwh=state_rate,
                    county_count=county_count,
                    utility_count=total_utilities,
                    total_weight=total_weight,
                    rate_range_min=rate_min,
                    rate_range_max=rate_max,
                    coverage_ratio=coverage_ratio
                )

                state_rates.append(state_rate_obj)

            except Exception as e:
                logger.error(f"Failed to aggregate state {state}: {e}")
                continue

        logger.info(f"Aggregated rates for {len(state_rates)} state-class combinations")
        return state_rates

    def validate_aggregation(self,
                           county_rates: List[CountyRate],
                           tolerance_cents: float = 50.0) -> Dict[str, List[str]]:
        """
        Validate aggregated rates and flag potential issues.

        Args:
            county_rates: List of county rates to validate
            tolerance_cents: Tolerance for outlier detection (¢/kWh)

        Returns:
            Dictionary of validation issues by category
        """
        issues = {
            'outliers': [],
            'single_utility_counties': [],
            'low_quality': [],
            'missing_coverage': []
        }

        if not county_rates:
            return issues

        # Calculate percentiles for outlier detection
        rates = [cr.weighted_rate_cents_per_kwh for cr in county_rates]
        df_rates = pd.Series(rates)
        q25, q75 = df_rates.quantile([0.25, 0.75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr

        for county_rate in county_rates:
            county_id = f"{county_rate.county_name}, {county_rate.state} ({county_rate.county_fips})"

            # Check for outliers
            rate = county_rate.weighted_rate_cents_per_kwh
            if rate < lower_bound or rate > upper_bound:
                issues['outliers'].append(f"{county_id}: {rate:.2f}¢/kWh")

            # Check for single utility coverage
            if county_rate.utility_count == 1:
                issues['single_utility_counties'].append(county_id)

            # Check for low quality
            if county_rate.data_quality_score < 0.5:
                issues['low_quality'].append(f"{county_id}: score={county_rate.data_quality_score:.2f}")

            # Check coverage flags
            if 'low_quality' in county_rate.coverage_flags or 'high_variance' in county_rate.coverage_flags:
                if county_id not in [item.split(':')[0] for item in issues['low_quality']]:
                    issues['missing_coverage'].append(county_id)

        # Filter out empty issue lists
        return {k: v for k, v in issues.items() if v}

    def get_aggregation_summary(self,
                               county_rates: List[CountyRate],
                               state_rates: List[StateRate]) -> Dict[str, any]:
        """
        Generate summary statistics for aggregated data.

        Args:
            county_rates: List of county rates
            state_rates: List of state rates

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'county_level': {
                'total_counties': len(county_rates),
                'states_represented': len(set(cr.state for cr in county_rates)),
                'customer_classes': len(set(cr.customer_class for cr in county_rates)),
                'avg_utilities_per_county': sum(cr.utility_count for cr in county_rates) / len(county_rates) if county_rates else 0,
                'single_utility_counties': sum(1 for cr in county_rates if cr.utility_count == 1),
                'avg_quality_score': sum(cr.data_quality_score for cr in county_rates) / len(county_rates) if county_rates else 0
            },
            'state_level': {
                'total_states': len(set(sr.state for sr in state_rates)),
                'customer_classes': len(set(sr.customer_class for sr in state_rates)),
                'avg_counties_per_state': sum(sr.county_count for sr in state_rates) / len(state_rates) if state_rates else 0,
                'total_utilities': sum(sr.utility_count for sr in state_rates)
            }
        }

        if county_rates:
            rates = [cr.weighted_rate_cents_per_kwh for cr in county_rates]
            rates_series = pd.Series(rates)
            summary['county_level'].update({
                'rate_stats': {
                    'mean': rates_series.mean(),
                    'median': rates_series.median(),
                    'std': rates_series.std(),
                    'min': rates_series.min(),
                    'max': rates_series.max()
                }
            })

        return summary

    def county_rates_to_dataframe(self, county_rates: List[CountyRate]) -> pd.DataFrame:
        """Convert county rates to pandas DataFrame."""
        if not county_rates:
            return pd.DataFrame()

        data = [rate.model_dump() for rate in county_rates]
        return pd.DataFrame(data)

    def state_rates_to_dataframe(self, state_rates: List[StateRate]) -> pd.DataFrame:
        """Convert state rates to pandas DataFrame."""
        if not state_rates:
            return pd.DataFrame()

        data = [rate.model_dump() for rate in state_rates]
        return pd.DataFrame(data)