"""
Enhanced pricing calculations with schema-agnostic parsing and plausibility filtering.
Integrates advanced functionality from existing workflow.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .pricing import RatePricer, EffectiveRate, UsageProfile
from .urdb_client import URDBRateStructure

logger = logging.getLogger(__name__)


class EnhancedRatePricer(RatePricer):
    """
    Enhanced rate pricer with schema-agnostic parsing and advanced filtering.
    """

    def __init__(self,
                 validate_units: bool = True,
                 tou_flattening_method: str = 'weighted_average',
                 energy_min: float = 0.0,
                 energy_max: float = 2.0):
        """
        Initialize enhanced pricer.

        Args:
            validate_units: Whether to validate units
            tou_flattening_method: TOU flattening method
            energy_min: Minimum plausible energy rate ($/kWh)
            energy_max: Maximum plausible energy rate ($/kWh)
        """
        super().__init__(validate_units, tou_flattening_method)
        self.energy_min = energy_min
        self.energy_max = energy_max

    def parse_schedule(self, schedule_str: Any) -> Optional[List[int]]:
        """
        Parse URDB schedule string into list of period IDs.

        Args:
            schedule_str: Schedule string from URDB

        Returns:
            List of period IDs or None if invalid
        """
        if pd.isna(schedule_str) or not isinstance(schedule_str, str):
            return None

        if not schedule_str.strip().startswith('['):
            return None

        try:
            arr = json.loads(schedule_str)
            vals = []
            for month in arr:
                if isinstance(month, list):
                    vals.extend(month)
                else:
                    vals.append(month)
            return vals
        except Exception as e:
            logger.debug(f"Failed to parse schedule: {e}")
            return None

    def is_time_of_use(self, weekday_schedule: Any, weekend_schedule: Any) -> bool:
        """
        Determine if tariff uses time-of-use pricing.

        Args:
            weekday_schedule: Weekday schedule string
            weekend_schedule: Weekend schedule string

        Returns:
            True if TOU is used
        """
        weekday_vals = self.parse_schedule(weekday_schedule)
        weekend_vals = self.parse_schedule(weekend_schedule)

        for vals in [weekday_vals, weekend_vals]:
            if vals and len(set(vals)) > 1:
                return True

        return False

    def get_schedule_shares(self, schedule_str: Any) -> Optional[Dict[int, float]]:
        """
        Calculate time shares for each period in a schedule.

        Args:
            schedule_str: Schedule string from URDB

        Returns:
            Dictionary mapping period ID to time share
        """
        vals = self.parse_schedule(schedule_str)
        if not vals:
            return None

        series = pd.Series(vals)
        value_counts = series.value_counts(normalize=True)
        return {int(k): float(v) for k, v in value_counts.items()}

    def build_period_rate_columns(self, tariff_data: Dict[str, Any]) -> Dict[int, str]:
        """
        Build mapping of period ID to rate column for tier 0.

        Args:
            tariff_data: Tariff data dictionary

        Returns:
            Dictionary mapping period ID to column name
        """
        period_cols = {}

        for col_name in tariff_data.keys():
            match = re.match(r"energyratestructure/period(\d+)/tier0rate", col_name)
            if match:
                period_id = int(match.group(1))
                period_cols[period_id] = col_name

        return period_cols

    def build_any_tier_rate_columns(self, tariff_data: Dict[str, Any]) -> Dict[int, str]:
        """
        Build mapping of period ID to any available tier rate column.

        Args:
            tariff_data: Tariff data dictionary

        Returns:
            Dictionary mapping period ID to first available rate column
        """
        period_cols = {}

        for col_name in tariff_data.keys():
            match = re.match(r"energyratestructure/period(\d+)/tier(\d+)rate", col_name)
            if match:
                period_id = int(match.group(1))
                if period_id not in period_cols:  # Take first tier found
                    period_cols[period_id] = col_name

        return period_cols

    def calculate_time_weighted_energy_rate(self,
                                          tariff_data: Dict[str, Any]) -> Tuple[float, bool, bool, bool]:
        """
        Calculate time-weighted energy rate with fallback methods.

        Args:
            tariff_data: Tariff data dictionary

        Returns:
            Tuple of (rate, used_filtered_avg, all_filtered, had_suspicious)
        """
        # Build rate column mappings
        tier0_cols = self.build_period_rate_columns(tariff_data)
        any_tier_cols = self.build_any_tier_rate_columns(tariff_data)

        # Try time-weighted calculation first
        rate = self._try_time_weighted_rate(tariff_data, tier0_cols, any_tier_cols)
        if self._is_plausible_rate(rate):
            return rate, False, False, False

        # Try simple average of available rates
        rate = self._try_simple_average_rate(tariff_data, tier0_cols, any_tier_cols)
        if self._is_plausible_rate(rate):
            return rate, False, False, False

        # Fall back to schema-agnostic filtering
        return self._try_filtered_schema_agnostic_rate(tariff_data)

    def _try_time_weighted_rate(self,
                               tariff_data: Dict[str, Any],
                               tier0_cols: Dict[int, str],
                               any_tier_cols: Dict[int, str]) -> float:
        """Try time-weighted rate calculation."""
        cols = tier0_cols if tier0_cols else any_tier_cols
        if not cols:
            return np.nan

        # Get schedule shares
        weekday_shares = self.get_schedule_shares(tariff_data.get('energyweekdayschedule'))
        weekend_shares = self.get_schedule_shares(tariff_data.get('energyweekendschedule'))

        if not weekday_shares and not weekend_shares:
            # No TOU - just use first available rate
            first_col = cols.get(0) or list(cols.values())[0]
            return self._safe_float(tariff_data.get(first_col))

        # Combine weekday and weekend shares
        combined_shares = {}
        for shares in [weekday_shares or {}, weekend_shares or {}]:
            for period_id, share in shares.items():
                combined_shares[period_id] = combined_shares.get(period_id, 0.0) + share

        # Normalize shares
        total_share = sum(combined_shares.values())
        if total_share <= 0:
            first_col = cols.get(0) or list(cols.values())[0]
            return self._safe_float(tariff_data.get(first_col))

        combined_shares = {k: v / total_share for k, v in combined_shares.items()}

        # Calculate weighted average
        weighted_sum = 0.0
        total_weight = 0.0

        for period_id, weight in combined_shares.items():
            col = cols.get(period_id)
            if col:
                rate = self._safe_float(tariff_data.get(col))
                if not np.isnan(rate):
                    weighted_sum += rate * weight
                    total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else np.nan

    def _try_simple_average_rate(self,
                                tariff_data: Dict[str, Any],
                                tier0_cols: Dict[int, str],
                                any_tier_cols: Dict[int, str]) -> float:
        """Try simple average of available rates."""
        cols = tier0_cols if tier0_cols else any_tier_cols
        if not cols:
            return np.nan

        rates = []
        for col in cols.values():
            rate = self._safe_float(tariff_data.get(col))
            if not np.isnan(rate):
                rates.append(rate)

        return float(np.mean(rates)) if rates else np.nan

    def _try_filtered_schema_agnostic_rate(self,
                                         tariff_data: Dict[str, Any]) -> Tuple[float, bool, bool, bool]:
        """
        Schema-agnostic rate calculation with plausibility filtering.

        Returns:
            Tuple of (rate, used_filtered_avg, all_filtered, had_suspicious)
        """
        # Find all energy rate columns
        energy_cols = []
        for col_name in tariff_data.keys():
            if re.match(r"^energyratestructure.*rate$", col_name):
                energy_cols.append(col_name)

        if not energy_cols:
            return np.nan, True, True, False

        # Extract and validate rates
        rates = []
        suspicious_count = 0

        for col in energy_cols:
            rate = self._safe_float(tariff_data.get(col))
            if not np.isnan(rate):
                if self._is_plausible_rate(rate):
                    rates.append(rate)
                else:
                    suspicious_count += 1

        had_suspicious = suspicious_count > 0
        all_filtered = len(rates) == 0 and len(energy_cols) > 0

        if not rates:
            return np.nan, True, all_filtered, had_suspicious

        return float(np.mean(rates)), True, all_filtered, had_suspicious

    def _safe_float(self, value: Any) -> float:
        """Safely convert value to float."""
        try:
            if pd.isna(value):
                return np.nan
            return float(value)
        except (ValueError, TypeError):
            return np.nan

    def _is_plausible_rate(self, rate: float) -> bool:
        """Check if rate is within plausible bounds."""
        return not np.isnan(rate) and self.energy_min < rate < self.energy_max

    def calculate_enhanced_effective_rate(self,
                                        tariff: URDBRateStructure,
                                        usage_profile: Optional[UsageProfile] = None,
                                        customer_class: Optional[str] = None) -> EffectiveRate:
        """
        Calculate effective rate with enhanced parsing and validation.

        Args:
            tariff: URDB rate structure
            usage_profile: Customer usage profile
            customer_class: Customer class

        Returns:
            EffectiveRate with additional quality flags
        """
        # Convert tariff to dictionary for processing
        tariff_dict = tariff.model_dump()

        # Calculate energy rate with enhanced method
        energy_rate, used_filtered, all_filtered, had_suspicious = self.calculate_time_weighted_energy_rate(tariff_dict)

        # Use default profile if none provided
        if usage_profile is None:
            if customer_class and customer_class.lower() in self.DEFAULT_PROFILES:
                usage_profile = self.DEFAULT_PROFILES[customer_class.lower()]
            else:
                usage_profile = self.DEFAULT_PROFILES['residential']

        # Calculate charges (simplified for now - could be enhanced further)
        energy_charge = energy_rate * usage_profile.monthly_kwh if not np.isnan(energy_rate) else 0.0

        # Demand charges (simplified)
        demand_charge = 0.0
        if usage_profile.peak_kw:
            # Extract demand rates (simplified)
            demand_rates = []
            for col, value in tariff_dict.items():
                if 'demandstructure' in col and 'rate' in col:
                    rate = self._safe_float(value)
                    if not np.isnan(rate):
                        demand_rates.append(rate)
            if demand_rates:
                demand_charge = np.mean(demand_rates) * usage_profile.peak_kw

        # Fixed charges (simplified)
        fixed_charge = 0.0
        for col, value in tariff_dict.items():
            if 'fixed' in col.lower() and 'charge' in col.lower():
                charge = self._safe_float(value)
                if not np.isnan(charge):
                    fixed_charge += charge

        total_bill = energy_charge + demand_charge + fixed_charge
        effective_rate = (total_bill / usage_profile.monthly_kwh * 100) if usage_profile.monthly_kwh > 0 else 0

        # Enhanced quality flags
        quality_flags = []
        if tariff.approved is False:
            quality_flags.append('not_approved')
        if not tariff.effective:
            quality_flags.append('no_effective_date')
        if used_filtered:
            quality_flags.append('used_filtered_average')
        if all_filtered:
            quality_flags.append('all_energy_cells_filtered')
        if had_suspicious:
            quality_flags.append('had_suspicious_cells')
        if np.isnan(energy_rate):
            quality_flags.append('no_valid_energy_rate')

        return EffectiveRate(
            utility_id=tariff.eiaid or 0,
            utility_name=tariff.utility,
            customer_class=customer_class or tariff.sector,
            tariff_label=tariff.label,
            effective_rate_cents_per_kwh=effective_rate,
            monthly_usage_kwh=usage_profile.monthly_kwh,
            monthly_bill_dollars=total_bill,
            energy_charge_dollars=energy_charge,
            demand_charge_dollars=demand_charge,
            fixed_charge_dollars=fixed_charge,
            calculation_method='enhanced',
            assumptions={
                'monthly_kwh': usage_profile.monthly_kwh,
                'peak_kw': usage_profile.peak_kw,
                'load_factor': usage_profile.load_factor,
                'tou_method': self.tou_flattening_method,
                'energy_min': self.energy_min,
                'energy_max': self.energy_max,
                'used_filtered_avg': used_filtered,
                'all_filtered': all_filtered,
                'had_suspicious': had_suspicious
            },
            data_quality_flags=quality_flags
        )