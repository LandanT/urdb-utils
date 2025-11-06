"""Electricity rate pricing calculations and effective ¢/kWh computations."""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from pydantic import BaseModel

from .urdb_client import URDBRateStructure

logger = logging.getLogger(__name__)


class PricingError(Exception):
    """Exception for pricing calculation errors."""
    pass


class EffectiveRate(BaseModel):
    """Model for computed effective electricity rates."""
    utility_id: int
    utility_name: str
    customer_class: str
    tariff_label: str
    effective_rate_cents_per_kwh: float
    monthly_usage_kwh: float
    monthly_bill_dollars: float
    energy_charge_dollars: float
    demand_charge_dollars: float
    fixed_charge_dollars: float
    calculation_method: str
    assumptions: Dict[str, Any]
    data_quality_flags: List[str]


class UsageProfile(BaseModel):
    """Model for customer usage profiles."""
    monthly_kwh: float = 1000.0  # Default residential usage
    peak_kw: Optional[float] = None  # For demand charges
    load_factor: float = 0.5  # Ratio of average to peak demand
    seasonal_pattern: Optional[Dict[str, float]] = None  # Month -> multiplier
    time_of_use_pattern: Optional[Dict[str, float]] = None  # Hour -> multiplier


class RatePricer:
    """
    Calculates effective electricity rates from URDB tariff structures.

    Handles:
    - Energy charges (flat, tiered, time-of-use)
    - Demand charges (flat, tiered, time-of-use)
    - Fixed charges
    - Unit conversions and validation
    - TOU flattening with configurable assumptions
    """

    # Standard usage profiles by customer class
    DEFAULT_PROFILES = {
        'residential': UsageProfile(monthly_kwh=1000.0, peak_kw=5.0, load_factor=0.4),
        'commercial': UsageProfile(monthly_kwh=5000.0, peak_kw=30.0, load_factor=0.6),
        'industrial': UsageProfile(monthly_kwh=50000.0, peak_kw=300.0, load_factor=0.8)
    }

    def __init__(self,
                 validate_units: bool = True,
                 tou_flattening_method: str = 'weighted_average'):
        """
        Initialize the rate pricer.

        Args:
            validate_units: Whether to validate and convert units
            tou_flattening_method: Method for flattening TOU rates
        """
        self.validate_units = validate_units
        self.tou_flattening_method = tou_flattening_method
        logger.info(f"Initialized RatePricer with TOU method: {tou_flattening_method}")

    def clean_rate_value(self, value: Any, expected_unit: str = "$/kWh") -> float:
        """
        Clean and validate a rate value.

        Args:
            value: Raw rate value from URDB
            expected_unit: Expected unit for validation

        Returns:
            Cleaned numeric rate value
        """
        if value is None:
            return 0.0

        # Handle different value types
        if isinstance(value, (int, float)):
            rate = float(value)
        elif isinstance(value, str):
            # Remove currency symbols and whitespace
            cleaned = value.strip().replace('$', '').replace(',', '')
            try:
                rate = float(cleaned)
            except ValueError:
                logger.warning(f"Could not parse rate value: {value}")
                return 0.0
        else:
            logger.warning(f"Unexpected rate value type: {type(value)}")
            return 0.0

        # Basic validation
        if rate < 0:
            logger.warning(f"Negative rate value: {rate}")
            return 0.0

        if rate > 1000:  # Sanity check for $/kWh
            logger.warning(f"Suspiciously high rate: {rate} {expected_unit}")

        return rate

    def parse_rate_structure(self, rate_structure: List[Dict[str, Any]]) -> Dict[str, List[Dict]]:
        """
        Parse and categorize rate structure components.

        Args:
            rate_structure: URDB rate structure list

        Returns:
            Dictionary categorizing rate components
        """
        components = {
            'energy': [],
            'demand': [],
            'fixed': []
        }

        for item in rate_structure:
            if not isinstance(item, dict):
                continue

            # Determine component type based on unit
            unit = item.get('unit', '').lower()

            if 'kwh' in unit:
                components['energy'].append(item)
            elif 'kw' in unit and 'kwh' not in unit:
                components['demand'].append(item)
            elif any(term in unit for term in ['month', 'customer', 'service', 'fixed']):
                components['fixed'].append(item)
            else:
                # Try to infer from other fields
                if 'energy' in str(item).lower():
                    components['energy'].append(item)
                elif 'demand' in str(item).lower():
                    components['demand'].append(item)
                else:
                    logger.debug(f"Unclassified rate component: {item}")

        return components

    def calculate_energy_charge(self,
                               energy_components: List[Dict],
                               usage_kwh: float,
                               usage_profile: UsageProfile) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate energy charges from rate components.

        Args:
            energy_components: Energy charge components from rate structure
            usage_kwh: Monthly usage in kWh
            usage_profile: Customer usage profile

        Returns:
            Tuple of (total_charge, calculation_details)
        """
        total_charge = 0.0
        details = {'components': [], 'method': 'tiered'}

        if not energy_components:
            return total_charge, details

        # Sort components by tier if applicable
        sorted_components = sorted(
            energy_components,
            key=lambda x: self.clean_rate_value(x.get('tier', 0))
        )

        remaining_usage = usage_kwh

        for component in sorted_components:
            if remaining_usage <= 0:
                break

            rate = self.clean_rate_value(component.get('rate', 0))
            tier_max = component.get('max', float('inf'))
            tier_min = component.get('min', 0)

            # Calculate usage in this tier
            tier_usage = min(remaining_usage, tier_max - tier_min) if tier_max != float('inf') else remaining_usage
            tier_usage = max(0, tier_usage)

            charge = tier_usage * rate
            total_charge += charge

            details['components'].append({
                'tier': component.get('tier', 0),
                'rate': rate,
                'usage_kwh': tier_usage,
                'charge': charge,
                'unit': component.get('unit')
            })

            remaining_usage -= tier_usage

        return total_charge, details

    def calculate_demand_charge(self,
                               demand_components: List[Dict],
                               peak_kw: float,
                               usage_profile: UsageProfile) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate demand charges from rate components.

        Args:
            demand_components: Demand charge components from rate structure
            peak_kw: Peak demand in kW
            usage_profile: Customer usage profile

        Returns:
            Tuple of (total_charge, calculation_details)
        """
        total_charge = 0.0
        details = {'components': [], 'method': 'simple'}

        if not demand_components or peak_kw <= 0:
            return total_charge, details

        for component in demand_components:
            rate = self.clean_rate_value(component.get('rate', 0))

            # Simple demand charge calculation
            charge = peak_kw * rate
            total_charge += charge

            details['components'].append({
                'rate': rate,
                'demand_kw': peak_kw,
                'charge': charge,
                'unit': component.get('unit')
            })

        return total_charge, details

    def calculate_fixed_charge(self,
                              fixed_components: List[Dict]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate fixed charges from rate components.

        Args:
            fixed_components: Fixed charge components from rate structure

        Returns:
            Tuple of (total_charge, calculation_details)
        """
        total_charge = 0.0
        details = {'components': []}

        for component in fixed_components:
            rate = self.clean_rate_value(component.get('rate', 0))
            total_charge += rate

            details['components'].append({
                'rate': rate,
                'charge': rate,
                'unit': component.get('unit')
            })

        return total_charge, details

    def calculate_effective_rate(self,
                                tariff: URDBRateStructure,
                                usage_profile: Optional[UsageProfile] = None,
                                customer_class: Optional[str] = None) -> EffectiveRate:
        """
        Calculate effective rate for a tariff and usage profile.

        Args:
            tariff: URDB rate structure
            usage_profile: Customer usage profile
            customer_class: Customer class for default profile

        Returns:
            EffectiveRate object with calculated values
        """
        # Use provided profile or default for customer class
        if usage_profile is None:
            if customer_class and customer_class.lower() in self.DEFAULT_PROFILES:
                usage_profile = self.DEFAULT_PROFILES[customer_class.lower()]
            else:
                usage_profile = self.DEFAULT_PROFILES['residential']

        # Parse rate structure
        components = self.parse_rate_structure(tariff.rate_structure)

        # Calculate charges
        energy_charge, energy_details = self.calculate_energy_charge(
            components['energy'], usage_profile.monthly_kwh, usage_profile
        )

        demand_charge, demand_details = self.calculate_demand_charge(
            components['demand'], usage_profile.peak_kw or 0, usage_profile
        )

        fixed_charge, fixed_details = self.calculate_fixed_charge(
            components['fixed']
        )

        # Total bill and effective rate
        total_bill = energy_charge + demand_charge + fixed_charge
        effective_rate = (total_bill / usage_profile.monthly_kwh * 100) if usage_profile.monthly_kwh > 0 else 0

        # Data quality flags
        quality_flags = []
        if not components['energy']:
            quality_flags.append('no_energy_charges')
        if tariff.approved is False:
            quality_flags.append('not_approved')
        if not tariff.effective:
            quality_flags.append('no_effective_date')

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
            calculation_method='standard',
            assumptions={
                'monthly_kwh': usage_profile.monthly_kwh,
                'peak_kw': usage_profile.peak_kw,
                'load_factor': usage_profile.load_factor,
                'tou_method': self.tou_flattening_method
            },
            data_quality_flags=quality_flags
        )

    def calculate_bulk_rates(self,
                            tariffs: List[URDBRateStructure],
                            usage_profiles: Optional[Dict[str, UsageProfile]] = None) -> List[EffectiveRate]:
        """
        Calculate effective rates for multiple tariffs.

        Args:
            tariffs: List of URDB rate structures
            usage_profiles: Dictionary of usage profiles by customer class

        Returns:
            List of EffectiveRate objects
        """
        if usage_profiles is None:
            usage_profiles = self.DEFAULT_PROFILES

        effective_rates = []

        for tariff in tariffs:
            try:
                # Determine customer class
                customer_class = tariff.sector.lower() if tariff.sector else 'residential'
                if customer_class in usage_profiles:
                    usage_profile = usage_profiles[customer_class]
                else:
                    usage_profile = usage_profiles.get('residential', self.DEFAULT_PROFILES['residential'])

                effective_rate = self.calculate_effective_rate(tariff, usage_profile, customer_class)
                effective_rates.append(effective_rate)

            except Exception as e:
                logger.error(f"Failed to calculate rate for {tariff.label}: {e}")
                continue

        logger.info(f"Calculated effective rates for {len(effective_rates)}/{len(tariffs)} tariffs")
        return effective_rates

    def validate_calculations(self, effective_rates: List[EffectiveRate]) -> Dict[str, List[str]]:
        """
        Validate calculated effective rates and flag potential issues.

        Args:
            effective_rates: List of calculated effective rates

        Returns:
            Dictionary of validation issues by category
        """
        issues = {
            'negative_rates': [],
            'zero_rates': [],
            'extremely_high_rates': [],
            'missing_charges': [],
            'quality_flags': []
        }

        for rate in effective_rates:
            rate_id = f"{rate.tariff_label} ({rate.utility_name})"

            # Check for negative rates
            if rate.effective_rate_cents_per_kwh < 0:
                issues['negative_rates'].append(rate_id)

            # Check for zero rates
            if rate.effective_rate_cents_per_kwh == 0:
                issues['zero_rates'].append(rate_id)

            # Check for extremely high rates (>100 ¢/kWh)
            if rate.effective_rate_cents_per_kwh > 100:
                issues['extremely_high_rates'].append(f"{rate_id}: {rate.effective_rate_cents_per_kwh:.2f}¢/kWh")

            # Check for missing charge components
            if (rate.energy_charge_dollars == 0 and
                rate.demand_charge_dollars == 0 and
                rate.fixed_charge_dollars == 0):
                issues['missing_charges'].append(rate_id)

            # Check for data quality flags
            if rate.data_quality_flags:
                issues['quality_flags'].append(f"{rate_id}: {', '.join(rate.data_quality_flags)}")

        # Filter out empty issue lists
        return {k: v for k, v in issues.items() if v}

    def rates_to_dataframe(self, effective_rates: List[EffectiveRate]) -> pd.DataFrame:
        """
        Convert effective rates to pandas DataFrame.

        Args:
            effective_rates: List of calculated effective rates

        Returns:
            DataFrame with rate data
        """
        if not effective_rates:
            return pd.DataFrame()

        data = [rate.model_dump() for rate in effective_rates]
        df = pd.DataFrame(data)

        # Expand assumptions and flags into separate columns if needed
        if 'assumptions' in df.columns:
            assumptions_df = pd.json_normalize(df['assumptions'])
            assumptions_df.columns = [f'assumption_{col}' for col in assumptions_df.columns]
            df = pd.concat([df.drop('assumptions', axis=1), assumptions_df], axis=1)

        return df

    @classmethod
    def create_usage_profile(cls,
                           monthly_kwh: float,
                           peak_kw: Optional[float] = None,
                           load_factor: float = 0.5) -> UsageProfile:
        """
        Create a usage profile with reasonable defaults.

        Args:
            monthly_kwh: Monthly usage in kWh
            peak_kw: Peak demand in kW (estimated from usage if not provided)
            load_factor: Ratio of average to peak demand

        Returns:
            UsageProfile object
        """
        if peak_kw is None:
            # Estimate peak demand from monthly usage and load factor
            # Assume 30 days, 24 hours
            avg_kw = monthly_kwh / (30 * 24)
            peak_kw = avg_kw / load_factor if load_factor > 0 else avg_kw * 2

        return UsageProfile(
            monthly_kwh=monthly_kwh,
            peak_kw=peak_kw,
            load_factor=load_factor
        )