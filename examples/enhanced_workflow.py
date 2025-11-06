#!/usr/bin/env python3
"""
This script is an example of how to process URDB data that achieves a similar outcome to my original workflow that was used as the seed for this library.
It integrates geographic utility mapping, tariff fetching and filtering, enhanced rate calculations, county-level aggregation with imputation, and natural gas rate comparison.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from urdb_county_rates.urdb_client import URDBClient
from urdb_county_rates.geo_mapping import GeographicUtilityMapper, EnhancedUtilityCountyCrosswalk
from urdb_county_rates.enhanced_pricing import EnhancedRatePricer
from urdb_county_rates.tariffs import TariffSelector
from urdb_county_rates.aggregate import RateAggregator
from urdb_county_rates.export import DataExporter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedURDBWorkflow:
    """
    Complete workflow that integrates my existing processing patterns.
    """

    def __init__(self,
                 api_key: str,
                 output_dir: str = "./enhanced_output",
                 energy_min: float = 0.0,
                 energy_max: float = 2.0,
                 target_kw: Optional[float] = None,
                 cutoff_start: str = "2023-01-01",
                 min_cutoff_year: int = 2016,
                 tou_only: bool = False,
                 keep_undated: bool = True):
        """
        Initialize the enhanced workflow.

        Args:
            api_key: OpenEI API key
            output_dir: Output directory for results
            energy_min: Minimum plausible energy rate ($/kWh)
            energy_max: Maximum plausible energy rate ($/kWh)
            target_kw: Target kW for demand band screening
            cutoff_start: Initial date cutoff
            min_cutoff_year: Minimum year to fall back to
            tou_only: Process only TOU rates
            keep_undated: Keep rates without dates
        """
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.energy_min = energy_min
        self.energy_max = energy_max
        self.target_kw = target_kw
        self.cutoff_start = pd.Timestamp(cutoff_start)
        self.min_cutoff_year = min_cutoff_year
        self.tou_only = tou_only
        self.keep_undated = keep_undated

        # Initialize components
        self.client = URDBClient(
            api_key=api_key,
            cache_dir=self.output_dir / "cache",
            cache_ttl_hours=24
        )

        self.geo_mapper = None
        self.enhanced_pricer = EnhancedRatePricer(
            energy_min=energy_min,
            energy_max=energy_max
        )

        self.selector = TariffSelector()
        self.exporter = DataExporter(self.output_dir)

        logger.info(f"Initialized enhanced workflow with output dir: {self.output_dir}")

    def setup_geographic_mapping(self,
                               county_csv_path: str,
                               force_refresh: bool = False) -> pd.DataFrame:
        """
        Set up geographic utility mapping similar to find_electric_utility_by_geo_location.py from old workflow.

        Args:
            county_csv_path: Path to county CSV with lat/lng data
            force_refresh: Whether to force refresh of mapping data

        Returns:
            DataFrame with county-utility mappings
        """
        logger.info("Setting up geographic utility mapping...")

        # First, get known utilities from URDB to improve name matching
        logger.info("Fetching known utilities from URDB...")
        known_utilities = set()
        try:
            utilities = self.client.get_utilities(limit=5000)
            known_utilities = {u.label for u in utilities if u.label}
            logger.info(f"Found {len(known_utilities)} known utilities")
        except Exception as e:
            logger.warning(f"Failed to fetch utilities: {e}")

        # Initialize geographic mapper
        self.geo_mapper = GeographicUtilityMapper(
            api_key=self.api_key,
            known_urdb_utilities=known_utilities
        )

        # Create or load county-utility mapping
        mapping_path = self.output_dir / "county_utilities_map.csv"

        if force_refresh or not mapping_path.exists():
            logger.info("Creating county-utility mapping via geographic lookup...")
            county_utils = self.geo_mapper.create_county_utility_mapping(
                county_csv_path=county_csv_path,
                output_csv_path=str(mapping_path),
                save_every=25,
                resume=True
            )
        else:
            logger.info(f"Loading existing mapping from {mapping_path}")
            county_utils = pd.read_csv(mapping_path, dtype={"county_fips": str})

        return county_utils

    def fetch_and_filter_tariffs(self,
                               customer_class: str = "commercial",
                               utility_list: Optional[List[str]] = None,
                               limit: int = 5000) -> List:
        """
        Fetch and filter tariffs similar to my existing workflow.

        Args:
            customer_class: Customer class to fetch
            utility_list: Optional list of specific utilities
            limit: Maximum tariffs to fetch

        Returns:
            List of filtered tariffs
        """
        logger.info(f"Fetching {customer_class} tariffs...")

        # Fetch tariffs
        if utility_list:
            all_tariffs = []
            for utility in utility_list:
                tariffs = self.client.get_rates(
                    utility=utility,
                    sector=customer_class,
                    approved=True,
                    limit=limit
                )
                all_tariffs.extend(tariffs)
        else:
            all_tariffs = self.client.get_rates(
                sector=customer_class,
                approved=True,
                limit=limit
            )

        logger.info(f"Fetched {len(all_tariffs)} raw tariffs")

        # Filter by TOU if specified
        if self.tou_only:
            tou_tariffs = []
            for tariff in all_tariffs:
                tariff_dict = tariff.model_dump()
                is_tou = self.enhanced_pricer.is_time_of_use(
                    tariff_dict.get('energyweekdayschedule'),
                    tariff_dict.get('energyweekendschedule')
                )
                if is_tou:
                    tou_tariffs.append(tariff)
            all_tariffs = tou_tariffs
            logger.info(f"Filtered to {len(all_tariffs)} TOU tariffs")

        # Apply date filtering with fallback (similar to my logic)
        filtered_tariffs = self._apply_date_filtering_with_fallback(all_tariffs)

        # Apply demand band screening if specified
        if self.target_kw is not None:
            filtered_tariffs = self._apply_demand_band_screening(filtered_tariffs)

        logger.info(f"Final filtered tariffs: {len(filtered_tariffs)}")
        return filtered_tariffs

    def _apply_date_filtering_with_fallback(self, tariffs: List) -> List:
        """Apply progressive date filtering with fallback."""
        cutoff = self.cutoff_start
        attempts = []

        while True:
            # Create criteria for this cutoff
            criteria = self.selector.create_criteria(
                as_of_date=cutoff.strftime("%Y-%m-%d"),
                approved_only=True,
                exclude_expired=True
            )

            filtered = self.selector.filter_tariffs(tariffs, criteria)
            attempts.append((cutoff, len(filtered)))

            if filtered or cutoff.year < self.min_cutoff_year:
                logger.info(f"Date filtering attempts: {[(d.strftime('%Y-%m-%d'), n) for d, n in attempts]}")

                if not filtered and cutoff.year < self.min_cutoff_year:
                    # Final fallback: latest per utility
                    logger.info("Using latest tariff per utility as final fallback")
                    utility_latest = {}
                    for tariff in tariffs:
                        utility = tariff.utility
                        effective_date = self.selector.parse_date(tariff.effective)
                        if utility not in utility_latest or (effective_date and
                            (not utility_latest[utility][1] or effective_date > utility_latest[utility][1])):
                            utility_latest[utility] = (tariff, effective_date)

                    filtered = [tariff for tariff, _ in utility_latest.values()]

                return filtered

            cutoff = cutoff - pd.DateOffset(years=1)

    def _apply_demand_band_screening(self, tariffs: List) -> List:
        """Apply demand band screening."""
        if not self.target_kw:
            return tariffs

        filtered = []
        for tariff in tariffs:
            tariff_dict = tariff.model_dump()

            # Check demand limits (simplified - could be more sophisticated)
            min_demand = tariff_dict.get('mindemand') or tariff_dict.get('min_demand')
            max_demand = tariff_dict.get('maxdemand') or tariff_dict.get('max_demand')

            include = True
            if min_demand is not None:
                try:
                    if float(min_demand) > self.target_kw:
                        include = False
                except (ValueError, TypeError):
                    pass

            if max_demand is not None:
                try:
                    if self.target_kw >= float(max_demand):
                        include = False
                except (ValueError, TypeError):
                    pass

            if include:
                filtered.append(tariff)

        logger.info(f"Demand band screening: {len(filtered)}/{len(tariffs)} tariffs passed")
        return filtered

    def calculate_enhanced_rates(self,
                               tariffs: List,
                               customer_class: str = "commercial",
                               monthly_usage: float = 5000.0,
                               peak_kw: Optional[float] = None) -> List:
        """
        Calculate effective rates with enhanced parsing.

        Args:
            tariffs: List of filtered tariffs
            customer_class: Customer class
            monthly_usage: Monthly usage in kWh
            peak_kw: Peak demand in kW

        Returns:
            List of EffectiveRate objects
        """
        logger.info("Calculating enhanced effective rates...")

        # Create usage profile
        if peak_kw is None:
            usage_profile = self.enhanced_pricer.create_usage_profile(monthly_usage)
        else:
            from urdb_county_rates.pricing import UsageProfile
            usage_profile = UsageProfile(monthly_kwh=monthly_usage, peak_kw=peak_kw)

        effective_rates = []
        failed_count = 0

        for tariff in tariffs:
            try:
                rate = self.enhanced_pricer.calculate_enhanced_effective_rate(
                    tariff, usage_profile, customer_class
                )
                effective_rates.append(rate)
            except Exception as e:
                logger.debug(f"Failed to calculate rate for {tariff.label}: {e}")
                failed_count += 1

        logger.info(f"Calculated {len(effective_rates)} rates, {failed_count} failed")

        # Log quality statistics
        self._log_quality_statistics(effective_rates)

        return effective_rates

    def _log_quality_statistics(self, effective_rates: List) -> None:
        """Log data quality statistics."""
        if not effective_rates:
            return

        flag_counts = {}
        for rate in effective_rates:
            for flag in rate.data_quality_flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

        logger.info("Data quality statistics:")
        for flag, count in sorted(flag_counts.items()):
            pct = (count / len(effective_rates)) * 100
            logger.info(f"  {flag}: {count} ({pct:.1f}%)")

    def create_county_analysis(self,
                             effective_rates: List,
                             county_utils_df: pd.DataFrame,
                             customer_class: str = "commercial") -> pd.DataFrame:
        """
        Create county-level analysis with geographic imputation.

        Args:
            effective_rates: List of effective rates
            county_utils_df: County-utility mapping DataFrame
            customer_class: Customer class

        Returns:
            Complete county analysis DataFrame
        """
        logger.info("Creating county-level analysis...")

        # Create enhanced crosswalk from county utilities
        crosswalk = EnhancedUtilityCountyCrosswalk()

        # Process county-utility mappings
        for _, row in county_utils_df.iterrows():
            if pd.isna(row['utilities']) or not row['utilities'].strip():
                continue

            utilities = [u.strip() for u in str(row['utilities']).split(';') if u.strip()]

            for utility_name in utilities:
                # Find matching utility ID from effective rates
                utility_id = None
                for rate in effective_rates:
                    if rate.utility_name == utility_name:
                        utility_id = rate.utility_id
                        break

                if utility_id:
                    crosswalk.add_mapping(
                        utility_id=utility_id,
                        utility_name=utility_name,
                        state=row.get('state_id', ''),
                        county_fips=str(row['county_fips']).zfill(5),
                        county_name=row.get('county', ''),
                        weight=1.0  # Could be enhanced with sales/meter weights
                    )

        # Add geographic coordinates to crosswalk mappings
        for mapping in crosswalk.mappings:
            county_info = county_utils_df[
                county_utils_df['county_fips'] == mapping.county_fips
            ]
            if not county_info.empty:
                row = county_info.iloc[0]
                # Add lat/lng to the mapping (extend the data structure as needed)
                # This would require enhancing the UtilityCountyMapping model

        # Aggregate to counties
        aggregator = RateAggregator(crosswalk, weighting_method="simple")
        county_rates = aggregator.aggregate_to_counties(effective_rates, [customer_class])

        # Convert to DataFrame
        county_df = aggregator.county_rates_to_dataframe(county_rates)

        # Add all counties from original data (for imputation)
        all_counties = county_utils_df[['county_fips', 'county', 'state_id', 'state_name', 'lat', 'lng']].copy()
        all_counties['county_fips'] = all_counties['county_fips'].astype(str).str.zfill(5)

        # Merge to get complete county list
        complete_county_df = all_counties.merge(
            county_df,
            left_on='county_fips',
            right_on='county_fips',
            how='left',
            suffixes=('', '_agg')
        )

        # Apply geographic imputation for missing counties
        crosswalk.build_utility_centroids()
        complete_county_df = crosswalk.impute_missing_counties(
            complete_county_df,
            prefer_same_state=True
        )

        logger.info(f"Created analysis for {len(complete_county_df)} counties")
        return complete_county_df

    def add_natural_gas_comparison(self,
                                 county_df: pd.DataFrame,
                                 ng_rates_csv: Optional[str] = None) -> pd.DataFrame:
        """
        Add natural gas rate comparison similar to my existing workflow.

        Args:
            county_df: County analysis DataFrame
            ng_rates_csv: Path to natural gas rates CSV

        Returns:
            DataFrame with NG comparison columns
        """
        if not ng_rates_csv or not os.path.exists(ng_rates_csv):
            logger.warning("Natural gas rates file not found, skipping NG comparison")
            return county_df

        try:
            ng_rates = pd.read_csv(ng_rates_csv)
            if "State" in ng_rates.columns and "$/Thousand Cubic Feet" in ng_rates.columns:
                ng_mapping = ng_rates.set_index("State")['$/Thousand Cubic Feet']
                county_df["$/Thousand Cubic Feet NG"] = county_df["state_name"].map(ng_mapping)

                # Calculate ratios (electricity to natural gas energy cost)
                def safe_ratio(elec_kwh, ng_mcf):
                    try:
                        if pd.isna(elec_kwh) or pd.isna(ng_mcf) or ng_mcf <= 0:
                            return np.nan
                        return (elec_kwh / 3412.0) / (ng_mcf / 1038000.0)
                    except:
                        return np.nan

                county_df['kwh_all_ratio'] = county_df.apply(
                    lambda r: safe_ratio(r.get('weighted_rate_cents_per_kwh'), r.get('$/Thousand Cubic Feet NG')),
                    axis=1
                )

                logger.info("Added natural gas comparison data")

        except Exception as e:
            logger.error(f"Failed to add natural gas comparison: {e}")

        return county_df

    def run_complete_workflow(self,
                            county_csv_path: str,
                            customer_class: str = "commercial",
                            monthly_usage: float = 5000.0,
                            peak_kw: Optional[float] = None,
                            ng_rates_csv: Optional[str] = None,
                            force_refresh_mapping: bool = False) -> Dict[str, Path]:
        """
        Run the complete enhanced workflow.

        Args:
            county_csv_path: Path to county CSV with lat/lng
            customer_class: Customer class to analyze
            monthly_usage: Monthly usage in kWh
            peak_kw: Peak demand in kW
            ng_rates_csv: Optional natural gas rates CSV
            force_refresh_mapping: Force refresh of geographic mapping

        Returns:
            Dictionary of output file paths
        """
        logger.info("Starting complete enhanced workflow...")

        # Step 1: Set up geographic mapping
        county_utils = self.setup_geographic_mapping(
            county_csv_path,
            force_refresh=force_refresh_mapping
        )

        # Step 2: Get list of utilities to focus on
        utility_list = []
        for utilities_str in county_utils['utilities'].dropna():
            utilities = [u.strip() for u in str(utilities_str).split(';') if u.strip()]
            utility_list.extend(utilities)
        utility_list = list(set(utility_list))
        logger.info(f"Focusing on {len(utility_list)} unique utilities")

        # Step 3: Fetch and filter tariffs
        tariffs = self.fetch_and_filter_tariffs(
            customer_class=customer_class,
            utility_list=utility_list[:100]  # Limit for demonstration
        )

        # Step 4: Calculate enhanced rates
        effective_rates = self.calculate_enhanced_rates(
            tariffs=tariffs,
            customer_class=customer_class,
            monthly_usage=monthly_usage,
            peak_kw=peak_kw
        )

        # Step 5: Create county analysis
        county_analysis = self.create_county_analysis(
            effective_rates=effective_rates,
            county_utils_df=county_utils,
            customer_class=customer_class
        )

        # Step 6: Add natural gas comparison
        if ng_rates_csv:
            county_analysis = self.add_natural_gas_comparison(county_analysis, ng_rates_csv)

        # Step 7: Export all results
        output_files = {}

        # Export effective rates
        if effective_rates:
            files = self.exporter.export_effective_rates(
                effective_rates,
                f"enhanced_effective_rates_{customer_class}",
                ['csv', 'json']
            )
            output_files.update(files)

        # Export county analysis
        county_csv_path = self.output_dir / f"enhanced_county_analysis_{customer_class}.csv"
        county_analysis.to_csv(county_csv_path, index=False)
        output_files['county_analysis'] = county_csv_path

        # Export summary statistics
        self._create_summary_report(county_analysis, effective_rates, customer_class)

        logger.info(f"Workflow complete! Output files: {len(output_files)}")
        for name, path in output_files.items():
            logger.info(f"  {name}: {path}")

        return output_files

    def _create_summary_report(self, county_df: pd.DataFrame, effective_rates: List, customer_class: str):
        """Create summary report similar to my existing workflow."""
        summary = {
            'workflow_config': {
                'customer_class': customer_class,
                'energy_min': self.energy_min,
                'energy_max': self.energy_max,
                'target_kw': self.target_kw,
                'tou_only': self.tou_only,
                'cutoff_start': self.cutoff_start.strftime('%Y-%m-%d')
            },
            'processing_stats': {
                'total_effective_rates': len(effective_rates),
                'total_counties': len(county_df),
                'counties_with_data': int((~county_df['weighted_rate_cents_per_kwh'].isna()).sum()),
                'counties_imputed': int(county_df.get('imputation_used', pd.Series(False)).sum())
            }
        }

        # Add rate statistics
        if 'weighted_rate_cents_per_kwh' in county_df.columns:
            rates = county_df['weighted_rate_cents_per_kwh'].dropna()
            if not rates.empty:
                summary['rate_statistics'] = {
                    'mean': float(rates.mean()),
                    'median': float(rates.median()),
                    'std': float(rates.std()),
                    'min': float(rates.min()),
                    'max': float(rates.max()),
                    'count': len(rates)
                }

        # Save report
        import json
        report_path = self.output_dir / f"enhanced_summary_{customer_class}.json"
        with open(report_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Summary report saved to {report_path}")


def main():
    """Example usage of the enhanced workflow."""

    # Configuration
    API_KEY = os.getenv("OPENEI_API_KEY")
    if not API_KEY:
        print("Please set OPENEI_API_KEY environment variable")
        return

    COUNTY_CSV = "./data/us_county_geo_information.csv"  # Your county data
    NG_RATES_CSV = "./data/eia_ng_prices_2023.csv"      # Optional NG rates

    # Initialize workflow
    workflow = EnhancedURDBWorkflow(
        api_key=API_KEY,
        output_dir="./enhanced_workflow_output",
        energy_min=0.0,
        energy_max=2.0,
        target_kw=None,  # Set to specific kW for demand screening
        tou_only=False,  # Set to True for TOU only
        keep_undated=True
    )

    # Run complete workflow
    try:
        output_files = workflow.run_complete_workflow(
            county_csv_path=COUNTY_CSV,
            customer_class="commercial",
            monthly_usage=5000.0,
            peak_kw=30.0,
            ng_rates_csv=NG_RATES_CSV,
            force_refresh_mapping=False
        )

        print("✅ Enhanced workflow completed successfully!")
        print("Output files:")
        for name, path in output_files.items():
            print(f"  - {name}: {path}")

    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        raise


if __name__ == "__main__":
    main()