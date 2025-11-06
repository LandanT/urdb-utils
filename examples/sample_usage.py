#!/usr/bin/env python3
"""
Sample usage script for URDB County Rates utility.

This script demonstrates how to use the library programmatically.
"""

import logging
from pathlib import Path

from urdb_county_rates.urdb_client import URDBClient
from urdb_county_rates.crosswalk import UtilityCountyCrosswalk
from urdb_county_rates.tariffs import TariffSelector
from urdb_county_rates.pricing import RatePricer
from urdb_county_rates.aggregate import RateAggregator
from urdb_county_rates.export import DataExporter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Example workflow for processing URDB data."""

    # Configuration
    API_KEY = "your_openei_api_key_here"  # Get from https://openei.org/services/api/signup/
    OUTPUT_DIR = Path("./example_output")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Step 1: Initialize URDB client
    logger.info("Initializing URDB client...")
    client = URDBClient(
        api_key=API_KEY,
        cache_dir=OUTPUT_DIR / "cache",
        cache_ttl_hours=24
    )

    # Step 2: Create sample crosswalk (in practice, load from CSV)
    logger.info("Setting up utility-county crosswalk...")
    crosswalk = UtilityCountyCrosswalk.create_sample_crosswalk()

    # Save sample crosswalk for reference
    crosswalk.save_to_csv(OUTPUT_DIR / "sample_crosswalk.csv")
    logger.info(f"Saved sample crosswalk with {len(crosswalk.mappings)} mappings")

    # Step 3: Fetch tariffs from URDB
    logger.info("Fetching tariffs from URDB...")
    try:
        # Fetch residential tariffs for California utilities
        tariffs = client.get_rates(
            sector="residential",
            approved=True,
            limit=100  # Limit for example
        )
        logger.info(f"Fetched {len(tariffs)} tariffs")

        if not tariffs:
            logger.error("No tariffs found. Check your API key and try again.")
            return

    except Exception as e:
        logger.error(f"Failed to fetch tariffs: {e}")
        logger.info("Using sample mode - creating mock data for demonstration")
        tariffs = []  # Would use mock data in real implementation
        return

    # Step 4: Filter and select tariffs
    logger.info("Filtering tariffs...")
    selector = TariffSelector()

    criteria = selector.create_criteria(
        customer_class="residential",
        approved_only=True,
        exclude_expired=True
    )

    filtered_tariffs = selector.filter_tariffs(tariffs, criteria)
    logger.info(f"Filtered to {len(filtered_tariffs)} tariffs")

    # Step 5: Calculate effective rates
    logger.info("Calculating effective rates...")
    pricer = RatePricer()

    effective_rates = pricer.calculate_bulk_rates(filtered_tariffs)
    logger.info(f"Calculated {len(effective_rates)} effective rates")

    # Step 6: Aggregate to county and state levels
    logger.info("Aggregating to county level...")
    aggregator = RateAggregator(crosswalk, weighting_method="simple")

    county_rates = aggregator.aggregate_to_counties(effective_rates, ["residential"])
    state_rates = aggregator.aggregate_to_states(county_rates, ["residential"])

    logger.info(f"Aggregated to {len(county_rates)} counties and {len(state_rates)} states")

    # Step 7: Export results
    logger.info("Exporting data...")
    exporter = DataExporter(OUTPUT_DIR)

    # Export all data formats
    all_files = exporter.create_data_package(
        county_rates=county_rates,
        state_rates=state_rates,
        effective_rates=effective_rates,
        package_name="example_urdb_data"
    )

    logger.info(f"Exported {len(all_files)} files:")
    for name, filepath in all_files.items():
        logger.info(f"  - {name}: {filepath}")

    # Step 8: Generate validation report
    logger.info("Validating data...")
    validation_issues = aggregator.validate_aggregation(county_rates)

    if validation_issues:
        validation_path = exporter.export_validation_report(
            validation_issues,
            "example_validation"
        )
        logger.info(f"Validation report saved to {validation_path}")

        for category, issues in validation_issues.items():
            logger.warning(f"{category}: {len(issues)} issues")
    else:
        logger.info("No validation issues found!")

    # Step 9: Show summary
    summary = aggregator.get_aggregation_summary(county_rates, state_rates)

    logger.info("\n=== SUMMARY ===")
    logger.info(f"Counties: {summary['county_level']['total_counties']}")
    logger.info(f"States: {summary['county_level']['states_represented']}")
    logger.info(f"Average utilities per county: {summary['county_level']['avg_utilities_per_county']:.1f}")

    if summary['county_level'].get('rate_stats'):
        stats = summary['county_level']['rate_stats']
        logger.info(f"Rate range: {stats['min']:.2f} - {stats['max']:.2f} ¢/kWh")
        logger.info(f"Average rate: {stats['mean']:.2f} ¢/kWh")

    logger.info(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()