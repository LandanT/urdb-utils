"""Command line interface for URDB county rates utility."""

import logging
import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from . import __version__
from .aggregate import RateAggregator
from .crosswalk import UtilityCountyCrosswalk
from .export import DataExporter
from .pricing import RatePricer, UsageProfile
from .tariffs import TariffSelector
from .urdb_client import URDBClient

# Initialize Typer app
app = typer.Typer(
    name="urdb-rates",
    help="Tools for mapping and analyzing electricity rates across U.S. counties using the DOE OpenEI URDB.",
    add_completion=False
)

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    version: bool = typer.Option(False, "--version", help="Show version and exit")
):
    """URDB County Rates - Electricity rate analysis tool."""
    if version:
        console.print(f"urdb-county-rates {__version__}")
        raise typer.Exit()

    setup_logging(verbose)


@app.command()
def fetch(
    api_key: str = typer.Option(..., "--api-key", "-k", help="OpenEI API key", envvar="OPENEI_API_KEY"),
    utility_ids: Optional[List[int]] = typer.Option(None, "--utility", "-u", help="Utility EIA IDs to fetch"),
    customer_class: Optional[str] = typer.Option("residential", "--class", "-c", help="Customer class (residential, commercial, industrial)"),
    state_codes: Optional[List[str]] = typer.Option(None, "--state", "-s", help="State codes to include"),
    output_dir: str = typer.Option("./data", "--output", "-o", help="Output directory"),
    cache_hours: int = typer.Option(24, "--cache-hours", help="Cache TTL in hours"),
    approved_only: bool = typer.Option(True, "--approved-only/--include-unapproved", help="Include only approved tariffs"),
    as_of_date: Optional[str] = typer.Option(None, "--as-of", help="As-of date (YYYY-MM-DD)")
):
    """Fetch tariff data from URDB API."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        # Initialize client
        task = progress.add_task("Initializing URDB client...", total=None)
        client = URDBClient(
            api_key=api_key,
            cache_dir=output_path / "cache",
            cache_ttl_hours=cache_hours
        )

        # Fetch tariffs
        progress.update(task, description="Fetching tariffs from URDB...")

        tariffs = []
        if utility_ids:
            for utility_id in utility_ids:
                utility_tariffs = client.get_rates(
                    eiaid=utility_id,
                    sector=customer_class,
                    approved=approved_only
                )
                tariffs.extend(utility_tariffs)
        else:
            tariffs = client.get_rates(
                sector=customer_class,
                approved=approved_only,
                limit=5000  # Increase limit for broader fetch
            )

        progress.update(task, description=f"Fetched {len(tariffs)} tariffs")

        # Filter tariffs
        if tariffs:
            selector = TariffSelector()
            criteria = selector.create_criteria(
                customer_class=customer_class,
                as_of_date=as_of_date,
                approved_only=approved_only,
                utility_ids=utility_ids,
                state_codes=state_codes
            )
            filtered_tariffs = selector.filter_tariffs(tariffs, criteria)
            progress.update(task, description=f"Filtered to {len(filtered_tariffs)} tariffs")
        else:
            filtered_tariffs = []

        # Export raw data
        if filtered_tariffs:
            import json
            raw_data = [tariff.model_dump() for tariff in filtered_tariffs]
            output_file = output_path / f"raw_tariffs_{customer_class}.json"
            with open(output_file, 'w') as f:
                json.dump(raw_data, f, indent=2, default=str)

            progress.update(task, description=f"Exported {len(filtered_tariffs)} tariffs to {output_file}")

        progress.remove_task(task)

    console.print(f"✅ Fetched and saved {len(filtered_tariffs)} tariffs to {output_dir}")


@app.command()
def build(
    crosswalk_file: str = typer.Option(..., "--crosswalk", "-x", help="Path to utility-county crosswalk CSV"),
    tariffs_file: str = typer.Option(..., "--tariffs", "-t", help="Path to raw tariffs JSON file"),
    customer_class: str = typer.Option("residential", "--class", "-c", help="Customer class to process"),
    output_dir: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    weighting_method: str = typer.Option("simple", "--weights", "-w", help="Weighting method (simple, sales, meters)"),
    monthly_usage: float = typer.Option(1000.0, "--usage", help="Monthly usage in kWh for rate calculations"),
    peak_demand: Optional[float] = typer.Option(None, "--peak-kw", help="Peak demand in kW"),
    formats: List[str] = typer.Option(["csv"], "--format", "-f", help="Export formats (csv, parquet, json, geojson)")
):
    """Build county and state rate datasets from raw tariff data."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        # Load crosswalk
        task = progress.add_task("Loading crosswalk...", total=None)
        try:
            crosswalk = UtilityCountyCrosswalk(crosswalk_file)
            progress.update(task, description=f"Loaded crosswalk with {len(crosswalk.mappings)} mappings")
        except Exception as e:
            console.print(f"❌ Failed to load crosswalk: {e}")
            raise typer.Exit(1)

        # Load tariffs
        progress.update(task, description="Loading tariffs...")
        try:
            import json
            from .urdb_client import URDBRateStructure

            with open(tariffs_file, 'r') as f:
                tariff_data = json.load(f)

            tariffs = [URDBRateStructure(**item) for item in tariff_data]
            progress.update(task, description=f"Loaded {len(tariffs)} tariffs")
        except Exception as e:
            console.print(f"❌ Failed to load tariffs: {e}")
            raise typer.Exit(1)

        # Calculate effective rates
        progress.update(task, description="Calculating effective rates...")
        pricer = RatePricer()

        # Create usage profile
        if peak_demand is None:
            usage_profile = RatePricer.create_usage_profile(monthly_usage)
        else:
            usage_profile = UsageProfile(monthly_kwh=monthly_usage, peak_kw=peak_demand)

        effective_rates = []
        for tariff in tariffs:
            try:
                rate = pricer.calculate_effective_rate(tariff, usage_profile, customer_class)
                effective_rates.append(rate)
            except Exception as e:
                logging.warning(f"Failed to calculate rate for {tariff.label}: {e}")
                continue

        progress.update(task, description=f"Calculated {len(effective_rates)} effective rates")

        # Aggregate to counties
        progress.update(task, description="Aggregating to county level...")
        aggregator = RateAggregator(crosswalk, weighting_method)
        county_rates = aggregator.aggregate_to_counties(effective_rates, [customer_class])

        # Aggregate to states
        progress.update(task, description="Aggregating to state level...")
        state_rates = aggregator.aggregate_to_states(county_rates, [customer_class])

        progress.update(task, description=f"Aggregated to {len(county_rates)} counties, {len(state_rates)} states")

        # Export data
        progress.update(task, description="Exporting data...")
        exporter = DataExporter(output_path)

        all_files = {}

        # Export effective rates
        if effective_rates:
            files = exporter.export_effective_rates(
                effective_rates,
                f"effective_rates_{customer_class}",
                formats
            )
            all_files.update(files)

        # Export county rates
        if county_rates:
            files = exporter.export_county_rates(
                county_rates,
                f"county_rates_{customer_class}",
                formats,
                include_geometry='geojson' in formats
            )
            all_files.update(files)

        # Export state rates
        if state_rates:
            files = exporter.export_state_rates(
                state_rates,
                f"state_rates_{customer_class}",
                formats
            )
            all_files.update(files)

        # Export summary report
        summary_path = exporter.export_summary_report(
            county_rates, state_rates, effective_rates,
            f"summary_{customer_class}"
        )
        all_files['summary'] = summary_path

        progress.remove_task(task)

    # Display results
    table = Table(title="Export Results")
    table.add_column("Dataset", style="cyan")
    table.add_column("Records", style="magenta")
    table.add_column("Files", style="green")

    table.add_row("Effective Rates", str(len(effective_rates)), str(len([f for f in all_files.keys() if 'effective' in str(f)])))
    table.add_row("County Rates", str(len(county_rates)), str(len([f for f in all_files.keys() if 'county' in str(f)])))
    table.add_row("State Rates", str(len(state_rates)), str(len([f for f in all_files.keys() if 'state' in str(f)])))

    console.print(table)
    console.print(f"✅ Built and exported datasets to {output_dir}")


@app.command()
def export(
    data_file: str = typer.Option(..., "--data", "-d", help="Path to data file (JSON)"),
    data_type: str = typer.Option(..., "--type", "-t", help="Data type (effective, county, state)"),
    output_dir: str = typer.Option("./output", "--output", "-o", help="Output directory"),
    formats: List[str] = typer.Option(["csv"], "--format", "-f", help="Export formats"),
    filename: Optional[str] = typer.Option(None, "--name", "-n", help="Output filename base")
):
    """Export data to various formats."""

    import json

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"❌ Failed to load data file: {e}")
        raise typer.Exit(1)

    # Set filename
    if filename is None:
        filename = f"{data_type}_rates"

    # Export data
    exporter = DataExporter(output_path)

    try:
        if data_type == "effective":
            from .pricing import EffectiveRate
            rates = [EffectiveRate(**item) for item in data]
            files = exporter.export_effective_rates(rates, filename, formats)

        elif data_type == "county":
            from .aggregate import CountyRate
            rates = [CountyRate(**item) for item in data]
            files = exporter.export_county_rates(rates, filename, formats)

        elif data_type == "state":
            from .aggregate import StateRate
            rates = [StateRate(**item) for item in data]
            files = exporter.export_state_rates(rates, filename, formats)

        else:
            console.print(f"❌ Unknown data type: {data_type}")
            raise typer.Exit(1)

        console.print(f"✅ Exported {len(data)} records to {len(files)} files in {output_dir}")

    except Exception as e:
        console.print(f"❌ Export failed: {e}")
        raise typer.Exit(1)


@app.command()
def validate(
    crosswalk_file: str = typer.Option(..., "--crosswalk", "-x", help="Path to crosswalk CSV"),
    data_files: List[str] = typer.Option(..., "--data", "-d", help="Path to data files"),
    output_dir: str = typer.Option("./validation", "--output", "-o", help="Output directory")
):
    """Validate crosswalk and rate data."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with console.status("Running validation..."):

        # Validate crosswalk
        try:
            crosswalk = UtilityCountyCrosswalk(crosswalk_file)

            # Check FIPS codes
            invalid_fips = crosswalk.validate_fips_codes()

            # Get coverage stats
            stats = crosswalk.get_coverage_stats()

            console.print("\n📊 Crosswalk Validation Results:")
            table = Table()
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            for key, value in stats.items():
                table.add_row(key.replace('_', ' ').title(), str(value))

            console.print(table)

            if invalid_fips:
                console.print(f"⚠️  Found {len(invalid_fips)} invalid FIPS codes")
                for fips in invalid_fips[:5]:  # Show first 5
                    console.print(f"   - {fips}")
                if len(invalid_fips) > 5:
                    console.print(f"   ... and {len(invalid_fips) - 5} more")

        except Exception as e:
            console.print(f"❌ Crosswalk validation failed: {e}")

    console.print("✅ Validation complete")


@app.command()
def enhance_crosswalk(
    crosswalk_file: str = typer.Option(..., "--crosswalk", "-x", help="Path to utility-county crosswalk CSV"),
    geo_data_file: str = typer.Option(..., "--geo-data", "-g", help="Path to county geographic data CSV"),
    output_file: str = typer.Option(..., "--output", "-o", help="Path to enhanced crosswalk output CSV"),
    show_summary: bool = typer.Option(True, "--summary/--no-summary", help="Show geographic data summary")
):
    """Enhance crosswalk with county geographic data (lat/lng, population, climate zones)."""
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Load crosswalk
        task = progress.add_task("Loading crosswalk...", total=None)
        try:
            crosswalk = UtilityCountyCrosswalk(crosswalk_file)
            progress.update(task, description=f"Loaded crosswalk with {len(crosswalk.mappings)} mappings")
        except Exception as e:
            console.print(f"❌ Failed to load crosswalk: {e}")
            raise typer.Exit(1)
        
        # Load geographic data
        progress.update(task, description="Loading county geographic data...")
        try:
            crosswalk.load_county_geographic_data(geo_data_file)
            progress.update(task, description="Enhanced crosswalk with geographic data")
        except Exception as e:
            console.print(f"❌ Failed to load geographic data: {e}")
            raise typer.Exit(1)
        
        # Save enhanced crosswalk
        progress.update(task, description="Saving enhanced crosswalk...")
        try:
            crosswalk.save_to_csv(output_file)
            progress.update(task, description=f"Saved enhanced crosswalk to {output_file}")
        except Exception as e:
            console.print(f"❌ Failed to save enhanced crosswalk: {e}")
            raise typer.Exit(1)
        
        progress.remove_task(task)
    
    # Show summary if requested
    if show_summary:
        console.print("\n🌍 Geographic Data Summary:")
        summary = crosswalk.get_geographic_summary()
        
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_column("Coverage", style="green")
        
        table.add_row("Total Counties", str(summary['total_counties']), "100%")
        table.add_row("With Coordinates", str(summary['counties_with_coordinates']), f"{summary['coordinate_coverage_pct']}%")
        table.add_row("With Population", str(summary['counties_with_population']), f"{summary['population_coverage_pct']}%")
        table.add_row("With Climate Zone", str(summary['counties_with_climate_zone']), f"{summary['climate_coverage_pct']}%")
        table.add_row("Climate Zones", str(summary['unique_climate_zones']), "-")
        
        console.print(table)
        
        if summary['climate_zones']:
            console.print(f"\n🌡️  Climate Zones: {', '.join(summary['climate_zones'][:10])}")
            if len(summary['climate_zones']) > 10:
                console.print(f"   ... and {len(summary['climate_zones']) - 10} more")
    
    console.print(f"✅ Enhanced crosswalk saved to {output_file}")


@app.command()
def info():
    """Show information about the URDB utility."""

    console.print(f"\n🔌 URDB County Rates v{__version__}")
    console.print("\nThis tool helps you:")
    console.print("• Fetch electricity tariff data from the DOE OpenEI URDB")
    console.print("• Calculate effective rates for different customer classes")
    console.print("• Map utility rates to counties using crosswalk data")
    console.print("• Aggregate rates to county and state levels")
    console.print("• Export data in multiple formats (CSV, Parquet, GeoJSON)")
    console.print("• Enhance crosswalks with geographic data (coordinates, population, climate)")

    console.print("\n📋 Typical workflow:")
    console.print("1. [cyan]enhance-crosswalk[/cyan] - Add geographic data to crosswalk")
    console.print("2. [cyan]fetch[/cyan] - Download tariff data from URDB")
    console.print("3. [cyan]build[/cyan] - Process and aggregate to county/state level")
    console.print("4. [cyan]export[/cyan] - Export to desired formats")
    console.print("5. [cyan]validate[/cyan] - Check data quality")

    console.print("\n🔑 You'll need:")
    console.print("• OpenEI API key (free at https://openei.org/services/api/signup/)")
    console.print("• Utility-county crosswalk CSV file")
    console.print("• County geographic data CSV (optional, for enhanced analysis)")

    console.print(f"\n📚 For more help: [cyan]urdb-rates COMMAND --help[/cyan]")


if __name__ == "__main__":
    app()