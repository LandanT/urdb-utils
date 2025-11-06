"""
Example: Working with Enhanced Geographic Data in URDB County Rates

This example demonstrates how to integrate your county geographic data
into the standard URDB workflow for enhanced analysis and mapping.
"""

import logging
from pathlib import Path

from urdb_county_rates import (
    UtilityCountyCrosswalk,
    URDBClient,
    TariffSelector,
    RatePricer,
    RateAggregator,
    DataExporter
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Demonstrate geographic data integration workflow."""
    
    # Paths (adjust these to your actual file locations)
    data_dir = Path("../data")
    output_dir = Path("../output/geographic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Your data files
    county_utilities_file = data_dir / "county_utilities_map.csv"
    county_geo_file = data_dir / "us_county_geo_information.csv"
    
    print("🌍 URDB Geographic Workflow Example")
    print("=" * 50)
    
    # Step 1: Create Enhanced Crosswalk with Geographic Data
    print("\n1. Creating enhanced crosswalk with geographic data...")
    
    # Option A: Load your existing county-utilities mapping
    if county_utilities_file.exists():
        print(f"   Loading utility mappings from {county_utilities_file}")
        
        # Your county_utilities_map.csv has a different structure - let's adapt it
        crosswalk = load_custom_crosswalk(county_utilities_file)
    else:
        # Option B: Create sample crosswalk for demonstration
        print("   Creating sample crosswalk...")
        crosswalk = UtilityCountyCrosswalk.create_sample_crosswalk()
    
    # Load geographic data to enhance the crosswalk
    if county_geo_file.exists():
        print(f"   Enhancing with geographic data from {county_geo_file}")
        crosswalk.load_county_geographic_data(county_geo_file)
        
        # Show geographic data summary
        geo_summary = crosswalk.get_geographic_summary()
        print(f"   ✅ Enhanced {geo_summary['total_counties']} counties")
        print(f"   📍 {geo_summary['coordinate_coverage_pct']}% have coordinates")
        print(f"   👥 {geo_summary['population_coverage_pct']}% have population data")
        print(f"   🌡️  {geo_summary['climate_coverage_pct']}% have climate zones")
        print(f"   🗺️  {geo_summary['unique_climate_zones']} unique climate zones")
    
    # Step 2: Demonstrate Geographic Queries
    print("\n2. Demonstrating geographic queries...")
    
    # Find counties by climate zone
    if geo_summary.get('climate_zones'):
        sample_climate = geo_summary['climate_zones'][0]
        climate_counties = crosswalk.get_counties_in_climate_zone(sample_climate)
        print(f"   Counties in climate zone {sample_climate}: {len(climate_counties)}")
    
    # Get coordinates for a specific county
    sample_counties = list(crosswalk.get_all_counties())[:3]
    for county_fips in sample_counties:
        coords = crosswalk.get_county_coordinates(county_fips)
        if coords:
            lat, lng = coords
            print(f"   County {county_fips}: ({lat:.2f}, {lng:.2f})")
    
    # Step 3: Standard URDB Workflow with Enhanced Data
    print("\n3. Running standard URDB workflow...")
    
    # This would normally fetch from URDB, but for demo we'll create sample data
    sample_rates = create_sample_effective_rates()
    
    # Aggregate with enhanced crosswalk
    aggregator = RateAggregator(crosswalk, weighting_method="simple")
    county_rates = aggregator.aggregate_to_counties(sample_rates, ["residential"])
    
    print(f"   ✅ Aggregated to {len(county_rates)} counties")
    
    # Step 4: Export with Geographic Enhancement
    print("\n4. Exporting enhanced data...")
    
    exporter = DataExporter(output_dir)
    
    # Export to multiple formats including GeoJSON with real coordinates
    formats = ['csv', 'json', 'geojson']
    files = export_with_crosswalk(exporter, county_rates, crosswalk, formats)
    
    print("   📁 Exported files:")
    for fmt, filepath in files.items():
        print(f"   • {fmt.upper()}: {filepath}")
    
    # Step 5: Analysis Examples
    print("\n5. Geographic analysis examples...")
    
    # Analyze rates by climate zone
    if county_rates:
        analyze_by_climate_zone(county_rates, crosswalk)
    
    print("\n✅ Geographic workflow complete!")
    print(f"📂 Check outputs in: {output_dir}")


def load_custom_crosswalk(county_utilities_file: Path) -> UtilityCountyCrosswalk:
    """Load your custom county-utilities mapping format."""
    import pandas as pd
    
    # Read your county_utilities_map.csv 
    df = pd.read_csv(county_utilities_file)
    
    crosswalk = UtilityCountyCrosswalk()
    
    # Your format: county_fips, county, state_id, lat, lng, utilities (semicolon-separated)
    for _, row in df.iterrows():
        if pd.notna(row.get('utilities')):
            utilities = str(row['utilities']).split(';')
            
            for i, utility_name in enumerate(utilities):
                utility_name = utility_name.strip()
                if utility_name:
                    # Create a utility ID (in real usage, you'd map to actual EIA IDs)
                    utility_id = hash(utility_name) % 100000  # Simple ID generation
                    
                    crosswalk.add_mapping(
                        utility_id=utility_id,
                        utility_name=utility_name,
                        state=str(row['state_id']),
                        county_fips=str(row['county_fips']).zfill(5),
                        county_name=str(row['county']),
                        weight=1.0 / len(utilities),  # Equal weight for multiple utilities
                        data_source="county_utilities_map"
                    )
    
    return crosswalk


def create_sample_effective_rates():
    """Create sample effective rates for demonstration."""
    from urdb_county_rates.pricing import EffectiveRate, EffectiveRateCalculation, RateAssumptions
    
    # Sample effective rates for demonstration
    rates = []
    for i in range(10):
        rate = EffectiveRate(
            utility_id=i + 1,
            utility_name=f"Sample Utility {i + 1}",
            tariff_label=f"Residential Rate {i + 1}",
            customer_class="residential",
            calculation=EffectiveRateCalculation(
                rate_cents_per_kwh=12.0 + i,
                monthly_bill_dollars=120.0 + i * 10,
                annual_bill_dollars=(120.0 + i * 10) * 12
            ),
            assumptions=RateAssumptions(
                monthly_kwh=1000.0,
                peak_kw=5.0,
                load_factor=0.6
            ),
            data_quality_flags=[]
        )
        rates.append(rate)
    
    return rates


def export_with_crosswalk(exporter, county_rates, crosswalk, formats):
    """Export county rates with crosswalk data for enhanced GeoJSON."""
    
    # For GeoJSON, we need to modify the export to use crosswalk data
    files = {}
    
    for fmt in formats:
        if fmt == 'geojson':
            # Use enhanced GeoJSON export with crosswalk
            filepath = exporter._export_county_geojson(
                county_rates, 
                "enhanced_county_rates", 
                crosswalk
            )
            files['geojson'] = filepath
        else:
            # Use standard export for other formats
            std_files = exporter.export_county_rates(
                county_rates,
                f"enhanced_county_rates",
                [fmt],
                include_geometry=False
            )
            files.update(std_files)
    
    return files


def analyze_by_climate_zone(county_rates, crosswalk):
    """Example analysis: average rates by climate zone."""
    from collections import defaultdict
    
    zone_rates = defaultdict(list)
    
    for county_rate in county_rates:
        # Get climate zone for this county
        mappings = crosswalk.get_utilities_for_county(county_rate.county_fips)
        if mappings and mappings[0].climate_zone:
            climate_zone = mappings[0].climate_zone
            zone_rates[climate_zone].append(county_rate.weighted_rate_cents_per_kwh)
    
    print("   📊 Average rates by climate zone:")
    for zone, rates in zone_rates.items():
        if rates:
            avg_rate = sum(rates) / len(rates)
            print(f"   • {zone}: {avg_rate:.2f} ¢/kWh ({len(rates)} counties)")


if __name__ == "__main__":
    main()