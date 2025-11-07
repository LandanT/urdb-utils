# URDB County Rates

Tools for mapping and analyzing electricity rates across U.S. counties using the DOE OpenEI Utility Rate Database (URDB).

## Overview

This utility provides a complete pipeline for:
- **Fetching** electricity tariff data from the DOE OpenEI URDB API
- **Processing** and filtering tariffs by customer class and date
- **Calculating** effective rates (¢/kWh) with configurable usage profiles
- **Mapping** utility rates to counties using crosswalk data
- **Aggregating** to county and state levels with weighting options
- **Exporting** data in multiple formats (CSV, Parquet, GeoJSON, JSON)

## Features

### 🔌 URDB API Client
- Automatic retry with exponential backoff
- Disk-based caching to minimize API calls
- JSON-safe error handling
- Rate limiting respect

### 🗺️ Utility-County Mapping
- Handles multiple utilities per county
- Multiple counties per utility
- Configurable weighting (simple, sales-based, meter-based)
- FIPS code validation

### 💰 Rate Calculations
- Energy charges (flat, tiered, time-of-use)
- Demand charges with peak calculations
- Fixed charges
- Effective ¢/kWh computation
- Data quality scoring

### 📊 Aggregation & Analysis
- County-level rollups with weights
- State-level summaries
- Coverage statistics
- Outlier detection and validation

### 📁 Multi-Format Export
- CSV for spreadsheet analysis
- Parquet for analytics workflows
- GeoJSON for mapping applications
- JSON for programmatic use

## Installation

### Prerequisites
- Python 3.11+
- OpenEI API key ([free signup](https://openei.org/services/api/signup/))

### Install from Source
```bash
git clone https://github.com/LandanT/urdb-utils.git
cd urdb-utils
pip install -e .
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Set up your API key
```bash
export OPENEI_API_KEY="your_api_key_here"
```

### 2. Fetch tariff data
```bash
urdb-rates fetch --class residential --state CA --output ./data
```

### 3. Create/obtain a utility-county crosswalk CSV
Required columns:
- `utility_id` (int): EIA utility ID
- `utility_name` (str): Utility name
- `state` (str): State abbreviation
- `county_fips` (str): 5-digit FIPS code
- `county_name` (str): County name
- `weight` (float, optional): Aggregation weight

### 4. Build county and state datasets
```bash
urdb-rates build \
  --crosswalk ./data/crosswalk.csv \
  --tariffs ./data/raw_tariffs_residential.json \
  --class residential \
  --output ./output \
  --format csv parquet json
```

### 5. View results
```bash
ls ./output/
# county_rates_residential.csv
# state_rates_residential.csv
# effective_rates_residential.csv
# summary_residential.json
```

## Command Line Interface

### Available Commands

- `fetch` - Download tariff data from URDB API
- `build` - Process and aggregate to county/state level
- `export` - Export data to various formats
- `validate` - Check data quality and coverage
- `info` - Show tool information and help

### Examples

```bash
# Fetch commercial tariffs for Texas
urdb-rates fetch --class commercial --state TX --output ./tx_data

# Build with sales-weighted aggregation
urdb-rates build \
  --crosswalk ./crosswalk.csv \
  --tariffs ./tx_data/raw_tariffs_commercial.json \
  --weights sales \
  --usage 5000 \
  --format csv parquet geojson

# Validate data quality
urdb-rates validate --crosswalk ./crosswalk.csv --data ./output/*.json

# Get help
urdb-rates --help
urdb-rates fetch --help
```

## Programmatic Usage

```python
from urdb_county_rates import URDBClient, UtilityCountyCrosswalk, TariffSelector, RatePricer, RateAggregator, DataExporter

# Initialize components
client = URDBClient(api_key="your_key")
crosswalk = UtilityCountyCrosswalk("crosswalk.csv")
selector = TariffSelector()
pricer = RatePricer()
aggregator = RateAggregator(crosswalk, weighting_method="simple")
exporter = DataExporter("./output")

# Fetch and process data
tariffs = client.get_rates(sector="residential", approved=True)
criteria = selector.create_criteria(customer_class="residential")
filtered_tariffs = selector.filter_tariffs(tariffs, criteria)
effective_rates = pricer.calculate_bulk_rates(filtered_tariffs)

# Aggregate and export
county_rates = aggregator.aggregate_to_counties(effective_rates)
state_rates = aggregator.aggregate_to_states(county_rates)
files = exporter.create_data_package(county_rates, state_rates, effective_rates)
```

## Data Quality & Validation

The tool includes comprehensive validation:
- **FIPS code validation** - Ensures valid 5-digit county codes
- **Rate outlier detection** - Flags unusually high/low rates
- **Coverage analysis** - Identifies gaps and single-utility counties
- **Data quality scoring** - Rates based on completeness and consistency

## Configuration

### Usage Profiles
Customize customer usage patterns:
```python
from urdb_county_rates.pricing import UsageProfile

# Custom residential profile
profile = UsageProfile(
    monthly_kwh=800,
    peak_kw=4.5,
    load_factor=0.35
)
```

### Weighting Methods
- `simple` - Equal weight for all utilities
- `sales` - Weight by electricity sales volume
- `meters` - Weight by number of customer meters
- `custom` - User-provided weights

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [DOE OpenEI](https://openei.org/) for providing the URDB API
- [EIA](https://www.eia.gov/) for utility identification standards

## Support
# Currently working on setting up additional support
- 📖 [Documentation](https://github.com/your-username/urdb-utils/wiki)
- 🐛 [Issue Tracker](https://github.com/your-username/urdb-utils/issues)
- 💬 [Discussions](https://github.com/your-username/urdb-utils/discussions)
