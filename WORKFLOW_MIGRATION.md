# Enhanced URDB Workflow Integration

## Overview

This document shows how to achieve the same functionality as your existing URDB processing workflow using the new structured codebase. The enhanced system provides all the capabilities of your original code while adding modularity, error handling, and extensibility.

## Key Workflow Comparisons

### Your Original Workflow vs Enhanced Codebase

| Original Feature | Enhanced Module | Key Improvements |
|-----------------|-----------------|------------------|
| Geographic utility lookup | `geo_mapping.py` | Robust retry logic, caching, name normalization |
| Schema-agnostic rate parsing | `enhanced_pricing.py` | Structured fallback methods, quality scoring |
| County imputation | `EnhancedUtilityCountyCrosswalk` | Distance-based imputation with state preference |
| Date fallback logic | `TariffSelector` + workflow | Per-utility fallback, configurable limits |
| Plausibility filtering | `EnhancedRatePricer` | Configurable bounds, suspicious data flagging |
| Utility name mapping | `GeographicUtilityMapper` | Extensible mapping dictionary |

## Step-by-Step Migration Guide

### 1. Install and Setup

```bash
# Install the enhanced package
cd c:\Users\ltaylor2\Documents\GitHub\urdb-utils
pip install -e .

# Set your API key
$env:OPENEI_API_KEY="your_api_key_here"
```

### 2. Geographic Utility Mapping (Replaces `find_electric_utility_by_geo_location.py`)

```python
from urdb_county_rates.geo_mapping import GeographicUtilityMapper

# Initialize mapper with known utilities for better name matching
mapper = GeographicUtilityMapper(api_key="your_key", known_urdb_utilities=known_utils)

# Create county-utility mapping (same as your existing script)
county_mapping = mapper.create_county_utility_mapping(
    county_csv_path="./data/us_county_geo_information.csv",
    output_csv_path="county_utilities_map.csv",
    save_every=25,
    resume=True  # Resumes from existing file
)
```

### 3. Enhanced Rate Processing (Replaces your main processing script)

```python
from urdb_county_rates.enhanced_pricing import EnhancedRatePricer
from urdb_county_rates.urdb_client import URDBClient

# Initialize with your plausibility bounds
pricer = EnhancedRatePricer(
    energy_min=0.0,
    energy_max=2.0,
    tou_flattening_method='weighted_average'
)

# Fetch tariffs
client = URDBClient(api_key="your_key")
tariffs = client.get_rates(sector="commercial", approved=True, limit=5000)

# Calculate rates with enhanced parsing
effective_rates = []
for tariff in tariffs:
    rate = pricer.calculate_enhanced_effective_rate(
        tariff,
        usage_profile,
        "commercial"
    )
    effective_rates.append(rate)
```

### 4. County-Level Analysis with Imputation

```python
from urdb_county_rates.geo_mapping import EnhancedUtilityCountyCrosswalk
from urdb_county_rates.aggregate import RateAggregator

# Create enhanced crosswalk with imputation capabilities
crosswalk = EnhancedUtilityCountyCrosswalk()

# Load county-utility mappings
crosswalk.load_mappings("county_utilities_map.csv")

# Aggregate to counties
aggregator = RateAggregator(crosswalk, weighting_method="simple")
county_rates = aggregator.aggregate_to_counties(effective_rates)

# Convert to DataFrame and add all counties (for imputation)
county_df = aggregator.county_rates_to_dataframe(county_rates)
all_counties = pd.read_csv("us_county_geo_information.csv")

# Apply geographic imputation for missing counties
complete_df = crosswalk.impute_missing_counties(
    all_counties.merge(county_df, how='left'),
    prefer_same_state=True
)
```

## Complete Workflow Example

Here's how to run your complete workflow using the enhanced system:

```python
from examples.enhanced_workflow import EnhancedURDBWorkflow

# Initialize workflow with your configuration
workflow = EnhancedURDBWorkflow(
    api_key="your_key",
    output_dir="./enhanced_output",
    energy_min=0.0,          # Your ENERGY_MIN
    energy_max=2.0,          # Your ENERGY_MAX
    target_kw=None,          # Your TARGET_KW
    cutoff_start="2023-01-01", # Your CUTOFF_START
    min_cutoff_year=2016,    # Your MIN_CUTOFF_YEAR
    tou_only=False,          # Your TOU_ONLY
    keep_undated=True        # Your KEEP_UNDATED
)

# Run complete workflow (replicates your entire process)
output_files = workflow.run_complete_workflow(
    county_csv_path="./data/us_county_geo_information.csv",
    customer_class="commercial",
    monthly_usage=5000.0,
    peak_kw=30.0,
    ng_rates_csv="./data/eia_ng_prices_2023.csv"
)
```

## Configuration Mapping

Your original configuration variables map to the enhanced system as follows:

```python
# Original variables -> Enhanced parameters
CSV_PATH = "./data/usurdb.csv"                    # -> URDBClient fetches directly
COUNTY_UTILITY_CSV_PATH = "county_utilities_map.csv" # -> GeographicUtilityMapper output
NG_EIA_RATE_CSV_PATH = "./data/eia_ng_prices_2023.csv" # -> add_natural_gas_comparison()

KEEP_UNDATED = True              # -> EnhancedURDBWorkflow.keep_undated
TARGET_KW = None                 # -> EnhancedURDBWorkflow.target_kw
CUTOFF_START = "2023-01-01"      # -> EnhancedURDBWorkflow.cutoff_start
MIN_CUTOFF_YEAR = 2016           # -> EnhancedURDBWorkflow.min_cutoff_year
TOU_ONLY = False                 # -> EnhancedURDBWorkflow.tou_only
ENERGY_MIN = 0.0                 # -> EnhancedRatePricer.energy_min
ENERGY_MAX = 2.0                 # -> EnhancedRatePricer.energy_max

# Utility rename mapping
UTILITY_RENAME_MAP = {...}       # -> GeographicUtilityMapper.UTILITY_RENAME_MAP
```

## Advanced Features Available

The enhanced system provides additional capabilities beyond your original workflow:

### 1. Data Quality Scoring
```python
# Automatic quality flags for each rate calculation
rate.data_quality_flags  # ['used_filtered_average', 'had_suspicious_cells', etc.]

# Quality scores for aggregated county data
county_rate.data_quality_score  # 0.0 to 1.0
```

### 2. Multiple Export Formats
```python
# Export to multiple formats simultaneously
exporter.export_county_rates(
    county_rates,
    formats=['csv', 'parquet', 'json', 'geojson']
)
```

### 3. Comprehensive Validation
```python
# Validate aggregated data
validation_issues = aggregator.validate_aggregation(county_rates)
# Returns: {'outliers': [...], 'single_utility_counties': [...], ...}
```

### 4. Caching and Performance
```python
# Automatic API response caching
client = URDBClient(api_key="key", cache_ttl_hours=24)

# Geographic lookup caching
mapper.point_cache  # Avoids repeated API calls for same coordinates
```

## Command Line Interface

You can also run the enhanced workflow via CLI:

```bash
# Fetch tariffs (replaces direct CSV loading)
urdb-rates fetch --class commercial --output ./data

# Build county analysis (replaces your processing script)
urdb-rates build \
  --crosswalk ./county_utilities_map.csv \
  --tariffs ./data/raw_tariffs_commercial.json \
  --class commercial \
  --usage 5000 \
  --peak-kw 30 \
  --weights simple \
  --format csv json \
  --output ./enhanced_output

# Validate results
urdb-rates validate --crosswalk ./county_utilities_map.csv --data ./enhanced_output/*.json
```

## Output Comparison

### Your Original Output
```
urdb_commercial_summary_by_county.csv
```

### Enhanced System Output
```
enhanced_county_analysis_commercial.csv          # Main county analysis
enhanced_effective_rates_commercial.csv          # Individual tariff rates
enhanced_summary_commercial.json                 # Summary statistics
county_utilities_map.csv                        # Geographic mapping
validation_report.json                          # Data quality report
```

## Benefits of Migration

1. **Modularity**: Each component can be used independently
2. **Error Handling**: Robust retry logic and graceful degradation
3. **Extensibility**: Easy to add new weighting methods, export formats, etc.
4. **Validation**: Built-in data quality checks and reporting
5. **Performance**: Intelligent caching reduces API calls
6. **Documentation**: Comprehensive docstrings and type hints
7. **Testing**: Unit tests ensure reliability
8. **CLI Support**: Both programmatic and command-line interfaces

## Migration Timeline

1. **Phase 1**: Install enhanced system alongside existing code
2. **Phase 2**: Test enhanced workflow with small dataset
3. **Phase 3**: Compare outputs between old and new systems
4. **Phase 4**: Switch to enhanced system for production use
5. **Phase 5**: Extend with new features (additional customer classes, etc.)

The enhanced system is designed to be a drop-in replacement that provides the same functionality with significant improvements in reliability, maintainability, and extensibility.