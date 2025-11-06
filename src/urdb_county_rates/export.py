"""Data export functionality for CSV, Parquet, and GeoJSON formats."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from .aggregate import CountyRate, StateRate
from .pricing import EffectiveRate

logger = logging.getLogger(__name__)


class ExportError(Exception):
    """Exception for export-related errors."""
    pass


class DataExporter:
    """
    Handles exporting electricity rate data to various formats.

    Supported formats:
    - CSV: Tabular data export
    - Parquet: Columnar format for analytics
    - GeoJSON: Geospatial format with county/state boundaries
    - JSON: Raw structured data
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the data exporter.

        Args:
            output_dir: Directory for output files (defaults to current directory)
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized DataExporter with output directory: {self.output_dir}")

    def export_effective_rates(self,
                              effective_rates: List[EffectiveRate],
                              filename: str = "effective_rates",
                              formats: List[str] = None) -> Dict[str, Path]:
        """
        Export effective rates to specified formats.

        Args:
            effective_rates: List of calculated effective rates
            filename: Base filename (without extension)
            formats: List of formats to export ('csv', 'parquet', 'json')

        Returns:
            Dictionary mapping format to output file path
        """
        if formats is None:
            formats = ['csv']

        if not effective_rates:
            logger.warning("No effective rates to export")
            return {}

        # Convert to DataFrame
        data = [rate.model_dump() for rate in effective_rates]
        df = pd.DataFrame(data)

        # Flatten nested columns
        if 'assumptions' in df.columns:
            assumptions_df = pd.json_normalize(df['assumptions'])
            assumptions_df.columns = [f'assumption_{col}' for col in assumptions_df.columns]
            df = pd.concat([df.drop('assumptions', axis=1), assumptions_df], axis=1)

        if 'data_quality_flags' in df.columns:
            df['data_quality_flags'] = df['data_quality_flags'].apply(
                lambda x: '|'.join(x) if isinstance(x, list) else str(x)
            )

        return self._export_dataframe(df, filename, formats)

    def export_county_rates(self,
                           county_rates: List[CountyRate],
                           filename: str = "county_rates",
                           formats: List[str] = None,
                           include_geometry: bool = False) -> Dict[str, Path]:
        """
        Export county-level rates to specified formats.

        Args:
            county_rates: List of county-level aggregated rates
            filename: Base filename (without extension)
            formats: List of formats to export
            include_geometry: Whether to include county geometries (for GeoJSON)

        Returns:
            Dictionary mapping format to output file path
        """
        if formats is None:
            formats = ['csv']

        if not county_rates:
            logger.warning("No county rates to export")
            return {}

        # Convert to DataFrame
        data = [rate.model_dump() for rate in county_rates]
        df = pd.DataFrame(data)

        # Flatten list columns
        if 'utilities' in df.columns:
            df['utilities'] = df['utilities'].apply(
                lambda x: '|'.join(map(str, x)) if isinstance(x, list) else str(x)
            )

        if 'coverage_flags' in df.columns:
            df['coverage_flags'] = df['coverage_flags'].apply(
                lambda x: '|'.join(x) if isinstance(x, list) else str(x)
            )

        output_files = self._export_dataframe(df, filename, formats)

        # Add GeoJSON export if requested and geometry is available
        if include_geometry and 'geojson' in formats:
            try:
                geojson_path = self._export_county_geojson(county_rates, filename, None)
                output_files['geojson'] = geojson_path
            except Exception as e:
                logger.error(f"Failed to export GeoJSON: {e}")

        return output_files

    def export_state_rates(self,
                          state_rates: List[StateRate],
                          filename: str = "state_rates",
                          formats: List[str] = None) -> Dict[str, Path]:
        """
        Export state-level rates to specified formats.

        Args:
            state_rates: List of state-level aggregated rates
            filename: Base filename (without extension)
            formats: List of formats to export

        Returns:
            Dictionary mapping format to output file path
        """
        if formats is None:
            formats = ['csv']

        if not state_rates:
            logger.warning("No state rates to export")
            return {}

        # Convert to DataFrame
        data = [rate.model_dump() for rate in state_rates]
        df = pd.DataFrame(data)

        return self._export_dataframe(df, filename, formats)

    def _export_dataframe(self,
                         df: pd.DataFrame,
                         filename: str,
                         formats: List[str]) -> Dict[str, Path]:
        """
        Export DataFrame to multiple formats.

        Args:
            df: DataFrame to export
            filename: Base filename
            formats: List of formats

        Returns:
            Dictionary mapping format to file path
        """
        output_files = {}

        for fmt in formats:
            try:
                if fmt == 'csv':
                    filepath = self.output_dir / f"{filename}.csv"
                    df.to_csv(filepath, index=False)
                    output_files['csv'] = filepath
                    logger.info(f"Exported CSV to {filepath}")

                elif fmt == 'parquet':
                    filepath = self.output_dir / f"{filename}.parquet"
                    df.to_parquet(filepath, index=False)
                    output_files['parquet'] = filepath
                    logger.info(f"Exported Parquet to {filepath}")

                elif fmt == 'json':
                    filepath = self.output_dir / f"{filename}.json"
                    df.to_json(filepath, orient='records', indent=2)
                    output_files['json'] = filepath
                    logger.info(f"Exported JSON to {filepath}")

                else:
                    logger.warning(f"Unsupported format: {fmt}")

            except Exception as e:
                logger.error(f"Failed to export {fmt} format: {e}")
                continue

        return output_files

    def _export_county_geojson(self,
                              county_rates: List[CountyRate],
                              filename: str,
                              crosswalk=None) -> Path:
        """
        Export county rates as GeoJSON with county point geometries.
        
        Uses coordinate data from enhanced crosswalk if available,
        otherwise creates placeholder points.

        Args:
            county_rates: List of county rates
            filename: Base filename
            crosswalk: Optional UtilityCountyCrosswalk with geographic data

        Returns:
            Path to exported GeoJSON file
        """
        features = []

        for county_rate in county_rates:
            # Try to get real coordinates from crosswalk
            longitude, latitude = -100.0, 40.0  # Default placeholder
            population = None
            climate_zone = None
            
            if crosswalk:
                coords = crosswalk.get_county_coordinates(county_rate.county_fips)
                if coords:
                    latitude, longitude = coords
                
                # Get additional geographic data
                county_mappings = crosswalk.get_utilities_for_county(county_rate.county_fips)
                if county_mappings:
                    mapping = county_mappings[0]  # Take first mapping for additional data
                    population = mapping.population
                    climate_zone = mapping.climate_zone

            # Create point geometry
            geometry = Point(longitude, latitude)

            properties = {
                'county_fips': county_rate.county_fips,
                'county_name': county_rate.county_name,
                'state': county_rate.state,
                'customer_class': county_rate.customer_class,
                'weighted_rate_cents_per_kwh': county_rate.weighted_rate_cents_per_kwh,
                'utility_count': county_rate.utility_count,
                'utilities': '|'.join(map(str, county_rate.utilities)),
                'total_weight': county_rate.total_weight,
                'rate_range_min': county_rate.rate_range_min,
                'rate_range_max': county_rate.rate_range_max,
                'data_quality_score': county_rate.data_quality_score,
                'coverage_flags': '|'.join(county_rate.coverage_flags),
                'latitude': latitude,
                'longitude': longitude
            }
            
            # Add geographic data if available
            if population is not None:
                properties['population'] = population
            if climate_zone is not None:
                properties['climate_zone'] = climate_zone

            feature = {
                'type': 'Feature',
                'geometry': geometry.__geo_interface__,
                'properties': properties
            }

            features.append(feature)

        geojson = {
            'type': 'FeatureCollection',
            'features': features,
            'metadata': {
                'description': f'County electricity rates with geographic data',
                'source': 'DOE OpenEI URDB',
                'coordinate_system': 'WGS84',
                'total_counties': len(features)
            }
        }

        filepath = self.output_dir / f"{filename}.geojson"
        with open(filepath, 'w') as f:
            json.dump(geojson, f, indent=2)

        logger.info(f"Exported GeoJSON with {len(features)} county points to {filepath}")
        return filepath

    def export_summary_report(self,
                             county_rates: List[CountyRate],
                             state_rates: List[StateRate],
                             effective_rates: List[EffectiveRate],
                             filename: str = "summary_report") -> Path:
        """
        Export a comprehensive summary report.

        Args:
            county_rates: List of county rates
            state_rates: List of state rates
            effective_rates: List of effective rates
            filename: Report filename

        Returns:
            Path to exported report
        """
        report = {
            'metadata': {
                'export_timestamp': pd.Timestamp.now().isoformat(),
                'data_counts': {
                    'effective_rates': len(effective_rates),
                    'county_rates': len(county_rates),
                    'state_rates': len(state_rates)
                }
            },
            'summary_statistics': {},
            'data_quality': {}
        }

        # Add summary statistics
        if county_rates:
            county_df = pd.DataFrame([cr.model_dump() for cr in county_rates])
            report['summary_statistics']['county_level'] = {
                'rate_statistics': {
                    'mean': county_df['weighted_rate_cents_per_kwh'].mean(),
                    'median': county_df['weighted_rate_cents_per_kwh'].median(),
                    'std': county_df['weighted_rate_cents_per_kwh'].std(),
                    'min': county_df['weighted_rate_cents_per_kwh'].min(),
                    'max': county_df['weighted_rate_cents_per_kwh'].max()
                },
                'coverage': {
                    'total_counties': len(county_rates),
                    'states_represented': county_df['state'].nunique(),
                    'customer_classes': county_df['customer_class'].nunique(),
                    'single_utility_counties': sum(cr.utility_count == 1 for cr in county_rates)
                }
            }

        if state_rates:
            state_df = pd.DataFrame([sr.model_dump() for sr in state_rates])
            report['summary_statistics']['state_level'] = {
                'rate_statistics': {
                    'mean': state_df['weighted_rate_cents_per_kwh'].mean(),
                    'median': state_df['weighted_rate_cents_per_kwh'].median(),
                    'std': state_df['weighted_rate_cents_per_kwh'].std(),
                    'min': state_df['weighted_rate_cents_per_kwh'].min(),
                    'max': state_df['weighted_rate_cents_per_kwh'].max()
                },
                'coverage': {
                    'total_states': len(state_rates),
                    'customer_classes': state_df['customer_class'].nunique()
                }
            }

        # Add data quality metrics
        if effective_rates:
            quality_flags = [flag for rate in effective_rates for flag in rate.data_quality_flags]
            flag_counts = pd.Series(quality_flags).value_counts().to_dict()

            report['data_quality'] = {
                'total_quality_flags': len(quality_flags),
                'flag_distribution': flag_counts,
                'clean_rate_percentage': (len(effective_rates) - len([r for r in effective_rates if r.data_quality_flags])) / len(effective_rates) * 100
            }

        # Export report
        filepath = self.output_dir / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Exported summary report to {filepath}")
        return filepath

    def export_validation_report(self,
                                validation_issues: Dict[str, List[str]],
                                filename: str = "validation_report") -> Path:
        """
        Export validation issues report.

        Args:
            validation_issues: Dictionary of validation issues
            filename: Report filename

        Returns:
            Path to exported report
        """
        report = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'total_issues': sum(len(issues) for issues in validation_issues.values()),
            'issues_by_category': validation_issues,
            'summary': {
                category: len(issues)
                for category, issues in validation_issues.items()
            }
        }

        filepath = self.output_dir / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Exported validation report to {filepath}")
        return filepath

    def create_data_package(self,
                           county_rates: List[CountyRate],
                           state_rates: List[StateRate],
                           effective_rates: List[EffectiveRate],
                           package_name: str = "urdb_county_rates") -> Dict[str, Path]:
        """
        Create a complete data package with all outputs.

        Args:
            county_rates: List of county rates
            state_rates: List of state rates
            effective_rates: List of effective rates
            package_name: Base name for the package

        Returns:
            Dictionary of all exported files
        """
        all_files = {}

        # Export effective rates
        if effective_rates:
            files = self.export_effective_rates(
                effective_rates,
                f"{package_name}_effective_rates",
                ['csv', 'parquet', 'json']
            )
            all_files.update({f"effective_rates_{k}": v for k, v in files.items()})

        # Export county rates
        if county_rates:
            files = self.export_county_rates(
                county_rates,
                f"{package_name}_county_rates",
                ['csv', 'parquet', 'json']
            )
            all_files.update({f"county_rates_{k}": v for k, v in files.items()})

        # Export state rates
        if state_rates:
            files = self.export_state_rates(
                state_rates,
                f"{package_name}_state_rates",
                ['csv', 'parquet', 'json']
            )
            all_files.update({f"state_rates_{k}": v for k, v in files.items()})

        # Export summary report
        summary_path = self.export_summary_report(
            county_rates, state_rates, effective_rates, f"{package_name}_summary"
        )
        all_files['summary_report'] = summary_path

        logger.info(f"Created data package with {len(all_files)} files")
        return all_files

    def get_export_metadata(self, files: Dict[str, Path]) -> Dict[str, Any]:
        """
        Get metadata about exported files.

        Args:
            files: Dictionary of exported files

        Returns:
            Dictionary with file metadata
        """
        metadata = {}

        for name, filepath in files.items():
            if filepath.exists():
                stat = filepath.stat()
                metadata[name] = {
                    'path': str(filepath),
                    'size_bytes': stat.st_size,
                    'modified': pd.Timestamp.fromtimestamp(stat.st_mtime).isoformat(),
                    'format': filepath.suffix.lstrip('.')
                }
            else:
                metadata[name] = {'error': 'File not found'}

        return metadata