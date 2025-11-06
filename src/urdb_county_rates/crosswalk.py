"""Utility-county crosswalk mapping functionality."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class UtilityCountyMapping(BaseModel):
    """Model for utility-county mapping data."""
    utility_id: int  # EIA utility ID
    utility_name: str
    state: str
    county_fips: str  # 5-digit FIPS code (state + county)
    county_name: str
    weight: float = 1.0  # Weight for aggregation (sales, meters, etc.)
    data_source: Optional[str] = None
    last_updated: Optional[str] = None
    
    # Enhanced geographic data (optional)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    population: Optional[int] = None
    climate_zone: Optional[str] = None
    state_id: Optional[str] = None


class CrosswalkError(Exception):
    """Exception for crosswalk-related errors."""
    pass


class UtilityCountyCrosswalk:
    """
    Manages mapping between utilities and county FIPS codes.

    Handles cases where:
    - One utility serves multiple counties
    - Multiple utilities serve one county
    - Weighting for aggregation purposes
    """

    def __init__(self, mapping_data: Optional[Union[str, Path, pd.DataFrame]] = None):
        """
        Initialize the crosswalk.

        Args:
            mapping_data: Path to CSV file, DataFrame, or None to start empty
        """
        self.mappings: List[UtilityCountyMapping] = []
        self._utility_to_counties: Dict[int, Set[str]] = {}
        self._county_to_utilities: Dict[str, Set[int]] = {}

        if mapping_data is not None:
            self.load_mappings(mapping_data)

    def load_mappings(self, data: Union[str, Path, pd.DataFrame]) -> None:
        """
        Load utility-county mappings from CSV file or DataFrame.

        Expected columns:
        - utility_id (int): EIA utility ID
        - utility_name (str): Utility name
        - state (str): State abbreviation
        - county_fips (str): 5-digit FIPS code
        - county_name (str): County name
        - weight (float, optional): Weight for aggregation
        - data_source (str, optional): Source of mapping data
        - last_updated (str, optional): When mapping was last updated

        Args:
            data: Path to CSV file or pandas DataFrame
        """
        try:
            if isinstance(data, (str, Path)):
                df = pd.read_csv(data)
                logger.info(f"Loaded crosswalk data from {data}")
            else:
                df = data.copy()
                logger.info("Loaded crosswalk data from DataFrame")

            # Validate required columns
            required_cols = ['utility_id', 'utility_name', 'state', 'county_fips', 'county_name']
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise CrosswalkError(f"Missing required columns: {missing_cols}")

            # Set default weight if not provided
            if 'weight' not in df.columns:
                df['weight'] = 1.0

            # Parse mappings
            self.mappings = []
            for _, row in df.iterrows():
                try:
                    mapping = UtilityCountyMapping(
                        utility_id=int(row['utility_id']),
                        utility_name=str(row['utility_name']),
                        state=str(row['state']),
                        county_fips=str(row['county_fips']).zfill(5),  # Ensure 5 digits
                        county_name=str(row['county_name']),
                        weight=float(row.get('weight', 1.0)),
                        data_source=row.get('data_source'),
                        last_updated=row.get('last_updated')
                    )
                    self.mappings.append(mapping)
                except (ValidationError, ValueError) as e:
                    logger.warning(f"Skipping invalid mapping row: {e}")
                    continue

            self._build_indices()
            logger.info(f"Loaded {len(self.mappings)} utility-county mappings")

        except Exception as e:
            logger.error(f"Failed to load crosswalk data: {e}")
            raise CrosswalkError(f"Failed to load crosswalk data: {e}")

    def _build_indices(self) -> None:
        """Build internal indices for fast lookups."""
        self._utility_to_counties = {}
        self._county_to_utilities = {}

        for mapping in self.mappings:
            # Utility -> Counties
            if mapping.utility_id not in self._utility_to_counties:
                self._utility_to_counties[mapping.utility_id] = set()
            self._utility_to_counties[mapping.utility_id].add(mapping.county_fips)

            # County -> Utilities
            if mapping.county_fips not in self._county_to_utilities:
                self._county_to_utilities[mapping.county_fips] = set()
            self._county_to_utilities[mapping.county_fips].add(mapping.utility_id)

    def add_mapping(
        self,
        utility_id: int,
        utility_name: str,
        state: str,
        county_fips: str,
        county_name: str,
        weight: float = 1.0,
        data_source: Optional[str] = None
    ) -> None:
        """
        Add a single utility-county mapping.

        Args:
            utility_id: EIA utility ID
            utility_name: Utility name
            state: State abbreviation
            county_fips: 5-digit FIPS code
            county_name: County name
            weight: Weight for aggregation
            data_source: Source of the mapping data
        """
        try:
            mapping = UtilityCountyMapping(
                utility_id=utility_id,
                utility_name=utility_name,
                state=state,
                county_fips=str(county_fips).zfill(5),
                county_name=county_name,
                weight=weight,
                data_source=data_source
            )

            self.mappings.append(mapping)
            self._build_indices()
            logger.debug(f"Added mapping: {utility_id} -> {county_fips}")

        except ValidationError as e:
            logger.error(f"Invalid mapping data: {e}")
            raise CrosswalkError(f"Invalid mapping data: {e}")

    def get_counties_for_utility(self, utility_id: int) -> List[UtilityCountyMapping]:
        """
        Get all counties served by a utility.

        Args:
            utility_id: EIA utility ID

        Returns:
            List of utility-county mappings
        """
        return [m for m in self.mappings if m.utility_id == utility_id]

    def get_utilities_for_county(self, county_fips: str) -> List[UtilityCountyMapping]:
        """
        Get all utilities serving a county.

        Args:
            county_fips: 5-digit FIPS code

        Returns:
            List of utility-county mappings
        """
        county_fips = str(county_fips).zfill(5)
        return [m for m in self.mappings if m.county_fips == county_fips]

    def get_all_utilities(self) -> Set[int]:
        """Get set of all utility IDs in the crosswalk."""
        return set(self._utility_to_counties.keys())

    def get_all_counties(self) -> Set[str]:
        """Get set of all county FIPS codes in the crosswalk."""
        return set(self._county_to_utilities.keys())

    def get_states(self) -> Set[str]:
        """Get set of all states in the crosswalk."""
        return {m.state for m in self.mappings}

    def filter_by_state(self, state: str) -> List[UtilityCountyMapping]:
        """
        Get all mappings for a specific state.

        Args:
            state: State abbreviation (e.g., 'CA', 'TX')

        Returns:
            List of mappings for the state
        """
        return [m for m in self.mappings if m.state.upper() == state.upper()]

    def get_coverage_stats(self) -> Dict[str, int]:
        """
        Get statistics about crosswalk coverage.

        Returns:
            Dictionary with coverage statistics
        """
        utilities_multi_county = sum(
            1 for uid in self._utility_to_counties
            if len(self._utility_to_counties[uid]) > 1
        )

        counties_multi_utility = sum(
            1 for fips in self._county_to_utilities
            if len(self._county_to_utilities[fips]) > 1
        )

        return {
            'total_mappings': len(self.mappings),
            'unique_utilities': len(self._utility_to_counties),
            'unique_counties': len(self._county_to_utilities),
            'unique_states': len(self.get_states()),
            'utilities_serving_multiple_counties': utilities_multi_county,
            'counties_served_by_multiple_utilities': counties_multi_utility
        }

    def validate_fips_codes(self) -> List[str]:
        """
        Validate FIPS codes and return any invalid ones.

        Returns:
            List of invalid FIPS codes
        """
        invalid_fips = []

        for mapping in self.mappings:
            fips = mapping.county_fips

            # Check length
            if len(fips) != 5:
                invalid_fips.append(f"{fips} (wrong length)")
                continue

            # Check if numeric
            if not fips.isdigit():
                invalid_fips.append(f"{fips} (not numeric)")
                continue

            # Check state code (first 2 digits should be 01-56)
            state_code = int(fips[:2])
            if state_code < 1 or state_code > 56:
                invalid_fips.append(f"{fips} (invalid state code)")

        return invalid_fips

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert mappings to pandas DataFrame.

        Returns:
            DataFrame with all mappings
        """
        if not self.mappings:
            return pd.DataFrame()

        data = [mapping.model_dump() for mapping in self.mappings]
        return pd.DataFrame(data)

    def save_to_csv(self, filepath: Union[str, Path]) -> None:
        """
        Save mappings to CSV file.

        Args:
            filepath: Path to output CSV file
        """
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(self.mappings)} mappings to {filepath}")

    @classmethod
    def create_sample_crosswalk(cls) -> 'UtilityCountyCrosswalk':
        """
        Create a sample crosswalk for testing/demonstration.

        Returns:
            UtilityCountyCrosswalk with sample data
        """
        crosswalk = cls()

        # Sample mappings for a few utilities and counties
        sample_data = [
            (1, "Pacific Gas & Electric Co", "CA", "06001", "Alameda County", 1.0),
            (1, "Pacific Gas & Electric Co", "CA", "06013", "Contra Costa County", 0.8),
            (1, "Pacific Gas & Electric Co", "CA", "06075", "San Francisco County", 1.0),
            (2, "Southern California Edison Co", "CA", "06037", "Los Angeles County", 0.9),
            (2, "Southern California Edison Co", "CA", "06059", "Orange County", 1.0),
            (3, "San Diego Gas & Electric Co", "CA", "06073", "San Diego County", 1.0),
            (4, "Commonwealth Edison Co", "IL", "17031", "Cook County", 1.0),
            (5, "Consolidated Edison Co-NY Inc", "NY", "36061", "New York County", 1.0),
        ]

        for utility_id, utility_name, state, county_fips, county_name, weight in sample_data:
            crosswalk.add_mapping(
                utility_id=utility_id,
                utility_name=utility_name,
                state=state,
                county_fips=county_fips,
                county_name=county_name,
                weight=weight,
                data_source="sample_data"
            )

        return crosswalk

    def load_county_geographic_data(self, geo_data_path: Union[str, Path]) -> None:
        """
        Load county geographic data and merge with existing mappings.
        
        Expected columns in geo data CSV:
        - county_fips: 5-digit FIPS code
        - county: County name (optional, for validation)
        - state_id: State abbreviation
        - lat/latitude: Latitude
        - lng/longitude: Longitude  
        - population: County population
        - climate_zone: Climate zone designation
        
        Args:
            geo_data_path: Path to county geographic data CSV
        """
        try:
            geo_df = pd.read_csv(geo_data_path)
            logger.info(f"Loading county geographic data from {geo_data_path}")
            
            # Normalize column names
            column_mapping = {
                'lat': 'latitude',
                'lng': 'longitude',
                'county': 'county_name_geo'  # Avoid conflicts
            }
            geo_df = geo_df.rename(columns=column_mapping)
            
            # Ensure county_fips is 5-digit string
            geo_df['county_fips'] = geo_df['county_fips'].astype(str).str.zfill(5)
            
            # Create lookup dictionary for geographic data
            geo_lookup = {}
            for _, row in geo_df.iterrows():
                county_fips = row['county_fips']
                geo_lookup[county_fips] = {
                    'latitude': row.get('latitude'),
                    'longitude': row.get('longitude'),
                    'population': row.get('population'),
                    'climate_zone': row.get('climate_zone'),
                    'state_id': row.get('state_id')
                }
            
            # Update existing mappings with geographic data
            updated_count = 0
            for i, mapping in enumerate(self.mappings):
                county_fips = mapping.county_fips
                if county_fips in geo_lookup:
                    geo_data = geo_lookup[county_fips]
                    
                    # Create updated mapping with geographic data
                    updated_mapping = UtilityCountyMapping(
                        **mapping.model_dump(),  # Keep existing data
                        **{k: v for k, v in geo_data.items() if v is not None}  # Add geo data
                    )
                    self.mappings[i] = updated_mapping
                    updated_count += 1
            
            logger.info(f"Enhanced {updated_count} mappings with geographic data")
            
        except Exception as e:
            logger.error(f"Failed to load county geographic data: {e}")
            raise CrosswalkError(f"Failed to load county geographic data: {e}")

    def get_county_coordinates(self, county_fips: str) -> Optional[Tuple[float, float]]:
        """
        Get latitude and longitude for a county.
        
        Args:
            county_fips: 5-digit FIPS code
            
        Returns:
            Tuple of (latitude, longitude) or None if not available
        """
        county_fips = str(county_fips).zfill(5)
        for mapping in self.mappings:
            if mapping.county_fips == county_fips and mapping.latitude and mapping.longitude:
                return (mapping.latitude, mapping.longitude)
        return None
    
    def get_counties_in_climate_zone(self, climate_zone: str) -> List[str]:
        """
        Get all county FIPS codes in a specific climate zone.
        
        Args:
            climate_zone: Climate zone identifier
            
        Returns:
            List of county FIPS codes in the climate zone
        """
        counties = []
        for mapping in self.mappings:
            if mapping.climate_zone == climate_zone and mapping.county_fips not in counties:
                counties.append(mapping.county_fips)
        return counties
    
    def get_geographic_summary(self) -> Dict[str, any]:
        """
        Get summary statistics of geographic data coverage.
        
        Returns:
            Dictionary with geographic data statistics
        """
        total_counties = len(self.get_all_counties())
        
        counties_with_coords = len([
            m for m in self.mappings 
            if m.latitude is not None and m.longitude is not None
        ])
        
        counties_with_population = len([
            m for m in self.mappings 
            if m.population is not None
        ])
        
        counties_with_climate = len([
            m for m in self.mappings 
            if m.climate_zone is not None
        ])
        
        climate_zones = set(
            m.climate_zone for m in self.mappings 
            if m.climate_zone is not None
        )
        
        return {
            'total_counties': total_counties,
            'counties_with_coordinates': counties_with_coords,
            'counties_with_population': counties_with_population, 
            'counties_with_climate_zone': counties_with_climate,
            'coordinate_coverage_pct': round(100 * counties_with_coords / total_counties, 1) if total_counties > 0 else 0,
            'population_coverage_pct': round(100 * counties_with_population / total_counties, 1) if total_counties > 0 else 0,
            'climate_coverage_pct': round(100 * counties_with_climate / total_counties, 1) if total_counties > 0 else 0,
            'unique_climate_zones': len(climate_zones),
            'climate_zones': sorted(list(climate_zones))
        }