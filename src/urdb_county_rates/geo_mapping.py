"""
Enhanced utility-county mapping with geographic lookup and name normalization.
Integrates the advanced functionality from your existing workflow.
"""

import logging
import os
import re
import time
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel

from .crosswalk import UtilityCountyCrosswalk, UtilityCountyMapping

logger = logging.getLogger(__name__)


class GeographicUtilityMapper:
    """
    Enhanced utility-county mapper that uses OpenEI geographic API
    to find utilities serving specific locations.
    """

    BASE_URL = "https://api.openei.org/utility_rates"

    # Utility name normalization mapping
    UTILITY_RENAME_MAP = {
        "AEP Ohio": "AEP Ohio - Columbus Southern Power",
        "AEP Texas": "AEP Texas Central",
        "City of Anaheim": "Anaheim Public Utilities",
        "City of Banning": "Banning Electric Utility",
        "City of Riverside": "Riverside Public Utilities",
        "City of Vernon": "Vernon City of",
        "ConEdison": "Consolidated Edison Co-NY Inc",
        "ComEd": "Commonwealth Edison",
        "Duke Energy": "Duke Energy Carolinas",
        "Entergy": "Entergy Arkansas",
        "LADWP": "Los Angeles Department of Water & Power",
        "OG&E": "Oklahoma Gas & Electric",
        "PECO": "PECO Energy Company",
        "PG&E": "Pacific Gas & Electric",
        "PSEG": "Public Service Electric & Gas",
        "SDG&E": "San Diego Gas & Electric",
        "SMUD": "Sacramento Municipal Utility District",
        "Tampa Electric": "Tampa Electric Company",
        "We Energies": "Wisconsin Electric Power Company"
    }

    def __init__(self, api_key: str, known_urdb_utilities: Optional[Set[str]] = None):
        """
        Initialize the geographic mapper.

        Args:
            api_key: OpenEI API key
            known_urdb_utilities: Set of known utility names from URDB data
        """
        self.api_key = api_key
        self.known_urdb_utilities = known_urdb_utilities or set()
        self.point_cache = {}  # Cache for lat/lon -> utilities

    def safe_sleep(self, seconds: float) -> None:
        """Safe sleep with error handling."""
        try:
            time.sleep(max(0, float(seconds)))
        except Exception:
            time.sleep(1)

    def normalize_utility_name(self, utility_name: str) -> Optional[str]:
        """
        Normalize utility name to match URDB conventions.

        Args:
            utility_name: Raw utility name

        Returns:
            Normalized name or None if no match found
        """
        if pd.isna(utility_name) or not str(utility_name).strip():
            return None

        name = str(utility_name).strip()

        # Try direct mapping first
        if name in self.UTILITY_RENAME_MAP:
            mapped = self.UTILITY_RENAME_MAP[name]
            if mapped in self.known_urdb_utilities:
                return mapped

        # Try exact match
        if name in self.known_urdb_utilities:
            return name

        # Try simplified name (remove "Company", "Co.", etc.)
        simplified = name.replace(" Company", "").replace(" Co.", " Co").strip()
        if simplified in self.known_urdb_utilities:
            return simplified

        return None

    def fetch_utilities_for_location(self,
                                   lat: float,
                                   lon: float,
                                   retries: int = 6,
                                   base_wait: float = 5,
                                   limit: int = 200,
                                   session: Optional[requests.Session] = None) -> List[str]:
        """
        Fetch utilities serving a specific geographic location.

        Args:
            lat: Latitude
            lon: Longitude
            retries: Number of retry attempts
            base_wait: Base wait time for retries
            limit: Results limit per request
            session: Optional requests session

        Returns:
            Sorted list of utility names
        """
        # Check cache first
        cache_key = (round(lat, 6), round(lon, 6))
        if cache_key in self.point_cache:
            return self.point_cache[cache_key]

        sess = session or requests.Session()
        params = {
            'version': 'latest',
            'format': 'json',
            'api_key': self.api_key,
            'lat': lat,
            'lon': lon,
            'sector': 'Commercial',
            'detail': 'minimal',
            'limit': limit,
            'offset': 0
        }

        utilities = set()

        while True:
            for attempt in range(retries):
                try:
                    response = sess.get(self.BASE_URL, params=params, timeout=45)

                    if response.status_code == 429:
                        # Handle rate limiting
                        retry_after = response.headers.get("Retry-After")
                        wait_time = float(retry_after) if retry_after else base_wait * (attempt + 1)
                        logger.warning(f"Rate limited at lat={lat}, lon={lon}; waiting {wait_time:.1f}s")
                        self.safe_sleep(wait_time)
                        continue

                    response.raise_for_status()
                    data = response.json()
                    break

                except requests.exceptions.RequestException as e:
                    if attempt < retries - 1:
                        wait_time = base_wait * (attempt + 1)
                        logger.warning(f"Request failed, retrying in {wait_time:.1f}s: {e}")
                        self.safe_sleep(wait_time)
                        continue
                    raise

            items = data.get("items", [])
            if not items:
                break

            for item in items:
                utility = item.get("utility")
                if utility:
                    normalized = self.normalize_utility_name(utility)
                    if normalized:
                        utilities.add(normalized)

            if len(items) < limit:
                break

            params["offset"] += limit
            self.safe_sleep(0.2)  # Be polite

        result = sorted(utilities)
        self.point_cache[cache_key] = result
        return result

    def create_county_utility_mapping(self,
                                    county_csv_path: str,
                                    output_csv_path: str = "county_utilities_map.csv",
                                    save_every: int = 25,
                                    resume: bool = True) -> pd.DataFrame:
        """
        Create county-to-utilities mapping using geographic lookup.

        Args:
            county_csv_path: Path to county CSV with lat/lng data
            output_csv_path: Output path for mapping CSV
            save_every: Save progress every N counties
            resume: Whether to resume from existing output file

        Returns:
            DataFrame with county-utility mappings
        """
        # Load county data
        counties = pd.read_csv(county_csv_path, dtype={"county_fips": str})
        counties["county_fips"] = counties["county_fips"].str.zfill(5)

        # Handle resumption
        if resume and os.path.exists(output_csv_path):
            existing = pd.read_csv(output_csv_path, dtype={"county_fips": str})
            existing["county_fips"] = existing["county_fips"].str.zfill(5)
            completed_fips = set(existing["county_fips"])
            logger.info(f"Resuming: found {len(completed_fips)} completed counties")
        else:
            existing = pd.DataFrame()
            completed_fips = set()

        session = requests.Session()
        new_rows = []
        processed = 0

        try:
            for idx, row in counties.iterrows():
                fips = str(row["county_fips"]).zfill(5)
                if fips in completed_fips:
                    continue

                lat = float(row["lat"])
                lon = float(row["lng"])

                utilities = self.fetch_utilities_for_location(lat, lon, session=session)

                new_rows.append({
                    "county_fips": fips,
                    "county": row.get("county"),
                    "state_id": row.get("state_id", row.get("state")),
                    "state_name": row.get("state_name"),
                    "lat": lat,
                    "lng": lon,
                    "utilities": ";".join(utilities)
                })

                processed += 1

                if processed % save_every == 0:
                    # Save progress
                    chunk_df = pd.DataFrame(new_rows)
                    if os.path.exists(output_csv_path):
                        chunk_df.to_csv(output_csv_path, mode="a", header=False, index=False)
                    else:
                        chunk_df.to_csv(output_csv_path, index=False)

                    completed_fips.update(chunk_df["county_fips"])
                    new_rows = []
                    logger.info(f"Saved progress: {processed} new counties processed")
                    self.safe_sleep(0.5)

            # Save final batch
            if new_rows:
                chunk_df = pd.DataFrame(new_rows)
                if os.path.exists(output_csv_path):
                    chunk_df.to_csv(output_csv_path, mode="a", header=False, index=False)
                else:
                    chunk_df.to_csv(output_csv_path, index=False)
                logger.info(f"Final save: {len(chunk_df)} counties")

        except Exception as e:
            # Save progress on error
            if new_rows:
                chunk_df = pd.DataFrame(new_rows)
                if os.path.exists(output_csv_path):
                    chunk_df.to_csv(output_csv_path, mode="a", header=False, index=False)
                else:
                    chunk_df.to_csv(output_csv_path, index=False)
                logger.info(f"Saved partial progress due to error: {len(chunk_df)} counties")
            raise

        # Return complete mapping
        return pd.read_csv(output_csv_path, dtype={"county_fips": str})


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate haversine distance between two points in kilometers.

    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates

    Returns:
        Distance in kilometers
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0088 * c  # Earth radius in km


class EnhancedUtilityCountyCrosswalk(UtilityCountyCrosswalk):
    """
    Enhanced crosswalk with geographic imputation capabilities.
    """

    def __init__(self, mapping_data=None):
        super().__init__(mapping_data)
        self.utility_centroids = {}
        self.utility_states = {}

    def build_utility_centroids(self) -> None:
        """Build centroid coordinates for each utility based on served counties."""
        if not self.mappings:
            return

        df = self.to_dataframe()
        if df.empty or 'lat' not in df.columns or 'lng' not in df.columns:
            return

        # Calculate utility centroids
        centroids = df.groupby('utility_id').agg({
            'lat': 'mean',
            'lng': 'mean'
        }).to_dict('index')

        self.utility_centroids = {
            uid: (coords['lat'], coords['lng'])
            for uid, coords in centroids.items()
        }

        # Build utility states mapping
        states = df.groupby('utility_id')['state'].apply(set).to_dict()
        self.utility_states = states

        logger.info(f"Built centroids for {len(self.utility_centroids)} utilities")

    def find_nearest_utility(self,
                           lat: float,
                           lon: float,
                           state: Optional[str] = None,
                           prefer_same_state: bool = True) -> Optional[Tuple[int, float]]:
        """
        Find nearest utility to a given location.

        Args:
            lat: Target latitude
            lon: Target longitude
            state: Target state (for preference)
            prefer_same_state: Whether to prefer utilities in same state

        Returns:
            Tuple of (utility_id, distance_km) or None
        """
        if not self.utility_centroids:
            self.build_utility_centroids()

        if not self.utility_centroids:
            return None

        candidates = []

        for utility_id, (u_lat, u_lng) in self.utility_centroids.items():
            if pd.isna(u_lat) or pd.isna(u_lng):
                continue

            distance = haversine_distance(lat, lon, u_lat, u_lng)
            utility_states = self.utility_states.get(utility_id, set())
            same_state = state in utility_states if state else False

            candidates.append((utility_id, distance, same_state))

        if not candidates:
            return None

        # Sort by preference: same state first, then by distance
        if prefer_same_state and state:
            candidates.sort(key=lambda x: (not x[2], x[1]))  # same_state DESC, distance ASC
        else:
            candidates.sort(key=lambda x: x[1])  # distance ASC

        return candidates[0][0], candidates[0][1]

    def impute_missing_counties(self,
                              county_df: pd.DataFrame,
                              prefer_same_state: bool = True) -> pd.DataFrame:
        """
        Impute utility data for counties with missing mappings.

        Args:
            county_df: DataFrame with county data including lat/lng
            prefer_same_state: Whether to prefer same-state utilities

        Returns:
            DataFrame with imputation columns added
        """
        result = county_df.copy()

        # Add imputation tracking columns
        result['imputation_used'] = False
        result['imputed_from_utility'] = np.nan
        result['imputed_distance_km'] = np.nan

        # Find counties needing imputation (could be customized based on your needs)
        needs_imputation = result['avg_kwh_all'].isna() if 'avg_kwh_all' in result.columns else pd.Series(True, index=result.index)

        for idx, row in result[needs_imputation].iterrows():
            if pd.isna(row.get('lat')) or pd.isna(row.get('lng')):
                continue

            nearest = self.find_nearest_utility(
                row['lat'],
                row['lng'],
                row.get('state_id', row.get('state')),
                prefer_same_state
            )

            if nearest:
                utility_id, distance = nearest
                result.at[idx, 'imputation_used'] = True
                result.at[idx, 'imputed_from_utility'] = utility_id
                result.at[idx, 'imputed_distance_km'] = distance

        return result