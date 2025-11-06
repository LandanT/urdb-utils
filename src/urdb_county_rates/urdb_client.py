"""URDB API client with retry, backoff, and caching capabilities."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import backoff
import requests
from diskcache import Cache
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


class URDBError(Exception):
    """Base exception for URDB client errors."""
    pass


class URDBAPIError(URDBError):
    """Exception for URDB API-specific errors."""

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data


class URDBRateStructure(BaseModel):
    """Pydantic model for URDB rate structure data."""
    label: str
    utility: str
    eiaid: Optional[int] = None
    sector: str
    rate_structure: List[Dict[str, Any]]
    approved: Optional[bool] = None
    effective: Optional[str] = None
    end_date: Optional[str] = None
    source: Optional[str] = None
    uri: Optional[str] = None


class URDBUtility(BaseModel):
    """Pydantic model for URDB utility data."""
    label: str
    eiaid: Optional[int] = None
    utility_info: Optional[Dict[str, Any]] = None
    ownership: Optional[str] = None


class URDBClient:
    """
    Client for interacting with the DOE OpenEI Utility Rate Database (URDB) API.

    Features:
    - Automatic retry with exponential backoff
    - Disk-based caching to avoid redundant API calls
    - JSON-safe error handling
    - Rate limiting respect
    """

    BASE_URL = "https://api.openei.org/utility_rates"

    def __init__(
        self,
        api_key: str,
        cache_dir: Optional[Union[str, Path]] = None,
        cache_ttl_hours: int = 24,
        max_retries: int = 5,
        timeout: int = 30
    ):
        """
        Initialize the URDB client.

        Args:
            api_key: Your OpenEI API key
            cache_dir: Directory for disk cache (defaults to ./urdb_cache)
            cache_ttl_hours: Cache time-to-live in hours
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

        # Set up caching
        if cache_dir is None:
            cache_dir = Path.cwd() / "urdb_cache"
        self.cache = Cache(str(cache_dir))
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        # Set up session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'urdb-county-rates/0.1.0 (Python)',
            'Accept': 'application/json'
        })

        logger.info(f"Initialized URDB client with cache at {cache_dir}")

    def _cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Generate a cache key for the request."""
        # Sort params for consistent keys
        sorted_params = sorted(params.items())
        return f"{endpoint}:{json.dumps(sorted_params, sort_keys=True)}"

    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if a cache entry is still valid."""
        if 'timestamp' not in cache_entry:
            return False

        cache_time = datetime.fromisoformat(cache_entry['timestamp'])
        return datetime.now() - cache_time < self.cache_ttl

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, requests.exceptions.Timeout),
        max_tries=5,
        factor=2,
        max_value=60
    )
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the URDB API with retry logic.

        Args:
            endpoint: API endpoint (relative to base URL)
            params: Query parameters

        Returns:
            JSON response data

        Raises:
            URDBAPIError: If the API returns an error
        """
        url = f"{self.BASE_URL}/{endpoint}"
        params = {**params, 'api_key': self.api_key}

        logger.debug(f"Making request to {url} with params: {params}")

        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()

            # Check for API-specific errors
            if isinstance(data, dict) and data.get('error'):
                raise URDBAPIError(
                    f"URDB API error: {data['error']}",
                    status_code=response.status_code,
                    response_data=data
                )

            return data

        except requests.exceptions.Timeout as e:
            logger.warning(f"Request timeout for {url}")
            raise URDBAPIError(f"Request timeout: {e}")

        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error for {url}: {e}")
            try:
                error_data = e.response.json() if e.response else {}
            except:
                error_data = {}

            raise URDBAPIError(
                f"HTTP {e.response.status_code}: {e}",
                status_code=e.response.status_code,
                response_data=error_data
            )

        except (ValueError, json.JSONDecodeError) as e:
            logger.error(f"Invalid JSON response from {url}: {e}")
            raise URDBAPIError(f"Invalid JSON response: {e}")

    def _get_cached_or_fetch(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get data from cache or fetch from API."""
        cache_key = self._cache_key(endpoint, params)

        # Try to get from cache first
        cached_entry = self.cache.get(cache_key)
        if cached_entry and self._is_cache_valid(cached_entry):
            logger.debug(f"Cache hit for {endpoint}")
            return cached_entry['data']

        # Fetch from API
        logger.debug(f"Cache miss for {endpoint}, fetching from API")
        data = self._make_request(endpoint, params)

        # Store in cache
        cache_entry = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        self.cache.set(cache_key, cache_entry)

        return data

    def get_rates(
        self,
        eiaid: Optional[int] = None,
        utility: Optional[str] = None,
        sector: Optional[str] = None,
        approved: Optional[bool] = None,
        limit: int = 500
    ) -> List[URDBRateStructure]:
        """
        Fetch rate structures from URDB.

        Args:
            eiaid: EIA utility ID
            utility: Utility name
            sector: Customer sector (residential, commercial, industrial)
            approved: Whether to include only approved rates
            limit: Maximum number of results

        Returns:
            List of rate structures
        """
        params = {'limit': limit}

        if eiaid is not None:
            params['eiaid'] = eiaid
        if utility is not None:
            params['utility'] = utility
        if sector is not None:
            params['sector'] = sector
        if approved is not None:
            params['approved'] = str(approved).lower()

        try:
            data = self._get_cached_or_fetch('v3/rates.json', params)

            # Handle both single items and lists
            if isinstance(data, dict):
                items = data.get('items', [data])
            else:
                items = data if isinstance(data, list) else []

            rates = []
            for item in items:
                try:
                    rate = URDBRateStructure(**item)
                    rates.append(rate)
                except ValidationError as e:
                    logger.warning(f"Failed to parse rate structure: {e}")
                    continue

            logger.info(f"Retrieved {len(rates)} rate structures")
            return rates

        except URDBAPIError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching rates: {e}")
            raise URDBError(f"Failed to fetch rates: {e}")

    def get_utilities(
        self,
        eiaid: Optional[int] = None,
        limit: int = 500
    ) -> List[URDBUtility]:
        """
        Fetch utilities from URDB.

        Args:
            eiaid: EIA utility ID
            limit: Maximum number of results

        Returns:
            List of utilities
        """
        params = {'limit': limit}

        if eiaid is not None:
            params['eiaid'] = eiaid

        try:
            data = self._get_cached_or_fetch('v3/utilities.json', params)

            # Handle both single items and lists
            if isinstance(data, dict):
                items = data.get('items', [data])
            else:
                items = data if isinstance(data, list) else []

            utilities = []
            for item in items:
                try:
                    utility = URDBUtility(**item)
                    utilities.append(utility)
                except ValidationError as e:
                    logger.warning(f"Failed to parse utility: {e}")
                    continue

            logger.info(f"Retrieved {len(utilities)} utilities")
            return utilities

        except URDBAPIError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching utilities: {e}")
            raise URDBError(f"Failed to fetch utilities: {e}")

    def clear_cache(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()
        logger.info("Cache cleared")

    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'cache_directory': str(self.cache.directory)
        }