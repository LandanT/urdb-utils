"""Tariff selection and filtering by customer class and effective date."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Set

import pandas as pd
from pydantic import BaseModel, ValidationError

from .urdb_client import URDBRateStructure

logger = logging.getLogger(__name__)


class TariffSelectionCriteria(BaseModel):
    """Criteria for selecting tariffs from URDB data."""
    customer_class: Optional[str] = None  # residential, commercial, industrial
    as_of_date: Optional[str] = None  # YYYY-MM-DD format
    approved_only: bool = True
    exclude_expired: bool = True
    utility_ids: Optional[Set[int]] = None
    state_codes: Optional[Set[str]] = None


class TariffError(Exception):
    """Exception for tariff-related errors."""
    pass


class TariffSelector:
    """
    Handles tariff selection and filtering based on various criteria.

    Key functionality:
    - Filter by customer class (residential, commercial, industrial)
    - Filter by effective date range
    - Handle tariff versioning and approval status
    - Pin to specific URDB schema versions
    """

    # Standard customer class mappings
    CUSTOMER_CLASS_ALIASES = {
        'residential': ['residential', 'res'],
        'commercial': ['commercial', 'comm', 'general'],
        'industrial': ['industrial', 'ind', 'manufacturing']
    }

    def __init__(self, schema_version: str = "v3"):
        """
        Initialize the tariff selector.

        Args:
            schema_version: URDB schema version to pin to
        """
        self.schema_version = schema_version
        logger.info(f"Initialized TariffSelector with schema version {schema_version}")

    def normalize_customer_class(self, sector: str) -> Optional[str]:
        """
        Normalize customer class/sector names to standard categories.

        Args:
            sector: Raw sector name from URDB

        Returns:
            Normalized customer class or None if no match
        """
        if not sector:
            return None

        sector_lower = sector.lower().strip()

        for standard_class, aliases in self.CUSTOMER_CLASS_ALIASES.items():
            if sector_lower in aliases or any(alias in sector_lower for alias in aliases):
                return standard_class

        logger.debug(f"Unrecognized customer class: {sector}")
        return sector_lower  # Return as-is if no match

    def parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """
        Parse date string to datetime object.

        Args:
            date_str: Date string in various formats

        Returns:
            Parsed datetime or None
        """
        if not date_str:
            return None

        # Common date formats in URDB
        formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ"
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue

        logger.warning(f"Could not parse date: {date_str}")
        return None

    def is_tariff_effective(
        self,
        tariff: URDBRateStructure,
        as_of_date: Optional[datetime] = None
    ) -> bool:
        """
        Check if a tariff is effective on a given date.

        Args:
            tariff: URDB rate structure
            as_of_date: Date to check against (defaults to today)

        Returns:
            True if tariff is effective on the date
        """
        if as_of_date is None:
            as_of_date = datetime.now()

        # Check effective date
        effective_date = self.parse_date(tariff.effective)
        if effective_date and effective_date > as_of_date:
            return False

        # Check end date
        end_date = self.parse_date(tariff.end_date)
        if end_date and end_date < as_of_date:
            return False

        return True

    def filter_tariffs(
        self,
        tariffs: List[URDBRateStructure],
        criteria: TariffSelectionCriteria
    ) -> List[URDBRateStructure]:
        """
        Filter tariffs based on selection criteria.

        Args:
            tariffs: List of URDB rate structures
            criteria: Selection criteria

        Returns:
            Filtered list of tariffs
        """
        filtered = []
        as_of_date = None

        if criteria.as_of_date:
            as_of_date = self.parse_date(criteria.as_of_date)
            if not as_of_date:
                raise TariffError(f"Invalid as_of_date format: {criteria.as_of_date}")

        for tariff in tariffs:
            # Filter by approval status
            if criteria.approved_only and not tariff.approved:
                continue

            # Filter by customer class
            if criteria.customer_class:
                normalized_sector = self.normalize_customer_class(tariff.sector)
                target_class = criteria.customer_class.lower()
                if normalized_sector != target_class:
                    continue

            # Filter by effective date
            if criteria.exclude_expired and not self.is_tariff_effective(tariff, as_of_date):
                continue

            # Filter by utility IDs
            if criteria.utility_ids and tariff.eiaid not in criteria.utility_ids:
                continue

            # Note: State filtering would require additional utility metadata
            # that's not directly in the rate structure

            filtered.append(tariff)

        logger.info(f"Filtered {len(tariffs)} tariffs to {len(filtered)} based on criteria")
        return filtered

    def select_best_tariff(
        self,
        tariffs: List[URDBRateStructure],
        preference_order: List[str] = None
    ) -> Optional[URDBRateStructure]:
        """
        Select the best tariff from a list based on preference order.

        Args:
            tariffs: List of candidate tariffs
            preference_order: Order of preference criteria

        Returns:
            Best tariff or None if no candidates
        """
        if not tariffs:
            return None

        if len(tariffs) == 1:
            return tariffs[0]

        # Default preference order
        if preference_order is None:
            preference_order = ['approved', 'recent_effective', 'has_end_date']

        candidates = tariffs.copy()

        for criterion in preference_order:
            if len(candidates) == 1:
                break

            if criterion == 'approved':
                # Prefer approved tariffs
                approved = [t for t in candidates if t.approved]
                if approved:
                    candidates = approved

            elif criterion == 'recent_effective':
                # Prefer more recently effective tariffs
                def effective_date_key(tariff):
                    date = self.parse_date(tariff.effective)
                    return date if date else datetime.min

                candidates.sort(key=effective_date_key, reverse=True)
                # Keep only the most recent
                if candidates:
                    most_recent_date = effective_date_key(candidates[0])
                    candidates = [
                        t for t in candidates
                        if effective_date_key(t) == most_recent_date
                    ]

            elif criterion == 'has_end_date':
                # Prefer tariffs without end dates (ongoing)
                no_end_date = [t for t in candidates if not t.end_date]
                if no_end_date:
                    candidates = no_end_date

        return candidates[0] if candidates else None

    def group_tariffs_by_utility(
        self,
        tariffs: List[URDBRateStructure]
    ) -> Dict[int, List[URDBRateStructure]]:
        """
        Group tariffs by utility ID.

        Args:
            tariffs: List of tariffs to group

        Returns:
            Dictionary mapping utility ID to list of tariffs
        """
        groups = {}

        for tariff in tariffs:
            if tariff.eiaid is None:
                logger.warning(f"Tariff {tariff.label} has no utility ID")
                continue

            if tariff.eiaid not in groups:
                groups[tariff.eiaid] = []
            groups[tariff.eiaid].append(tariff)

        logger.info(f"Grouped {len(tariffs)} tariffs into {len(groups)} utilities")
        return groups

    def get_tariff_summary(self, tariffs: List[URDBRateStructure]) -> pd.DataFrame:
        """
        Create a summary DataFrame of tariffs.

        Args:
            tariffs: List of tariffs to summarize

        Returns:
            DataFrame with tariff summary
        """
        if not tariffs:
            return pd.DataFrame()

        summary_data = []

        for tariff in tariffs:
            summary_data.append({
                'label': tariff.label,
                'utility': tariff.utility,
                'eiaid': tariff.eiaid,
                'sector': tariff.sector,
                'normalized_sector': self.normalize_customer_class(tariff.sector),
                'approved': tariff.approved,
                'effective': tariff.effective,
                'end_date': tariff.end_date,
                'is_current': self.is_tariff_effective(tariff),
                'rate_structure_count': len(tariff.rate_structure),
                'source': tariff.source,
                'uri': tariff.uri
            })

        return pd.DataFrame(summary_data)

    def validate_tariff_data(self, tariffs: List[URDBRateStructure]) -> Dict[str, List[str]]:
        """
        Validate tariff data and return any issues found.

        Args:
            tariffs: List of tariffs to validate

        Returns:
            Dictionary of validation issues by category
        """
        issues = {
            'missing_utility_id': [],
            'missing_effective_date': [],
            'invalid_date_format': [],
            'empty_rate_structure': [],
            'unknown_sector': []
        }

        for tariff in tariffs:
            tariff_id = f"{tariff.label} ({tariff.utility})"

            # Check for missing utility ID
            if tariff.eiaid is None:
                issues['missing_utility_id'].append(tariff_id)

            # Check for missing effective date
            if not tariff.effective:
                issues['missing_effective_date'].append(tariff_id)
            elif not self.parse_date(tariff.effective):
                issues['invalid_date_format'].append(f"{tariff_id}: {tariff.effective}")

            # Check for empty rate structure
            if not tariff.rate_structure:
                issues['empty_rate_structure'].append(tariff_id)

            # Check for unknown sector
            if tariff.sector:
                normalized = self.normalize_customer_class(tariff.sector)
                if normalized not in ['residential', 'commercial', 'industrial'] and normalized == tariff.sector.lower():
                    issues['unknown_sector'].append(f"{tariff_id}: {tariff.sector}")

        # Filter out empty issue lists
        return {k: v for k, v in issues.items() if v}

    @classmethod
    def create_criteria(
        self,
        customer_class: Optional[str] = None,
        as_of_date: Optional[str] = None,
        approved_only: bool = True,
        exclude_expired: bool = True,
        utility_ids: Optional[List[int]] = None,
        state_codes: Optional[List[str]] = None
    ) -> TariffSelectionCriteria:
        """
        Convenience method to create selection criteria.

        Args:
            customer_class: residential, commercial, or industrial
            as_of_date: Date in YYYY-MM-DD format
            approved_only: Include only approved tariffs
            exclude_expired: Exclude expired tariffs
            utility_ids: List of utility IDs to include
            state_codes: List of state codes to include

        Returns:
            TariffSelectionCriteria object
        """
        return TariffSelectionCriteria(
            customer_class=customer_class,
            as_of_date=as_of_date,
            approved_only=approved_only,
            exclude_expired=exclude_expired,
            utility_ids=set(utility_ids) if utility_ids else None,
            state_codes=set(state_codes) if state_codes else None
        )