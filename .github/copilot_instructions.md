# Copilot Instructions for URDB County Rates

## Project context
This repo builds county- and state-level electricity-rate datasets using the DOE OpenEI **Utility Rate Database (URDB)**. It includes:
- URDB API client with retry/backoff + caching
- Utility↔county (FIPS) mapping
- Effective ¢/kWh calculations (flat and TOU-flattened in roadmap)
- Aggregations by customer class and weighting method
- Exports to CSV/Parquet/GeoJSON

## Coding preferences
- Language: Python 3.11+
- Style: PEP8; prefer type hints; docstrings in Google style
- Libraries: `requests`/`httpx`, `pandas` (or `polars` if faster), `pydantic` for schemas, `typer` for CLI
- Logging over prints (`logging` with INFO/WARN)
- Write pure functions; avoid hidden I/O in logic layers

## Architecture
- `urdb_client.py` — API calls, retries, caching; do not embed business logic here
- `crosswalk.py` — utility↔county FIPS mapping; allow multiple utilities per county
- `tariffs.py` — tariff selection by class + as_of date; pin URDB schema version
- `pricing.py` — compute effective ¢/kWh; clean units; handle demand charges explicitly
- `aggregate.py` — county/state rollups with configurable weights (sales, meters, simple)
- `export.py` — CSV/Parquet/GeoJSON writers
- `cli.py` — commands: `fetch`, `build`, `export`, `map`

## Data & assumptions
- Always surface and pass through **customer class** and **as_of** date.
- Treat counties with multiple utilities via weights; never collapse to a single name.
- Prefer **utility IDs** over names in joins and outputs.
- Make TOU flattening assumptions explicit; keep defaults conservative and documented.

## Testing & QA
- Unit tests for tariff parsing, crosswalk merges, and pricing math.
- QA checks: non-negativity, percentile bounds, compare to EIA state averages (tolerance).
- Provide a small deterministic fixture for CI.

## Good tasks to ask Copilot
- “Create a `URDBClient` with retry/backoff and JSON-safe error handling.”
- “Write `county_rates()` that merges tariffs with crosswalk weights and returns a tidy DataFrame.”
- “Add a Typer CLI with `fetch/build/export` commands and `--class/--asof/--weights` flags.”
- “Generate a validation report listing outliers and counties with zero utilities.”

## Non-goals
- Don’t hard-code API keys or secrets.
- Don’t assume one utility per county.
- Don’t mix I/O with core math functions.

## Output quality
- Prefer deterministic functions, typed signatures, clear exceptions.
- Include docstrings and usage examples for new public APIs.
