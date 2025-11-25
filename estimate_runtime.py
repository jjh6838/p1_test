#!/usr/bin/env python3
"""Utility to estimate process_country_supply runtimes per country."""

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import process_country_supply as pcs
except ImportError as exc:  # pragma: no cover
    print(f"Unable to import process_country_supply: {exc}")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count centroids and facilities per country to gauge runtime complexity"
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        help="Explicit ISO3 codes to evaluate (overrides --countries-file)",
    )
    parser.add_argument(
        "--countries-file",
        default="countries_list.txt",
        help="Plain-text file with one ISO3 code per line",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most N countries after filtering",
    )
    parser.add_argument(
        "--facility-year",
        type=int,
        default=pcs.ANALYSIS_YEAR,
        help="Facility projection year to count (defaults to ANALYSIS_YEAR)",
    )
    parser.add_argument(
        "--sort-by",
        choices=["runtime", "centroids", "facilities"],
        default="runtime",
        help="Column used for sorting the summary table",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort results in ascending order",
    )
    parser.add_argument(
        "--output-csv",
        help="Optional path to write the summary as CSV",
    )
    return parser.parse_args()


def load_country_codes(explicit: Optional[Iterable[str]], file_path: str) -> List[str]:
    codes: List[str] = []
    if explicit:
        codes = [code.strip().upper() for code in explicit if code.strip()]
        return codes

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Country list file not found: {file_path}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            code = line.strip().upper()
            if code:
                codes.append(code)
    return codes


def estimate_runtime_minutes(centroids: int, facilities: int) -> float:
    if centroids <= 0 or facilities <= 0:
        return 0.25

    centroid_term = centroids / 20000.0
    facility_term = facilities / 150.0
    interaction_term = (centroids * math.log1p(facilities)) / 500000.0
    baseline = 1.0
    return round(baseline + centroid_term + facility_term + interaction_term, 2)


def collect_country_metrics(country_iso3: str, facility_year: int) -> dict:
    entry = {
        "iso3": country_iso3,
        "centroids": None,
        "facilities": None,
        "centroids_per_facility": None,
        "estimated_minutes": None,
        "elapsed_s": None,
        "error": "",
    }

    start = time.time()
    admin_boundaries = None
    centroids_gdf = None
    facilities_gdf = None
    try:
        admin_boundaries = pcs.load_admin_boundaries(country_iso3)
        country_bbox = pcs.get_country_bbox(admin_boundaries)
        centroids_gdf = pcs.load_population_centroids(country_bbox, admin_boundaries)
        facilities_gdf = pcs.load_energy_facilities(country_iso3, year=facility_year)

        centroid_count = len(centroids_gdf)
        facility_count = len(facilities_gdf)

        entry["centroids"] = centroid_count
        entry["facilities"] = facility_count
        entry["centroids_per_facility"] = (
            centroid_count / facility_count if facility_count else None
        )
        entry["estimated_minutes"] = estimate_runtime_minutes(
            centroid_count, facility_count
        )
    except Exception as exc:  # pragma: no cover - diagnostic path
        entry["error"] = str(exc)
    finally:
        entry["elapsed_s"] = time.time() - start
        # Release memory sooner for large countries
        del centroids_gdf
        del facilities_gdf
        del admin_boundaries
    return entry


def sort_results(results: List[dict], sort_field: str, ascending: bool) -> List[dict]:
    def sort_key(entry: dict):
        value = entry.get(sort_field)
        if value is None:
            return float("inf") if ascending else float("-inf")
        return value

    return sorted(results, key=sort_key, reverse=not ascending)


def print_summary(results: List[dict]) -> None:
    headers = [
        ("ISO3", 5),
        ("Centroids", 12),
        ("Facilities", 12),
        ("C/F", 10),
        ("Est. Min", 10),
        ("Elapsed (s)", 12),
        ("Error", 40),
    ]
    header_line = " ".join(f"{title:<{width}}" for title, width in headers)
    print(header_line)
    print("-" * len(header_line))
    for row in results:
        cf = row["centroids_per_facility"]
        cf_display = f"{cf:.1f}" if cf and math.isfinite(cf) else "-"
        est = row["estimated_minutes"]
        est_display = f"{est:.2f}" if est is not None else "-"
        elapsed = row["elapsed_s"]
        elapsed_display = f"{elapsed:.1f}" if elapsed is not None else "-"
        values = [
            row["iso3"],
            f"{row['centroids']:,}" if row["centroids"] is not None else "-",
            f"{row['facilities']:,}" if row["facilities"] is not None else "-",
            cf_display,
            est_display,
            elapsed_display,
            row["error"][:38] + ".." if len(row["error"]) > 40 else row["error"],
        ]
        formatted = " ".join(
            f"{value:<{width}}" for value, (_, width) in zip(values, headers)
        )
        print(formatted)

    successful = [r for r in results if not r["error"]]
    if successful:
        total_centroids = sum(r["centroids"] or 0 for r in successful)
        total_facilities = sum(r["facilities"] or 0 for r in successful)
        avg_minutes = sum(r["estimated_minutes"] or 0 for r in successful) / len(
            successful
        )
        print()
        print(
            f"Processed {len(successful)}/{len(results)} countries | "
            f"Centroids: {total_centroids:,} | Facilities: {total_facilities:,} | "
            f"Avg est. runtime: {avg_minutes:.2f} min"
        )


def write_csv(results: List[dict], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "iso3",
        "centroids",
        "facilities",
        "centroids_per_facility",
        "estimated_minutes",
        "elapsed_s",
        "error",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main() -> int:
    args = parse_args()
    try:
        country_codes = load_country_codes(args.countries, args.countries_file)
    except Exception as exc:
        print(f"Failed to load country codes: {exc}", file=sys.stderr)
        return 1

    if args.limit is not None:
        country_codes = country_codes[: args.limit]

    if not country_codes:
        print("No country codes supplied", file=sys.stderr)
        return 1

    results = []
    for idx, iso3 in enumerate(country_codes, start=1):
        print(f"[{idx}/{len(country_codes)}] Evaluating {iso3}...")
        metrics = collect_country_metrics(iso3, args.facility_year)
        results.append(metrics)

    sort_map = {
        "runtime": "estimated_minutes",
        "centroids": "centroids",
        "facilities": "facilities",
    }
    sorted_results = sort_results(results, sort_map[args.sort_by], args.ascending)
    print()
    print_summary(sorted_results)

    if args.output_csv:
        write_csv(sorted_results, args.output_csv)
        print(f"\nSummary written to {args.output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
