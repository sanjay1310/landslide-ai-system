from __future__ import annotations

import argparse
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert shapefiles to GeoJSON using ogr2ogr.")
    parser.add_argument("--district-shp", required=True, help="Path to official district shapefile")
    parser.add_argument("--state-shp", required=True, help="Path to official state shapefile")
    parser.add_argument(
        "--district-out",
        default="data/gis/india_districts_official.geojson",
        help="Output path for district GeoJSON",
    )
    parser.add_argument(
        "--state-out",
        default="data/gis/india_states_official.geojson",
        help="Output path for state GeoJSON",
    )
    args = parser.parse_args()

    if shutil.which("ogr2ogr") is None:
        print("ogr2ogr is not installed. Install GDAL first to convert shapefiles.")
        sys.exit(1)

    commands = [
        ["ogr2ogr", "-f", "GeoJSON", args.district_out, args.district_shp],
        ["ogr2ogr", "-f", "GeoJSON", args.state_out, args.state_shp],
    ]
    for command in commands:
        subprocess.run(command, check=True)
        print("Converted:", " ".join(command[3:]))


if __name__ == "__main__":
    main()
