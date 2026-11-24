"""Build the expanded study's feature input from a local VISUELLE 2.0 copy."""
import argparse
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent / "outputs" / "COMP5064_Demo_and_Results"
sys.path.insert(0, str(PACKAGE / "src"))

from build_features import build


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="Directory containing the four source CSVs")
    args = parser.parse_args()
    features = build(args.data)
    destination = PACKAGE / "out" / "features.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(destination, index=False)
    print(f"Wrote {len(features):,} rows to {destination}")


if __name__ == "__main__":
    main()
