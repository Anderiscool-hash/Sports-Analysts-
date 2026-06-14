"""Build replay-ready aligned model/market parquet files from raw price caches."""

from __future__ import annotations

import argparse

from sportedge.config import load_config
from sportedge.market.align import build_aligned_directory, build_aligned_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", help="One raw Polymarket parquet file to align.")
    parser.add_argument("--directory", default="data/cache", help="Directory of raw Polymarket parquet files.")
    parser.add_argument("--pattern", default="polymarket_*_10m.parquet", help="Glob pattern for directory mode.")
    parser.add_argument("--training", default="data/cache/training.parquet", help="Training cache parquet.")
    parser.add_argument("--output-dir", default="data/cache", help="Directory for aligned replay files.")
    parser.add_argument("--prefix", default="aligned_generated", help="Output filename prefix.")
    parser.add_argument("--config", default="config/config.yaml", help="Config file with model path.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.raw:
        summaries = [
            build_aligned_file(
                args.raw,
                args.training,
                args.output_dir,
                cfg.model.path,
                prefix=args.prefix,
            )
        ]
    else:
        summaries = build_aligned_directory(
            args.directory,
            args.training,
            args.output_dir,
            cfg.model.path,
            pattern=args.pattern,
            prefix=args.prefix,
        )

    built = 0
    skipped = 0
    rows = 0
    for item in summaries:
        if item.skipped:
            skipped += 1
            print(f"SKIP {item.source_path}: {item.reason}")
            continue
        built += 1
        rows += item.rows
        print(f"BUILT {item.output_path}: game={item.game_id} side={item.side} rows={item.rows}")
    print(f"SUMMARY built={built} skipped={skipped} rows={rows}")


if __name__ == "__main__":
    main()
