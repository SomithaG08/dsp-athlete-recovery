"""Split a dataset into N files of exactly 10 rows in raw_data, to simulate data arriving over time.

Example:
  python scripts/split_dataset.py --dataset data/corrupted/athlete_recovery_corrupted.csv \
      --raw-data data/raw_data --num-files 800 --p-missing-column 0.02
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROTECTED = {"Athlete_ID"}   # never dropped, so every file can still be identified
ROWS_PER_FILE = 10           # required by the project spec


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/corrupted/athlete_recovery_corrupted.csv", help="dataset path")
    p.add_argument("--raw-data", default="data/raw_data", help="raw_data folder path")
    p.add_argument("--num-files", type=int, required=True, help="number of files to generate")
    p.add_argument("--p-missing-column", type=float, default=0.0,
                   help="probability that a file loses one required column (schema error)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--keep-existing", action="store_true", help="do not delete existing batch_*.csv files")
    p.add_argument("--prefix", default="batch", help="file name prefix (use a new one with --keep-existing)")
    return p.parse_args(argv)


def split(df, num_files, p_missing_column=0.0, seed=None):
    """Yield (index, dataframe) chunks of exactly ROWS_PER_FILE rows.

    At most num_files chunks are produced (fewer if the dataset is too small); leftover rows
    are not written. Some chunks lose a column with the given probability.
    """
    if num_files < 1:
        raise ValueError("--num-files must be at least 1")
    if len(df) < ROWS_PER_FILE:
        raise ValueError(f"dataset needs at least {ROWS_PER_FILE} rows")
    rng = np.random.default_rng(seed)
    droppable = [c for c in df.columns if c not in PROTECTED]
    n_files = min(num_files, len(df) // ROWS_PER_FILE)
    for i in range(1, n_files + 1):
        chunk = df.iloc[(i - 1) * ROWS_PER_FILE:i * ROWS_PER_FILE]
        if p_missing_column and rng.random() < p_missing_column:
            chunk = chunk.drop(columns=[rng.choice(droppable)])
        yield i, chunk


def main(argv=None):
    args = parse_args(argv)
    df = pd.read_csv(args.dataset)
    out = Path(args.raw_data)
    out.mkdir(parents=True, exist_ok=True)
    if not args.keep_existing:
        for old in out.glob(f"{args.prefix}_*.csv"):
            old.unlink()
    width = len(str(args.num_files))
    dropped = written = 0
    for i, chunk in split(df, args.num_files, args.p_missing_column, args.seed):
        dropped += int(len(chunk.columns) < len(df.columns))
        chunk.to_csv(out / f"{args.prefix}_{i:0{width}d}.csv", index=False)
        written += 1
    if written < args.num_files:
        print(f"Warning: dataset only has enough rows for {written} files of {ROWS_PER_FILE} rows")
    print(f"Wrote {written} files of {ROWS_PER_FILE} rows from {args.dataset} to {out}"
          f" ({dropped} with a missing column)")


if __name__ == "__main__":
    main()
