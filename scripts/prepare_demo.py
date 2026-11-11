"""Prepare the data folders for the defense demo, without deleting anything.

  python scripts/prepare_demo.py reset     # before the demo
  python scripts/prepare_demo.py restore   # after the demo

reset   moves every CSV of raw_data, good_data and bad_data into
        data/pre_demo_backup/<timestamp>/<folder>/, so the three folders start empty and the
        ingestion job can only pick the demo files you copy into raw_data (data/demo/*.csv).
restore moves the raw_data backlog of the latest backup back into raw_data, so data keeps
        flowing into the dashboards after the demo. good_data / bad_data stay in the backup
        (they were already ingested and predicted).
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
FOLDERS = ("raw_data", "good_data", "bad_data")
BACKUP_ROOT = DATA / "pre_demo_backup"


def reset():
    backup = BACKUP_ROOT / datetime.now().strftime("%Y%m%dT%H%M%S")
    for folder in FOLDERS:
        files = sorted((DATA / folder).glob("*.csv"))
        if not files:
            print(f"{folder:<10} already empty")
            continue
        target = backup / folder
        target.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.move(str(f), str(target / f.name))
        print(f"{folder:<10} {len(files)} file(s) moved to {target.relative_to(DATA.parent)}")


def restore():
    backups = sorted(p for p in BACKUP_ROOT.glob("*") if (p / "raw_data").is_dir())
    if not backups:
        print("No raw_data backup found.")
        return
    source = backups[-1] / "raw_data"
    files = sorted(source.glob("*.csv"))
    for f in files:
        shutil.move(str(f), str(DATA / "raw_data" / f.name))
    print(f"{len(files)} file(s) moved back from {source.relative_to(DATA.parent)} to data/raw_data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["reset", "restore"])
    {"reset": reset, "restore": restore}[parser.parse_args().action]()
