"""Simulate and ingest a new drive (the entry point of the Airflow DAG).

In a fleet, new recordings land in a drop zone and a marker file says the
upload is complete. Here, a "drive" is either new raw files dropped into
data/landing/nuscenes/ (same layout as data/raw/nuscenes, e.g. another part
of the nuScenes trainval release) and/or a list of scene names to add to the
ingested set, which lets nuScenes mini grow step by step.

    # start small: ingest only 4 of the 10 mini scenes
    python scripts/ingest_drive.py --set scene-0061 scene-0103 scene-0553 scene-0655

    # later, a "new drive" arrives
    python scripts/ingest_drive.py --stage scene-0757 scene-0796 scene-0916
    python scripts/ingest_drive.py --apply     # what the Airflow task runs
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PARAMS = Path("params.yaml")
LANDING = Path("data/landing")
RAW = Path("data/raw/nuscenes")
SCENES_LINE = re.compile(
    r"^(?P<indent>\s+)scenes:\s*(?P<value>[^#\n]*?)(?P<comment>\s*#.*)?$", re.M
)


def current_scenes() -> list[str] | str:
    m = SCENES_LINE.search(PARAMS.read_text())
    if not m:
        raise SystemExit("could not find `scenes:` in params.yaml")
    value = m["value"].strip()
    if value == "all":
        return "all"
    return [s.strip() for s in value.strip("[]").split(",") if s.strip()]


def write_scenes(scenes: list[str] | str) -> None:
    text = PARAMS.read_text()
    value = "all" if scenes == "all" else "[" + ", ".join(scenes) + "]"
    PARAMS.write_text(
        SCENES_LINE.sub(
            lambda m: f"{m['indent']}scenes: {value}{m['comment'] or ''}", text, count=1
        )
    )


def stage(scenes: list[str]) -> None:
    LANDING.mkdir(parents=True, exist_ok=True)
    (LANDING / "scenes.txt").write_text("\n".join(scenes) + "\n")
    (LANDING / "READY").touch()
    print(f"staged {len(scenes)} scene(s) in {LANDING}")


def apply() -> None:
    if not (LANDING / "READY").exists():
        print("no READY marker - nothing to ingest")
        return
    changed_raw = False
    drop = LANDING / "nuscenes"
    if drop.exists():
        shutil.copytree(drop, RAW, dirs_exist_ok=True)
        changed_raw = True
        print(f"merged {drop} into {RAW}")
    scene_file = LANDING / "scenes.txt"
    if scene_file.exists():
        new = [s for s in scene_file.read_text().split() if s]
        cur = current_scenes()
        if cur != "all":
            write_scenes(sorted(set(cur) | set(new)))
            print(f"scenes: {len(cur)} -> {len(set(cur) | set(new))}")
    if changed_raw:
        subprocess.run(["dvc", "add", str(RAW)], check=True)

    archive = LANDING / "processed" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    archive.mkdir(parents=True)
    for f in ("scenes.txt", "READY"):
        if (LANDING / f).exists():
            shutil.move(LANDING / f, archive / f)
    if drop.exists():
        shutil.rmtree(drop)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--stage", nargs="+", metavar="SCENE", help="announce a new drive")
    g.add_argument("--apply", action="store_true", help="ingest whatever is in the landing zone")
    g.add_argument("--set", nargs="+", metavar="SCENE", help="set the ingested scene list directly")
    args = ap.parse_args()
    if args.stage:
        stage(args.stage)
    elif args.set:
        write_scenes(args.set if args.set != ["all"] else "all")
        print(f"scenes set to {current_scenes()}")
    else:
        apply()


if __name__ == "__main__":
    main()
