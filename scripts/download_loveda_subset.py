#!/usr/bin/env python
"""Download a deterministic subset of LoveDA from the official-structure mirror.

This is an explicit convenience script, not an automatic download inside the
training pipeline. It fetches the exact files listed by the Hugging Face mirror
(``ahsennazir/loveDA``, which mirrors the Zenodo release) and writes them into
the standard ``LoveDA/{Train,Val}/{Rural,Urban}/{images_png,masks_png}`` layout.

Example::

    python scripts/download_loveda_subset.py --root data/LoveDA \
        --train-rural 300 --train-urban 300 --val-rural 100 --val-urban 100

The selected stems are recorded in ``data/LoveDA/subset_manifest.json`` so the
exact subset can be reproduced. Set all counts to ``-1`` to download everything.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

API = "https://huggingface.co/api/datasets/ahsennazir/loveDA/tree/main/{path}"
RESOLVE = "https://huggingface.co/datasets/ahsennazir/loveDA/resolve/main/{path}"
USER_AGENT = "pointseg-dataset-downloader/1.0"


def list_files(directory: str) -> list[str]:
    url = API.format(path=directory)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    files = sorted(item["path"] for item in payload if item["type"] == "file")
    if not files:
        raise RuntimeError(f"no files listed at {url}")
    return files


def download(remote_path: str, destination: Path, retries: int = 3) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    url = RESOLVE.format(path=remote_path)
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response:
                tmp.write_bytes(response.read())
            if tmp.stat().st_size == 0:
                raise RuntimeError("empty response")
            tmp.replace(destination)
            return
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def select(files: list[str], count: int) -> list[str]:
    return files if count < 0 else files[:count]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/LoveDA")
    parser.add_argument("--train-rural", type=int, default=300)
    parser.add_argument("--train-urban", type=int, default=300)
    parser.add_argument("--val-rural", type=int, default=100)
    parser.add_argument("--val-urban", type=int, default=100)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.root)
    plan = {
        ("Train", "Rural"): args.train_rural,
        ("Train", "Urban"): args.train_urban,
        ("Val", "Rural"): args.val_rural,
        ("Val", "Urban"): args.val_urban,
    }
    jobs: list[tuple[str, Path]] = []
    manifest: dict[str, object] = {
        "source": "https://huggingface.co/datasets/ahsennazir/loveDA",
        "upstream": "https://doi.org/10.5281/zenodo.5706578",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }
    for (split, domain), count in plan.items():
        stems = []
        for kind in ("images_png", "masks_png"):
            directory = f"{split}/{domain}/{kind}"
            files = list_files(directory)
            selected = select(files, count)
            for remote in selected:
                jobs.append((remote, root / remote))
            if kind == "images_png":
                stems = [Path(path).stem for path in selected]
        manifest["splits"][f"{split}/{domain}"] = {
            "count": len(stems),
            "stems": stems,
        }
        print(f"{split}/{domain}: {len(stems)} pairs")

    print(f"downloading {len(jobs)} files to {root} with {args.workers} workers")
    started = time.time()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, remote, dest): remote for remote, dest in jobs}
        for index, future in enumerate(as_completed(futures), start=1):
            remote = futures[future]
            try:
                future.result()
            except Exception as exc:
                failures.append((remote, str(exc)))
            if index % 50 == 0 or index == len(jobs):
                elapsed = time.time() - started
                print(f"  {index}/{len(jobs)} files ({elapsed:.0f}s)")

    if failures:
        print(f"{len(failures)} downloads failed", file=sys.stderr)
        for remote, error in failures[:10]:
            print(f"  {remote}: {error}", file=sys.stderr)
        return 1

    root.mkdir(parents=True, exist_ok=True)
    (root / "subset_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done in {time.time() - started:.0f}s; manifest: {root / 'subset_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
