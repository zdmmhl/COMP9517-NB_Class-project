"""Resumable iNaturalist archive downloader with MD5 verification."""

import argparse
import hashlib
import math
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


FILES = [
    {
        "name": "train_mini.tar.gz",
        "url": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz",
        "md5": "db6ed8330e634445efc8fec83ae81442",
    },
    {
        "name": "train_mini.json.tar.gz",
        "url": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz",
        "md5": "395a35be3651d86dc3b0d365b8ea5f92",
    },
    {
        "name": "val.tar.gz",
        "url": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz",
        "md5": "f6f6e0e242e3d4c9569ba56400938afc",
    },
    {
        "name": "val.json.tar.gz",
        "url": "https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz",
        "md5": "4d761e0f6a86cc63e8f7afc91f6a8f0b",
    },
]


def fmt_size(n):
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024


def md5_file(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def head_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as res:
        return int(res.headers["Content-Length"])


def request_range(url, start, end):
    req = urllib.request.Request(url)
    req.add_header("Range", f"bytes={start}-{end}")
    return urllib.request.urlopen(req, timeout=120)


def download_part(url, part_path, start, end, retries):
    expected = end - start + 1
    part_path.parent.mkdir(parents=True, exist_ok=True)
    existing = part_path.stat().st_size if part_path.exists() else 0
    if existing == expected:
        return expected
    if existing > expected:
        part_path.unlink()
        existing = 0

    for attempt in range(1, retries + 1):
        try:
            offset = existing
            with request_range(url, start + offset, end) as res, part_path.open("ab") as out:
                while True:
                    chunk = res.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    offset += len(chunk)
            size = part_path.stat().st_size
            if size == expected:
                return size
            existing = size
            raise IOError(f"incomplete part: expected {expected}, got {size}")
        except (urllib.error.URLError, TimeoutError, IOError) as exc:
            if attempt == retries:
                raise
            wait = min(60, 5 * attempt)
            print(f"Retrying {part_path.name} after error: {exc} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    return part_path.stat().st_size


def part_progress(parts_dir):
    total = 0
    if parts_dir.exists():
        for p in parts_dir.glob("part_*"):
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def combine_parts(parts, final_path):
    tmp_path = final_path.with_suffix(final_path.suffix + ".assembling")
    if tmp_path.exists():
        tmp_path.unlink()
    with tmp_path.open("wb") as out:
        for part in parts:
            with part.open("rb") as src:
                shutil.copyfileobj(src, out, length=8 * 1024 * 1024)
    os.replace(tmp_path, final_path)


def download_file(item, archive_dir, chunk_size, workers, retries):
    final_path = archive_dir / item["name"]
    expected_md5 = item["md5"]

    if final_path.exists():
        print(f"Checking existing {item['name']}...", flush=True)
        current_md5 = md5_file(final_path)
        if current_md5 == expected_md5:
            print(f"Already complete and verified: {item['name']}", flush=True)
            return
        print(f"Existing file failed MD5 ({current_md5}); replacing with verified download.", flush=True)
        final_path.unlink()

    total_size = head_size(item["url"])
    parts_dir = archive_dir / ".parts" / item["name"]
    part_count = math.ceil(total_size / chunk_size)
    print(
        f"Downloading {item['name']} ({fmt_size(total_size)}) in {part_count} parts with {workers} workers",
        flush=True,
    )

    stop = threading.Event()

    def reporter():
        started = time.time()
        while not stop.wait(15):
            done = part_progress(parts_dir)
            elapsed = max(1, time.time() - started)
            speed = done / elapsed
            pct = done / total_size * 100
            remaining = max(0, total_size - done)
            eta = remaining / speed if speed > 0 else 0
            print(
                f"{item['name']}: {pct:5.1f}% {fmt_size(done)}/{fmt_size(total_size)} "
                f"at {fmt_size(speed)}/s, ETA {eta/3600:.2f}h",
                flush=True,
            )

    thread = threading.Thread(target=reporter, daemon=True)
    thread.start()

    part_paths = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for idx in range(part_count):
                start = idx * chunk_size
                end = min(total_size - 1, start + chunk_size - 1)
                part_path = parts_dir / f"part_{idx:05d}"
                part_paths.append(part_path)
                futures.append(pool.submit(download_part, item["url"], part_path, start, end, retries))
            for future in as_completed(futures):
                future.result()
    finally:
        stop.set()
        thread.join(timeout=1)

    print(f"Combining parts for {item['name']}...", flush=True)
    combine_parts(part_paths, final_path)

    print(f"Verifying MD5 for {item['name']}...", flush=True)
    current_md5 = md5_file(final_path)
    if current_md5 != expected_md5:
        raise RuntimeError(f"MD5 mismatch for {item['name']}: expected {expected_md5}, got {current_md5}")

    shutil.rmtree(parts_dir, ignore_errors=True)
    print(f"Verified: {item['name']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="datasets/inat2021")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-mb", type=int, default=256)
    parser.add_argument("--retries", type=int, default=8)
    args = parser.parse_args()

    root = Path(args.root)
    archive_dir = root / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = args.chunk_mb * 1024 * 1024

    print(f"Dataset root: {root}", flush=True)
    print(f"Archive dir: {archive_dir}", flush=True)

    for item in FILES:
        download_file(item, archive_dir, chunk_size, args.workers, args.retries)

    print("All four required iNaturalist-2021 archives are downloaded and MD5 verified.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Re-run the same command to resume part downloads.", file=sys.stderr)
        raise
