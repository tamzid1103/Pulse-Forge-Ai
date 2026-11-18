"""
Download a manageable fan-only subset from the official MIMII dataset on Zenodo.

Uses HTTP Range requests to extract audio files directly from the 9.7 GB
remote zip archive without downloading the full archive.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import time
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pulseforge.download")

DEFAULT_ZENODO_URL = "https://zenodo.org/api/records/3384388/files/6_dB_fan.zip/content"


class RemoteSeekableFile(io.RawIOBase):
    """Seekable read-only stream over HTTP Range requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "PulseForgeAI/1.0"})
        with urllib.request.urlopen(req) as resp:
            self.size = int(resp.headers.get("Content-Length", 0))
        self.pos = 0

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size == -1 or self.pos + size > self.size:
            size = self.size - self.pos
        if size <= 0:
            return b""
        req = urllib.request.Request(
            self.url,
            headers={
                "User-Agent": "PulseForgeAI/1.0",
                "Range": f"bytes={self.pos}-{self.pos + size - 1}",
            },
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = resp.read()
                self.pos += len(data)
                return data
            except Exception as e:
                if attempt == 4:
                    raise
                logger.warning(f"Network error on range read (attempt {attempt + 1}/5): {e}. Retrying in 2s...")
                time.sleep(2)
        return b""


def download_fan_subset(
    output_dir: Path,
    machine_id: str = "id_00",
    max_normal: int = 60,
    max_abnormal: int = 20,
    zenodo_url: str = DEFAULT_ZENODO_URL,
) -> dict[str, Any]:
    """Download a manageable subset of normal and abnormal WAVs for a machine ID."""
    logger.info(f"Connecting to Zenodo remote zip ({zenodo_url})...")
    remote_file = RemoteSeekableFile(zenodo_url)
    zf = zipfile.ZipFile(remote_file)

    prefix_normal = f"fan/{machine_id}/normal/"
    prefix_abnormal = f"fan/{machine_id}/abnormal/"

    all_names = zf.namelist()
    normal_files = sorted([n for n in all_names if n.startswith(prefix_normal) and n.endswith(".wav")])
    abnormal_files = sorted([n for n in all_names if n.startswith(prefix_abnormal) and n.endswith(".wav")])

    logger.info(f"Available for {machine_id}: {len(normal_files)} normal, {len(abnormal_files)} abnormal files.")

    selected_normal = normal_files[:max_normal]
    selected_abnormal = abnormal_files[:max_abnormal]

    target_files = [(f, "normal") for f in selected_normal] + [(f, "abnormal") for f in selected_abnormal]

    logger.info(f"Target download: {len(selected_normal)} normal + {len(selected_abnormal)} abnormal = {len(target_files)} files.")

    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[dict[str, Any]] = []
    start_time = time.time()

    for idx, (archive_path, condition) in enumerate(target_files, 1):
        filename = Path(archive_path).name
        dest_subdir = output_dir / machine_id / condition
        dest_subdir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_subdir / filename

        if dest_path.exists() and dest_path.stat().st_size > 0:
            logger.debug(f"[{idx}/{len(target_files)}] Already exists: {dest_path}")
        else:
            data = zf.read(archive_path)
            dest_path.write_bytes(data)
            logger.info(f"[{idx}/{len(target_files)}] Downloaded: {archive_path} ({len(data)} bytes) -> {dest_path.name}")

        downloaded.append({
            "archive_path": archive_path,
            "local_path": dest_path.as_posix(),
            "machine_id": machine_id,
            "condition": condition,
            "size_bytes": dest_path.stat().st_size,
        })

    elapsed = round(time.time() - start_time, 2)
    logger.info(f"Download complete: {len(downloaded)} files in {elapsed}s.")

    summary = {
        "source_url": zenodo_url,
        "machine_id": machine_id,
        "noise_condition": "6_dB",
        "total_files": len(downloaded),
        "normal_files": len(selected_normal),
        "abnormal_files": len(selected_abnormal),
        "output_directory": output_dir.as_posix(),
        "elapsed_seconds": elapsed,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Download manageable MIMII fan subset")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Target output directory")
    parser.add_argument("--machine-id", type=str, default="id_00", help="Machine ID (e.g. id_00)")
    parser.add_argument("--max-normal", type=int, default=60, help="Max normal files to download")
    parser.add_argument("--max-abnormal", type=int, default=20, help="Max abnormal files to download")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / args.output_dir
    summary = download_fan_subset(
        output_dir=out_dir,
        machine_id=args.machine_id,
        max_normal=args.max_normal,
        max_abnormal=args.max_abnormal,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

