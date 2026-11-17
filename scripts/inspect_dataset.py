"""
MIMII Dataset Inspector for PulseForge AI.

Inspects local MIMII audio files or queries the remote Zenodo archive
to report machine types, machine IDs, conditions, channel counts,
sample rates, and durations.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("pulseforge.inspect")

ZENODO_RECORD_URL = "https://zenodo.org/api/records/3384388"
ZENODO_FAN_6DB_URL = "https://zenodo.org/api/records/3384388/files/6_dB_fan.zip/content"


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
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        self.pos += len(data)
        return data


def inspect_remote_zenodo() -> dict[str, Any]:
    """Query Zenodo MIMII record metadata and remote zip layout."""
    logger.info("Querying Zenodo API for record 3384388...")
    req = urllib.request.Request(ZENODO_RECORD_URL, headers={"User-Agent": "PulseForgeAI/1.0"})
    with urllib.request.urlopen(req) as resp:
        meta = json.loads(resp.read().decode("utf-8"))

    fan_archives = []
    for f in meta.get("files", []):
        key = f.get("key", "")
        if "fan" in key.lower():
            fan_archives.append({
                "filename": key,
                "size_bytes": f.get("size", 0),
                "size_mb": round(f.get("size", 0) / (1024 * 1024), 1),
                "download_url": f.get("links", {}).get("self", ""),
                "checksum": f.get("checksum", ""),
            })

    result = {
        "title": meta.get("metadata", {}).get("title", ""),
        "doi": meta.get("metadata", {}).get("doi", ""),
        "license": meta.get("metadata", {}).get("license", {}).get("id", ""),
        "publication_date": meta.get("metadata", {}).get("publication_date", ""),
        "fan_archives": fan_archives,
    }
    return result


def inspect_local_directory(directory: Path) -> dict[str, Any]:
    """Inspect local audio files in the given directory."""
    logger.info(f"Scanning local directory: {directory}")
    wav_files = sorted(directory.glob("**/*.wav"))
    logger.info(f"Found {len(wav_files)} WAV files.")

    counts: dict[str, int] = {}
    sample_info: dict[str, Any] = {}

    for path in wav_files:
        rel = path.relative_to(directory).as_posix()
        parts = rel.split("/")
        # e.g. id_00/normal/00000000.wav or fan/id_00/normal/00000000.wav
        condition = "unknown"
        for part in parts:
            if part in ("normal", "abnormal"):
                condition = part
                break
        counts[condition] = counts.get(condition, 0) + 1

        if not sample_info and path.is_file():
            try:
                info = sf.info(str(path))
                sample_info = {
                    "path": rel,
                    "sample_rate_hz": info.samplerate,
                    "channels": info.channels,
                    "duration_s": round(info.duration, 2),
                    "format": info.format,
                    "subtype": info.subtype,
                }
            except Exception as e:
                logger.warning(f"Could not read {path}: {e}")

    return {
        "total_files": len(wav_files),
        "condition_counts": counts,
        "sample_audio": sample_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect MIMII dataset")
    parser.add_argument("--local-dir", type=str, default="", help="Path to local audio directory")
    parser.add_argument("--remote", action="store_true", help="Query remote Zenodo record")
    args = parser.parse_args()

    if args.remote:
        info = inspect_remote_zenodo()
        print(json.dumps(info, indent=2))
    elif args.local_dir:
        info = inspect_local_directory(Path(args.local_dir))
        print(json.dumps(info, indent=2))
    else:
        # Default: try local data dir first, fallback to remote check
        data_dir = Path(__file__).resolve().parent.parent / "data"
        if any(data_dir.glob("**/*.wav")):
            info = inspect_local_directory(data_dir)
            print(json.dumps(info, indent=2))
        else:
            print("No local audio files found. Querying Zenodo record...")
            info = inspect_remote_zenodo()
            print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

