"""Fetch the official TenSet CPU archive and safely extract one platform.

The upstream dataset is CC BY 4.0. Extraction retains its provenance and never
loads the trusted-only Python pickle containing the TVM workload registry.
"""

import hashlib
import importlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZipFile, ZipInfo

CPU_ARCHIVE_ID = "1JQwGEe8jCpuhZPnUxO0Sb1CJJ06uevy6"
CPU_ARCHIVE_URL = f"https://drive.google.com/uc?id={CPU_ARCHIVE_ID}"
ARCHIVE_VERSION = "dataset_cpu_v3.3"
SOURCE_URL = (
    "https://github.com/tlc-pack/tenset/blob/main/"
    "docs/get_started_with_cost_model_experiments.md"
)


def sha256_file(path: Path) -> str:
    """Hash a potentially large archive without reading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_cpu_archive(destination: Path) -> Path:
    """Download the official archive using optional gdown, without overwriting."""
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"archive already exists: {destination}")
    try:
        gdown = importlib.import_module("gdown")
    except ImportError as error:
        raise RuntimeError(
            "Downloading TenSet requires gdown; run uv sync --extra tenset "
            "or supply --archive with an existing CPU ZIP."
        ) from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".tenset-download-", dir=destination.parent
    ) as temporary:
        partial = Path(temporary) / "dataset_cpu_v3.3.zip"
        downloaded = gdown.download(CPU_ARCHIVE_URL, str(partial), quiet=False)
        if not downloaded or not partial.is_file():
            raise RuntimeError("TenSet download failed; no completed archive was saved")
        # A hard link publishes a complete file atomically and refuses collisions.
        os.link(partial, destination)
    return destination


def _archive_path(member: ZipInfo) -> PurePosixPath:
    name = member.filename
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or "\0" in name
        or path.is_absolute()
        or ".." in path.parts
        or any(":" in part for part in path.parts)
    ):
        raise ValueError(f"unsafe archive path: {name!r}")
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"archive symlinks are not supported: {name!r}")
    if member.flag_bits & 1:
        raise ValueError("encrypted TenSet archive entries are not supported")
    return path


def _selected_path(
    path: PurePosixPath,
    platform: str,
    *,
    include_network_tasks: bool = False,
) -> Path | None:
    parts = path.parts
    for index in range(len(parts)):
        tail = parts[index:]
        if (
            len(tail) == 3
            and tail[:2] == ("measure_records", platform)
            and tail[2].endswith((".json", ".jsonl"))
        ):
            return Path(*tail)
        if tail == ("network_info", "all_tasks.pkl"):
            return Path(*tail)
        if (
            include_network_tasks
            and len(tail) == 2
            and tail[0] == "network_info"
            and tail[1].endswith(".task.pkl")
        ):
            return Path(*tail)
    return None


def extract_cpu_archive(
    archive_path: Path,
    output_dir: Path,
    *,
    platform: str = "e5-2673",
    expected_sha256: str | None = None,
    include_network_tasks: bool = False,
) -> dict[str, object]:
    """Extract selected measurements atomically and record archive/file hashes.

    The archive is treated as untrusted ZIP data. The optional expected SHA must
    come from a separately trusted source; no official checksum is invented.
    """
    archive_path, output_dir = Path(archive_path), Path(output_dir)
    if type(include_network_tasks) is not bool:
        raise ValueError("include_network_tasks must be a boolean")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", platform):
        raise ValueError("platform must be a single directory name")
    if output_dir.is_symlink() or (
        output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir()))
    ):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    archive_hash = sha256_file(archive_path)
    if expected_sha256 is not None:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("expected_sha256 must contain 64 hexadecimal characters")
        if archive_hash != expected_sha256.lower():
            raise ValueError("TenSet archive SHA-256 does not match expected_sha256")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".tenset-extract-", dir=output_dir.parent
    ) as temporary:
        staging = Path(temporary) / "data"
        staging.mkdir()
        extracted = []
        destinations: set[Path] = set()
        with ZipFile(archive_path) as archive:
            for member in archive.infolist():
                path = _archive_path(member)
                if member.is_dir():
                    continue
                relative = _selected_path(
                    path, platform, include_network_tasks=include_network_tasks
                )
                if relative is None:
                    continue
                if relative in destinations:
                    raise ValueError(f"duplicate archive destination: {relative}")
                destinations.add(relative)
                destination = staging / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, destination.open("xb") as output:
                    shutil.copyfileobj(source, output)
                extracted.append(
                    {
                        "path": relative.as_posix(),
                        "bytes": destination.stat().st_size,
                        "sha256": sha256_file(destination),
                    }
                )
        if not any(item["path"].startswith("measure_records/") for item in extracted):
            raise ValueError(f"archive contains no measurement files for {platform}")
        manifest: dict[str, object] = {
            "schema_version": 1,
            "dataset": "tenset",
            "archive_version": ARCHIVE_VERSION,
            "archive_path": str(archive_path.resolve()),
            "archive_sha256": archive_hash,
            "expected_sha256_verified": expected_sha256 is not None,
            "official_download_url": CPU_ARCHIVE_URL,
            "source": SOURCE_URL,
            "license": "CC BY 4.0",
            "citation": "Zheng et al., TenSet, NeurIPS Datasets and Benchmarks 2021",
            "platform": platform,
            "files": sorted(extracted, key=lambda item: item["path"]),
        }
        if include_network_tasks:
            manifest["include_network_tasks"] = True
        (staging / "download-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        # Refuse output created while extraction was in progress.
        if output_dir.exists():
            output_dir.rmdir()
        staging.rename(output_dir)
    return manifest
