"""Download and extract checksum-verified TpuGraphs tile:xla archives."""

import hashlib
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class ArchiveSpec:
    """A published dataset archive and its expected contents."""

    url: str
    sha256: str
    npz_files: int


TILE_XLA_ARCHIVES: dict[str, ArchiveSpec] = {
    "train": ArchiveSpec(
        url="http://download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_train.tar",
        sha256="ef5c9b5a2e14ac5eb0c83e1f6421fe177e53fa72a65268ec58dcefa342fe25de",
        npz_files=5709,
    ),
    "valid": ArchiveSpec(
        url="http://download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_valid.tar",
        sha256="2a56dc731f59f6ffa7d312e8d07aa15307c136dd6bc3bb7bc04dc5fa6d526803",
        npz_files=676,
    ),
    "test": ArchiveSpec(
        url="http://download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_test.tar",
        sha256="a10487a99f33a1a64093507a02be1fe6006fc4480094d36549b00fbec87007d5",
        npz_files=844,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_sha256: str) -> None:
    partial = destination.with_name(f"{destination.name}.part")
    try:
        with (
            urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
            partial.open("wb") as file,
        ):
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                file.write(chunk)
        actual = _sha256(partial)
        if actual != expected_sha256:
            raise ValueError(
                f'checksum mismatch for "{url}": expected {expected_sha256}, '
                f"got {actual}"
            )
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def _split_directory(data_dir: Path, split: str) -> Path:
    return data_dir / "npz" / "tile" / "xla" / split


def _npz_count(directory: Path) -> int:
    return sum(1 for _ in directory.glob("*.npz"))


def download_tile_archives(
    data_dir: Path,
    splits: Sequence[str],
    *,
    archives: Mapping[str, ArchiveSpec] = TILE_XLA_ARCHIVES,
    log: Callable[[str], None] = print,
) -> None:
    """Download, verify, and extract tile:xla splits into ``data_dir``.

    Verified archives and fully extracted splits are reused. An existing
    archive with a different checksum is an error and is never overwritten.
    """

    unknown = [split for split in splits if split not in archives]
    if unknown:
        raise ValueError(f"unknown TpuGraphs split(s): {', '.join(unknown)}")

    archive_dir = data_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        spec = archives[split]
        archive = archive_dir / Path(spec.url).name
        split_dir = _split_directory(data_dir, split)

        if archive.exists():
            if _sha256(archive) != spec.sha256:
                raise ValueError(
                    f'existing archive "{archive}" has an unexpected checksum; '
                    "delete it and rerun"
                )
            log(f"{split}: archive verified")
        else:
            log(f"{split}: downloading {spec.url}")
            _download(spec.url, archive, spec.sha256)
            log(f"{split}: archive verified")

        if split_dir.is_dir() and _npz_count(split_dir) == spec.npz_files:
            log(f"{split}: already extracted to {split_dir}")
            continue

        with tarfile.open(archive) as tar:
            tar.extractall(data_dir, filter="data")
        count = _npz_count(split_dir) if split_dir.is_dir() else 0
        if count != spec.npz_files:
            raise ValueError(
                f'expected {spec.npz_files} .npz files in "{split_dir}", found {count}'
            )
        log(f"{split}: extracted {count} graphs to {split_dir}")
