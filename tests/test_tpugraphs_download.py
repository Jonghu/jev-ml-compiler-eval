import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from jev_compiler_eval.datasets.tpugraphs_download import (
    TILE_XLA_ARCHIVES,
    ArchiveSpec,
    download_tile_archives,
)


def _make_archive(path: Path, members: dict[str, bytes]) -> ArchiveSpec:
    with tarfile.open(path, "w") as tar:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return ArchiveSpec(
        url=path.as_uri(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        npz_files=sum(name.endswith(".npz") for name in members),
    )


@pytest.fixture
def valid_archive(tmp_path: Path) -> ArchiveSpec:
    source = tmp_path / "source"
    source.mkdir()
    return _make_archive(
        source / "npz_tile_xla_valid.tar",
        {
            "npz/tile/xla/valid/a_1.npz": b"a",
            "npz/tile/xla/valid/b_2.npz": b"b",
        },
    )


def test_published_archives_cover_all_splits() -> None:
    assert set(TILE_XLA_ARCHIVES) == {"train", "valid", "test"}


def test_downloads_verifies_and_extracts(
    tmp_path: Path, valid_archive: ArchiveSpec
) -> None:
    data_dir = tmp_path / "data"

    download_tile_archives(
        data_dir, ["valid"], archives={"valid": valid_archive}, log=lambda _: None
    )

    assert (data_dir / "archives" / "npz_tile_xla_valid.tar").is_file()
    extracted = sorted((data_dir / "npz" / "tile" / "xla" / "valid").iterdir())
    assert [path.name for path in extracted] == ["a_1.npz", "b_2.npz"]
    assert not list((data_dir / "archives").glob("*.part"))


def test_reuses_verified_archive_and_extracted_split(
    tmp_path: Path, valid_archive: ArchiveSpec
) -> None:
    data_dir = tmp_path / "data"
    archives = {"valid": valid_archive}
    download_tile_archives(data_dir, ["valid"], archives=archives, log=lambda _: None)
    messages: list[str] = []

    download_tile_archives(data_dir, ["valid"], archives=archives, log=messages.append)

    assert messages == [
        "valid: archive verified",
        f"valid: already extracted to {data_dir / 'npz' / 'tile' / 'xla' / 'valid'}",
    ]


def test_rejects_download_with_wrong_checksum(
    tmp_path: Path, valid_archive: ArchiveSpec
) -> None:
    data_dir = tmp_path / "data"
    corrupted = ArchiveSpec(valid_archive.url, "0" * 64, valid_archive.npz_files)

    with pytest.raises(ValueError, match="checksum mismatch"):
        download_tile_archives(
            data_dir, ["valid"], archives={"valid": corrupted}, log=lambda _: None
        )

    assert list((data_dir / "archives").iterdir()) == []


def test_never_overwrites_existing_archive_with_wrong_checksum(
    tmp_path: Path, valid_archive: ArchiveSpec
) -> None:
    archive = tmp_path / "data" / "archives" / "npz_tile_xla_valid.tar"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"stale")

    with pytest.raises(ValueError, match="unexpected checksum"):
        download_tile_archives(
            tmp_path / "data",
            ["valid"],
            archives={"valid": valid_archive},
            log=lambda _: None,
        )

    assert archive.read_bytes() == b"stale"


def test_rejects_archive_with_unexpected_file_count(
    tmp_path: Path, valid_archive: ArchiveSpec
) -> None:
    short = ArchiveSpec(valid_archive.url, valid_archive.sha256, npz_files=3)

    with pytest.raises(ValueError, match="expected 3 .npz files"):
        download_tile_archives(
            tmp_path / "data", ["valid"], archives={"valid": short}, log=lambda _: None
        )


def test_refuses_archive_members_outside_data_dir(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    spec = _make_archive(source / "evil.tar", {"../escaped.npz": b"x"})

    with pytest.raises(tarfile.OutsideDestinationError):
        download_tile_archives(
            tmp_path / "data", ["valid"], archives={"valid": spec}, log=lambda _: None
        )

    assert not (tmp_path / "escaped.npz").exists()


def test_rejects_unknown_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown TpuGraphs split"):
        download_tile_archives(tmp_path, ["holdout"], log=lambda _: None)
