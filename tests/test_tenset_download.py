"""Archive ingestion checks use small synthetic ZIPs, never benchmark data."""

import json
import stat
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile, ZipInfo

import pytest

from jev_compiler_eval.datasets.tenset_download import (
    CPU_ARCHIVE_URL,
    download_cpu_archive,
    extract_cpu_archive,
    sha256_file,
)


def make_archive(tmp_path: Path, entries: dict[str, bytes]) -> Path:
    archive = tmp_path / "synthetic.zip"
    with ZipFile(archive, "w") as output:
        for name, content in entries.items():
            output.writestr(name, content)
    return archive


def test_extracts_only_selected_platform_and_registry_with_provenance(tmp_path: Path):
    archive = make_archive(
        tmp_path,
        {
            "dataset_cpu/measure_records/e5-2673/workload.json": b"synthetic records\n",
            "dataset_cpu/measure_records/platinum-8272/other.json": b"not selected",
            "dataset_cpu/network_info/all_tasks.pkl": b"never unpickle this",
            "dataset_cpu/network_info/network.relay.pkl": b"not selected",
        },
    )
    output = tmp_path / "raw"
    manifest = extract_cpu_archive(
        archive, output, expected_sha256=sha256_file(archive)
    )
    assert (output / "measure_records/e5-2673/workload.json").read_bytes() == (
        b"synthetic records\n"
    )
    assert (output / "network_info/all_tasks.pkl").read_bytes() == (
        b"never unpickle this"
    )
    assert not (output / "measure_records/platinum-8272").exists()
    assert len(manifest["files"]) == 2
    assert manifest["expected_sha256_verified"] is True
    assert manifest["license"] == "CC BY 4.0"
    assert json.loads((output / "download-manifest.json").read_text()) == manifest
    for entry in manifest["files"]:
        assert sha256_file(output / entry["path"]) == entry["sha256"]


@pytest.mark.parametrize("include_network_tasks", [False, True])
def test_network_task_inventory_requires_explicit_opt_in(
    tmp_path: Path, include_network_tasks: bool
):
    archive = make_archive(
        tmp_path,
        {
            "dataset/measure_records/platinum-8272/workload.json": b"records",
            "dataset/measure_records/e5-2673/other.json": b"not selected",
            "dataset/network_info/all_tasks.pkl": b"registry",
            "dataset/network_info/resnet.task.pkl": b"task inventory",
            "dataset/network_info/bert.task.pkl": b"another task inventory",
            "dataset/network_info/resnet.relay.pkl": b"not selected",
        },
    )
    destination = tmp_path / "raw"
    manifest = extract_cpu_archive(
        archive,
        destination,
        platform="platinum-8272",
        include_network_tasks=include_network_tasks,
    )
    expected = {
        "measure_records/platinum-8272/workload.json",
        "network_info/all_tasks.pkl",
    }
    if include_network_tasks:
        expected.update({"network_info/resnet.task.pkl", "network_info/bert.task.pkl"})
        assert manifest["include_network_tasks"] is True
    else:
        assert "include_network_tasks" not in manifest
    assert {entry["path"] for entry in manifest["files"]} == expected
    assert not (destination / "network_info/resnet.relay.pkl").exists()
    for entry in manifest["files"]:
        assert sha256_file(destination / entry["path"]) == entry["sha256"]


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", [], {}])
def test_network_task_option_rejects_non_booleans(tmp_path: Path, invalid):
    with pytest.raises(ValueError, match="must be a boolean"):
        extract_cpu_archive(
            tmp_path / "unused.zip",
            tmp_path / "raw",
            include_network_tasks=invalid,
        )
    assert not (tmp_path / "raw").exists()


def test_network_task_destinations_reject_duplicate_archive_prefixes(tmp_path: Path):
    archive = make_archive(
        tmp_path,
        {
            "one/measure_records/platinum-8272/good.json": b"records",
            "one/network_info/resnet.task.pkl": b"first",
            "two/network_info/resnet.task.pkl": b"second",
        },
    )
    with pytest.raises(ValueError, match="duplicate archive destination"):
        extract_cpu_archive(
            archive,
            tmp_path / "raw",
            platform="platinum-8272",
            include_network_tasks=True,
        )
    assert not (tmp_path / "raw").exists()


def test_network_task_option_preserves_unselected_entry_validation(tmp_path: Path):
    archive = make_archive(
        tmp_path,
        {
            "dataset/measure_records/platinum-8272/good.json": b"records",
            "dataset/network_info/good.task.pkl": b"task inventory",
            "unselected/../bad.relay.pkl": b"unsafe",
        },
    )
    with pytest.raises(ValueError, match="unsafe archive path"):
        extract_cpu_archive(
            archive,
            tmp_path / "raw",
            platform="platinum-8272",
            include_network_tasks=True,
        )
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escaped",
        "/escaped",
        "dataset/../../escaped",
        "C:/escaped",
        "dir\\escaped",
    ],
)
def test_rejects_traversal_even_in_unselected_entries(tmp_path: Path, unsafe: str):
    archive = make_archive(
        tmp_path,
        {
            "dataset/measure_records/e5-2673/good.json": b"good",
            unsafe: b"bad",
        },
    )
    output = tmp_path / "raw"
    with pytest.raises(ValueError, match="unsafe archive path"):
        extract_cpu_archive(archive, output)
    assert not output.exists()
    assert not (tmp_path / "escaped").exists()


def test_rejects_archive_symlinks(tmp_path: Path):
    archive = tmp_path / "synthetic.zip"
    with ZipFile(archive, "w") as output:
        link = ZipInfo("dataset/measure_records/e5-2673/link.json")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        output.writestr(link, "../../escape")
    with pytest.raises(ValueError, match="symlinks"):
        extract_cpu_archive(archive, tmp_path / "raw")


def test_rejects_normalized_duplicate_destinations(tmp_path: Path):
    archive = make_archive(
        tmp_path,
        {
            "outer/measure_records/e5-2673/w.json": b"one",
            "other/measure_records/e5-2673/w.json": b"two",
        },
    )
    with pytest.raises(ValueError, match="duplicate archive destination"):
        extract_cpu_archive(archive, tmp_path / "raw")
    assert not (tmp_path / "raw").exists()


def test_sha_mismatch_and_missing_platform_do_not_publish_data(tmp_path: Path):
    archive = make_archive(
        tmp_path,
        {
            "measure_records/e5-2673/good.json": b"synthetic",
        },
    )
    with pytest.raises(ValueError, match="does not match"):
        extract_cpu_archive(archive, tmp_path / "raw", expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="contains no measurement"):
        extract_cpu_archive(archive, tmp_path / "raw", platform="other")
    assert not (tmp_path / "raw").exists()


def test_existing_data_and_symlink_destination_are_protected(tmp_path: Path):
    archive = make_archive(tmp_path, {"measure_records/e5-2673/good.json": b"new"})
    output = tmp_path / "raw"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("existing")
    with pytest.raises(FileExistsError):
        extract_cpu_archive(archive, output)
    assert marker.read_text() == "existing"
    alias = tmp_path / "alias"
    alias.symlink_to(output)
    with pytest.raises(FileExistsError):
        extract_cpu_archive(archive, alias)


def test_optional_gdown_download_is_atomic_and_preserves_existing_file(
    tmp_path: Path,
    monkeypatch,
):
    def fake_download(url, output, quiet):
        assert url == CPU_ARCHIVE_URL
        Path(output).write_bytes(b"synthetic downloaded ZIP")
        return output

    monkeypatch.setattr(
        "jev_compiler_eval.datasets.tenset_download.importlib.import_module",
        lambda name: SimpleNamespace(download=fake_download),
    )
    destination = tmp_path / "archive.zip"
    assert download_cpu_archive(destination) == destination
    assert destination.read_bytes() == b"synthetic downloaded ZIP"
    with pytest.raises(FileExistsError):
        download_cpu_archive(destination)


def test_failed_download_does_not_leave_archive(tmp_path: Path, monkeypatch):
    def fake_download(url, output, quiet):
        Path(output).write_bytes(b"partial")
        return None

    monkeypatch.setattr(
        "jev_compiler_eval.datasets.tenset_download.importlib.import_module",
        lambda name: SimpleNamespace(download=fake_download),
    )
    destination = tmp_path / "archive.zip"
    with pytest.raises(RuntimeError, match="download failed"):
        download_cpu_archive(destination)
    assert not destination.exists()


def test_missing_optional_dependency_has_actionable_error(tmp_path: Path, monkeypatch):
    def fail_import(name):
        raise ImportError("synthetic missing gdown")

    monkeypatch.setattr(
        "jev_compiler_eval.datasets.tenset_download.importlib.import_module",
        fail_import,
    )
    with pytest.raises(RuntimeError, match="uv sync --extra tenset"):
        download_cpu_archive(tmp_path / "archive.zip")
