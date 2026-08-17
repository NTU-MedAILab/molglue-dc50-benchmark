"""Source acquisition, hashing, archive extraction, and snapshot validation."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


USER_AGENT = "MolGlue-DC50-reproducibility-builder/1.0"


class SnapshotMismatch(RuntimeError):
    """Raised when an official file differs from the frozen study snapshot."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_source_files(manifest: dict[str, Any], include_optional: bool = False) -> Iterable[tuple[str, dict[str, Any]]]:
    for source in manifest["sources"]:
        for item in source["files"]:
            if item.get("required", False) or include_optional:
                yield source["database"], item


def verify_file(path: Path, expected_sha256: str, expected_size: int | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_size = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if expected_size is not None and actual_size != expected_size:
        raise SnapshotMismatch(
            f"Size mismatch for {path.name}: expected {expected_size}, found {actual_size}. "
            "The official source may have been updated."
        )
    if actual_sha256 != expected_sha256:
        raise SnapshotMismatch(
            f"SHA-256 mismatch for {path.name}: expected {expected_sha256}, found {actual_sha256}. "
            "Do not mix this file into the frozen reconstruction; record it as a new snapshot."
        )
    return {"path": str(path), "size_bytes": actual_size, "sha256": actual_sha256}


def verify_required_sources(raw_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for database, item in iter_source_files(manifest, include_optional=False):
        result = verify_file(raw_dir / item["filename"], item["sha256"], item.get("size_bytes"))
        result.update({"database": database, "key": item["key"]})
        results.append(result)
        if item.get("archive_member"):
            extracted = verify_file(
                raw_dir / item["extracted_filename"],
                item["extracted_sha256"],
                item.get("extracted_size_bytes"),
            )
            extracted.update({"database": database, "key": f"{item['key']}:extracted"})
            results.append(extracted)
    return results


def _request_for(item: dict[str, Any]) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT, **item.get("headers", {})}
    if item["method"] == "GET":
        return urllib.request.Request(item["url"], headers=headers, method="GET")
    if item["method"] == "POST_JSON":
        payload = json.dumps(item["json_body"], separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/octet-stream,*/*"
        return urllib.request.Request(item["url"], data=payload, headers=headers, method="POST")
    raise ValueError(f"Unsupported download method: {item['method']}")


def _download_atomic(item: dict[str, Any], destination: Path, timeout: int = 180) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = _request_for(item)
    with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
            response_headers = dict(response.headers.items())
            final_url = response.geturl()
        verification = verify_file(temporary, item["sha256"], item.get("size_bytes"))
        temporary.replace(destination)
    except (urllib.error.URLError, OSError, SnapshotMismatch):
        temporary.unlink(missing_ok=True)
        raise
    verification.update(
        {
            "requested_url": item["url"],
            "final_url": final_url,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "response_headers": response_headers,
        }
    )
    return verification


def _safe_extract_member(archive_path: Path, member_name: str, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as archive:
        matches = [member for member in archive.getmembers() if Path(member.name).name == member_name]
        if len(matches) != 1 or not matches[0].isfile():
            raise RuntimeError(f"Expected exactly one regular archive member named {member_name!r}")
        extracted = archive.extractfile(matches[0])
        if extracted is None:
            raise RuntimeError(f"Could not read archive member {member_name!r}")
        with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            shutil.copyfileobj(extracted, handle, length=1024 * 1024)
    return {"temporary": temporary, "archive_member": matches[0].name}


def download_sources(
    raw_dir: Path,
    manifest: dict[str, Any],
    *,
    include_optional: bool = False,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Download frozen official snapshots and refuse any byte-level mismatch."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    acquisition_log: list[dict[str, Any]] = []
    for database, item in iter_source_files(manifest, include_optional=include_optional):
        destination = raw_dir / item["filename"]
        if destination.exists() and not force:
            entry = verify_file(destination, item["sha256"], item.get("size_bytes"))
            entry["status"] = "existing_verified"
        else:
            entry = _download_atomic(item, destination)
            entry["status"] = "downloaded_verified"
        entry.update({"database": database, "key": item["key"]})
        acquisition_log.append(entry)

        if item.get("archive_member"):
            extracted_destination = raw_dir / item["extracted_filename"]
            if extracted_destination.exists() and not force:
                extracted_entry = verify_file(
                    extracted_destination,
                    item["extracted_sha256"],
                    item.get("extracted_size_bytes"),
                )
                extracted_entry["status"] = "existing_verified"
            else:
                extracted_info = _safe_extract_member(destination, item["archive_member"], extracted_destination)
                temporary = extracted_info.pop("temporary")
                try:
                    extracted_entry = verify_file(
                        temporary,
                        item["extracted_sha256"],
                        item.get("extracted_size_bytes"),
                    )
                    temporary.replace(extracted_destination)
                except (OSError, SnapshotMismatch):
                    temporary.unlink(missing_ok=True)
                    raise
                extracted_entry.update(extracted_info)
                extracted_entry["status"] = "extracted_verified"
            extracted_entry.update({"database": database, "key": f"{item['key']}:extracted"})
            acquisition_log.append(extracted_entry)

    log_path = raw_dir / "acquisition_log.json"
    with log_path.open("w", encoding="utf-8") as handle:
        json.dump(acquisition_log, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return acquisition_log

