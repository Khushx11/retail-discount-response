"""Validate the complete artefact set for a bootstrap replicate."""

import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile


def artifact_paths(root, rep):
    return [
        root / f"refit_schema_{rep}.json",
        root / f"refit_tree_{rep}.txt",
        root / f"refit_poisson_{rep}.pt",
        root / "refit_parts" / f"blocks_{rep}.csv",
    ]


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def artifact_hashes(root, rep):
    return {str(path.relative_to(root)): digest(path) for path in artifact_paths(root, rep)}


def read_complete(root, rep, feature_sha256):
    marker = root / "refit_parts" / f"rep_{rep}.json"
    try:
        record = json.loads(marker.read_text())
        if not isinstance(record, dict) or record.get("replicate") != rep:
            return None
        if record.get("feature_sha256") != feature_sha256:
            return None
        if record.get("artifact_sha256") != artifact_hashes(root, rep):
            return None
    except (OSError, ValueError):
        return None
    return record


def publish(root, record, feature_sha256):
    rep = record["replicate"]
    folder = root / "refit_parts"
    marker = folder / f"rep_{rep}.json"
    record = {
        **record,
        "feature_sha256": feature_sha256,
        "artifact_sha256": artifact_hashes(root, rep),
    }
    with NamedTemporaryFile(mode="w", dir=folder, suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(record, file, indent=2, allow_nan=False)
            file.flush()
            temporary.replace(marker)
        finally:
            temporary.unlink(missing_ok=True)
