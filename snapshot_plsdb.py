#!/usr/bin/env python3
"""Pull a deterministic PLSDB sample and freeze it as normalized JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from build_dataset import pull_plsdb_records


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(output: Path, *, count: int, seed: int, overwrite: bool) -> dict[str, object]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} already exists; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    records = pull_plsdb_records(count, seed)
    if not records:
        raise RuntimeError("PLSDB pull returned no normalized records")
    identities = [str(record["record_identity"]) for record in records]
    if len(set(identities)) != len(identities):
        raise RuntimeError("PLSDB pull contains duplicate record identities")

    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output)

    manifest_path = output.with_suffix(".manifest.json")
    manifest = {
        "source": "PLSDB API via plsdbapi",
        "query": {
            "NUCCORE_Source": "RefSeq",
            "NUCCORE_Topology": "circular",
        },
        "sampling_seed": seed,
        "requested_records": count,
        "returned_records": len(records),
        "unique_record_identities": len(set(identities)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_path": str(output.resolve()),
        "snapshot_sha256": sha256_file(output),
        "durability_note": (
            "A Colab runtime filesystem is temporary. Copy this JSONL and manifest "
            "to persistent storage before ending the runtime."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest_path": str(manifest_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = snapshot(
        args.output,
        count=args.count,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
