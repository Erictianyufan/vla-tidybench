#!/usr/bin/env python3
"""Bridge a public GCS checkpoint through a networked host to an SSH server."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import requests

CHUNK_SIZE = 16 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prefix", help="copy every object below this GCS prefix")
    source.add_argument("--object", help="copy one exact GCS object into remote-root")
    parser.add_argument("--remote", required=True, help="SSH host or config alias")
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path(os.environ.get("TEMP", "/tmp")) / "vla-tidybench-gcs-transfer",
    )
    parser.add_argument("--download-workers", type=int, default=4)
    parser.add_argument("--range-size-mb", type=int, default=64)
    return parser.parse_args()


def list_objects(session: requests.Session, bucket: str, prefix: str) -> list[dict[str, str]]:
    objects: list[dict[str, str]] = []
    page_token: str | None = None
    while True:
        params = {"prefix": prefix, "maxResults": "1000"}
        if page_token:
            params["pageToken"] = page_token
        response = session.get(
            f"https://storage.googleapis.com/storage/v1/b/{quote(bucket, safe='')}/o",
            params=params,
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        objects.extend(item for item in payload.get("items", []) if not item["name"].endswith("/"))
        page_token = payload.get("nextPageToken")
        if not page_token:
            return objects


def get_public_object(session: requests.Session, bucket: str, object_name: str) -> dict[str, str]:
    """Read size and checksum without requiring anonymous bucket-list access."""

    url = f"https://storage.googleapis.com/{quote(bucket, safe='')}/{quote(object_name, safe='/')}"
    response = session.head(url, timeout=90)
    response.raise_for_status()
    hashes = {
        key.strip(): value.strip()
        for item in response.headers.get("x-goog-hash", "").split(",")
        if "=" in item
        for key, value in [item.split("=", 1)]
    }
    if "md5" not in hashes or "Content-Length" not in response.headers:
        raise RuntimeError(f"GCS object metadata has no size/MD5: gs://{bucket}/{object_name}")
    return {"name": object_name, "size": response.headers["Content-Length"], "md5Hash": hashes["md5"]}


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_md5(item: dict[str, str]) -> str:
    return base64.b64decode(item["md5Hash"]).hex()


def download_object(
    session: requests.Session,
    *,
    bucket: str,
    item: dict[str, str],
    destination: Path,
    workers: int,
    range_size: int,
) -> None:
    expected_size = int(item["size"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if workers < 1 or range_size < 1:
        raise ValueError("download workers and range size must be positive")
    state_path = destination.with_suffix(destination.suffix + ".ranges.json")
    url = f"https://storage.googleapis.com/{quote(bucket, safe='')}/{quote(item['name'], safe='/')}"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("expected_size") != expected_size:
            raise RuntimeError(f"stale range state: {state_path}")
        completed_ranges = {tuple(value) for value in state.get("completed_ranges", [])}
    else:
        prefix_size = destination.stat().st_size if destination.exists() else 0
        if prefix_size > expected_size:
            destination.unlink()
            prefix_size = 0
        completed_ranges = {(0, prefix_size - 1)} if prefix_size else set()
        state = {"expected_size": expected_size, "completed_ranges": [list(value) for value in completed_ranges]}
        state_path.write_text(json.dumps(state), encoding="utf-8")
    with destination.open("ab") as output:
        output.truncate(expected_size)

    covered_prefix = max((end + 1 for start, end in completed_ranges if start == 0), default=0)
    ranges = [
        (start, min(start + range_size - 1, expected_size - 1))
        for start in range(covered_prefix, expected_size, range_size)
    ]
    pending = [value for value in ranges if value not in completed_ranges]
    lock = threading.Lock()
    completed_bytes = sum(end - start + 1 for start, end in completed_ranges)

    def fetch(byte_range: tuple[int, int]) -> int:
        start, end = byte_range
        expected_length = end - start + 1
        for attempt in range(1, 9):
            try:
                response = session.get(
                    url,
                    headers={"Range": f"bytes={start}-{end}"},
                    timeout=(30, 180),
                )
                response.raise_for_status()
                if response.status_code != 206 or len(response.content) != expected_length:
                    raise RuntimeError(
                        f"unexpected range response {response.status_code}/{len(response.content)}"
                    )
                with destination.open("r+b", buffering=0) as output:
                    output.seek(start)
                    output.write(response.content)
                with lock:
                    completed_ranges.add(byte_range)
                    state["completed_ranges"] = [list(value) for value in sorted(completed_ranges)]
                    temporary = state_path.with_suffix(state_path.suffix + ".tmp")
                    temporary.write_text(json.dumps(state), encoding="utf-8")
                    temporary.replace(state_path)
                return expected_length
            except (requests.RequestException, OSError, RuntimeError) as error:
                print(f"  range {start}-{end} attempt {attempt}/8 interrupted: {error}", flush=True)
                if attempt < 8:
                    time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"range download failed: {start}-{end}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, byte_range) for byte_range in pending]
        for future in concurrent.futures.as_completed(futures):
            completed_bytes += future.result()
            print(f"  downloaded {completed_bytes / 1e9:.2f}/{expected_size / 1e9:.2f} GB", flush=True)
    state_path.unlink(missing_ok=True)


def run(command: list[str], *, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=capture_output)


def main() -> int:
    args = parse_args()
    args.identity_file = args.identity_file.expanduser().resolve()
    args.staging_dir = args.staging_dir.expanduser().resolve()
    args.staging_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    if args.object:
        object_name = args.object.lstrip("/")
        objects = [get_public_object(session, args.bucket, object_name)]
        relative_names = {object_name: PurePosixPath(object_name).name}
        source_label = object_name
    else:
        prefix = args.prefix.lstrip("/")
        if not prefix.endswith("/"):
            prefix += "/"
        objects = list_objects(session, args.bucket, prefix)
        relative_names = {item["name"]: item["name"][len(prefix) :] for item in objects}
        source_label = prefix
    remote_root = args.remote_root.rstrip("/")
    ssh_options = ["-i", str(args.identity_file), "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes"]

    if not objects:
        raise RuntimeError(f"no objects found for gs://{args.bucket}/{source_label}")
    total_size = sum(int(item["size"]) for item in objects)
    print(f"Transferring {len(objects)} objects ({total_size} bytes)", flush=True)

    for index, item in enumerate(objects, start=1):
        relative_path = relative_names[item["name"]]
        local_path = args.staging_dir / Path(*PurePosixPath(relative_path).parts)
        remote_path = f"{remote_root}/{relative_path}"
        size = int(item["size"])
        digest = expected_md5(item)
        print(f"[{index}/{len(objects)}] {relative_path} ({size} bytes)", flush=True)

        remote_check = subprocess.run(
            [
                "ssh",
                *ssh_options,
                args.remote,
                (
                    f"test \"$(stat -c %s {shlex.quote(remote_path)} 2>/dev/null)\" = {size} && "
                    f"md5sum {shlex.quote(remote_path)} | cut -d' ' -f1"
                ),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if remote_check.returncode == 0 and remote_check.stdout.strip() == digest:
            local_path.unlink(missing_ok=True)
            local_path.with_suffix(local_path.suffix + ".ranges.json").unlink(missing_ok=True)
            print(f"[{index}/{len(objects)}] already verified on {args.remote}", flush=True)
            continue

        if not local_path.exists() or local_path.stat().st_size != size or md5_file(local_path) != digest:
            download_object(
                session,
                bucket=args.bucket,
                item=item,
                destination=local_path,
                workers=args.download_workers,
                range_size=args.range_size_mb * 1024 * 1024,
            )
        if local_path.stat().st_size != size or md5_file(local_path) != digest:
            raise RuntimeError(f"local verification failed: {relative_path}")

        remote_parent = str(PurePosixPath(remote_path).parent)
        run(["ssh", *ssh_options, args.remote, f"mkdir -p -- {shlex.quote(remote_parent)}"])
        run(["scp", *ssh_options, str(local_path), f"{args.remote}:{remote_path}"])
        verification = run(
            [
                "ssh",
                *ssh_options,
                args.remote,
                (
                    f"test \"$(stat -c %s {shlex.quote(remote_path)})\" = {size} && "
                    f"md5sum {shlex.quote(remote_path)} | cut -d' ' -f1"
                ),
            ],
            capture_output=True,
        )
        if verification.stdout.strip() != digest:
            raise RuntimeError(f"remote verification failed: {relative_path}")

        local_path.unlink()
        print(f"[{index}/{len(objects)}] verified on {args.remote}", flush=True)

    print(f"Transfer complete: {args.remote}:{remote_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
