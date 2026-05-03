"""R2 (S3-compatible) storage backend for trading-report bundles.

Cloudflare R2 is the canonical store when the AWS_* env vars are present.
Local ``trading-reports/`` stays as a fallback so:

  - Old bundles that haven't been backfilled still browse.
  - The window between ``save_run_bundle()`` returning and the R2 upload
    finishing doesn't 404 the user.
  - Disabling R2 (unsetting env) cleanly degrades to the original behavior.

Bucket layout mirrors the on-disk one, under a ``reports/`` prefix:

    s3://<bucket>/reports/<bundle_name>/complete_report.md
    s3://<bucket>/reports/<bundle_name>/forecast.png
    s3://<bucket>/reports/<bundle_name>/1_analysts/market.md
    ...

Public surface:

  - ``is_enabled()``                       — env vars present
  - ``list_bundle_names()``                — TTL-cached list of bundle names
  - ``bundle_exists(name)``
  - ``get_text(name, relpath)``            — UTF-8 text fetch
  - ``stream_object(name, relpath)``       — (iterator, content_type, length)
  - ``list_bundle_files(name, subdir)``    — keys under bundle (or subdir)
  - ``put_bundle_dir(local_dir, name)``    — upload entire bundle (idempotent)
"""

from __future__ import annotations

import mimetypes
import os
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

# boto3 is only required when R2 is enabled. Import lazily inside helpers
# so a no-R2 environment doesn't pay the import cost or fail if the dep
# isn't installed.

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PREFIX = "reports"  # bucket key prefix; aligns with the local dir name's role
_LIST_TTL = 30.0    # seconds — short cache so new uploads appear quickly
_LIST_CACHE: dict[str, tuple[float, list[str]]] = {}
_LIST_LOCK = threading.Lock()


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def is_enabled() -> bool:
    """All four env vars must be set for R2 to be considered configured."""
    return all(
        _env(k)
        for k in (
            "AWS_ENDPOINT_URL_S3",
            "AWS_S3_BUCKET",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        )
    )


def _bucket() -> str:
    return _env("AWS_S3_BUCKET")


_CLIENT = None
_CLIENT_LOCK = threading.Lock()


def _client():
    """Lazy boto3 S3 client pointed at R2."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    with _CLIENT_LOCK:
        if _CLIENT is not None:
            return _CLIENT
        import boto3
        from botocore.config import Config

        _CLIENT = boto3.client(
            "s3",
            endpoint_url=_env("AWS_ENDPOINT_URL_S3"),
            aws_access_key_id=_env("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=_env("AWS_SECRET_ACCESS_KEY"),
            # R2 expects "auto" — boto3 still requires a value here.
            region_name=_env("AWS_REGION") or "auto",
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
                # R2 doesn't support virtual-host-style by default.
                s3={"addressing_style": "path"},
            ),
        )
    return _CLIENT


# ---------------------------------------------------------------------------
# Listing
# ---------------------------------------------------------------------------


def list_bundle_names() -> list[str]:
    """Return all bundle directory names in R2 (without the prefix).

    Cached for ``_LIST_TTL`` seconds. Returns ``[]`` on any error so the
    webapp can fall back to local listing without crashing.
    """
    if not is_enabled():
        return []

    key = "bundles"
    now = time.time()
    with _LIST_LOCK:
        cached = _LIST_CACHE.get(key)
        if cached and cached[0] > now:
            return cached[1]

    try:
        client = _client()
        paginator = client.get_paginator("list_objects_v2")
        names: set[str] = set()
        for page in paginator.paginate(
            Bucket=_bucket(),
            Prefix=f"{PREFIX}/",
            Delimiter="/",
        ):
            for cp in page.get("CommonPrefixes") or []:
                # cp["Prefix"] = "reports/<name>/"
                p = cp["Prefix"]
                if not p.startswith(f"{PREFIX}/") or not p.endswith("/"):
                    continue
                name = p[len(PREFIX) + 1 : -1]
                if name:
                    names.add(name)
        out = sorted(names)
    except Exception:
        out = []

    with _LIST_LOCK:
        _LIST_CACHE[key] = (now + _LIST_TTL, out)
    return out


def invalidate_list_cache() -> None:
    """Drop the TTL cache (call after an upload so fresh state appears)."""
    with _LIST_LOCK:
        _LIST_CACHE.clear()


def bundle_exists(name: str) -> bool:
    """Cheap existence check — list one object under the prefix."""
    if not is_enabled():
        return False
    try:
        resp = _client().list_objects_v2(
            Bucket=_bucket(),
            Prefix=f"{PREFIX}/{name}/",
            MaxKeys=1,
        )
        return resp.get("KeyCount", 0) > 0
    except Exception:
        return False


def list_bundle_files(name: str, subdir: str = "") -> list[str]:
    """Relative paths of every object under a bundle (optionally a subdir).

    Returned strings are bundle-relative (e.g. ``"1_analysts/market.md"``).
    """
    if not is_enabled():
        return []
    prefix = f"{PREFIX}/{name}/"
    if subdir:
        prefix += subdir.rstrip("/") + "/"
    try:
        client = _client()
        paginator = client.get_paginator("list_objects_v2")
        out: list[str] = []
        for page in paginator.paginate(Bucket=_bucket(), Prefix=prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                rel = key[len(f"{PREFIX}/{name}/") :]
                if rel:
                    out.append(rel)
        out.sort()
        return out
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def _key(name: str, relpath: str) -> str:
    return f"{PREFIX}/{name}/{relpath.lstrip('/')}"


def get_text(name: str, relpath: str) -> Optional[str]:
    """Return UTF-8 text for an object, or None on miss/error."""
    if not is_enabled():
        return None
    try:
        resp = _client().get_object(Bucket=_bucket(), Key=_key(name, relpath))
        return resp["Body"].read().decode("utf-8", errors="replace")
    except Exception:
        return None


def stream_object(
    name: str, relpath: str
) -> Optional[tuple[Iterator[bytes], str, Optional[int]]]:
    """Stream an object's bytes back. Returns None if missing.

    Returned tuple: (chunk iterator, content_type, content_length).
    """
    if not is_enabled():
        return None
    try:
        resp = _client().get_object(Bucket=_bucket(), Key=_key(name, relpath))
    except Exception:
        return None

    body = resp["Body"]
    content_type = resp.get("ContentType") or _guess_content_type(relpath)
    length = resp.get("ContentLength")

    def _it() -> Iterator[bytes]:
        try:
            for chunk in body.iter_chunks(chunk_size=64 * 1024):
                yield chunk
        finally:
            body.close()

    return _it(), content_type, length


def _guess_content_type(relpath: str) -> str:
    ct, _ = mimetypes.guess_type(relpath)
    return ct or "application/octet-stream"


# ---------------------------------------------------------------------------
# Writes (used by the run finisher and the one-time backfill)
# ---------------------------------------------------------------------------


def put_file(local_path: Path, name: str, relpath: str) -> bool:
    """Upload one file. Skips if the same-size object already exists."""
    if not is_enabled():
        return False
    if not local_path.is_file():
        return False
    key = _key(name, relpath)
    client = _client()

    # Skip if a same-size object exists — makes the backfill safely idempotent.
    try:
        head = client.head_object(Bucket=_bucket(), Key=key)
        if head.get("ContentLength") == local_path.stat().st_size:
            return True
    except Exception:
        pass

    try:
        client.upload_file(
            Filename=str(local_path),
            Bucket=_bucket(),
            Key=key,
            ExtraArgs={"ContentType": _guess_content_type(relpath)},
        )
        return True
    except Exception:
        return False


def put_bundle_dir(local_dir: Path, name: Optional[str] = None) -> tuple[int, int]:
    """Upload every file under ``local_dir`` to ``reports/<name>/...``.

    Returns ``(uploaded, skipped)``.
    """
    if not is_enabled():
        return (0, 0)
    if not local_dir.is_dir():
        return (0, 0)
    bundle_name = name or local_dir.name

    uploaded = 0
    skipped = 0
    for f in sorted(local_dir.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(local_dir).as_posix()
        ok = put_file(f, bundle_name, rel)
        if ok:
            uploaded += 1
        else:
            skipped += 1

    invalidate_list_cache()
    return uploaded, skipped
