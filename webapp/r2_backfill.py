"""One-time backfill: push every local ``trading-reports/`` bundle to R2.

Idempotent — ``storage.put_file`` skips uploads where the same-size object
already exists at the destination key, so re-running this script is safe.

Usage (inside the webapp container, where boto3 + .env are available):

    docker compose run --rm --entrypoint python webapp -m webapp.r2_backfill
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# .env must be loaded before storage reads the AWS_* vars.
load_dotenv()
load_dotenv(".env.enterprise", override=False)

from webapp import reports as reports_mod  # noqa: E402
from webapp import storage  # noqa: E402


REPORTS_ROOT = Path("trading-reports")


def main() -> int:
    if not storage.is_enabled():
        print(
            "R2 not configured. Set AWS_ENDPOINT_URL_S3 / AWS_S3_BUCKET / "
            "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY in .env.",
            file=sys.stderr,
        )
        return 2

    if not REPORTS_ROOT.is_dir():
        print(f"No local {REPORTS_ROOT}/ — nothing to backfill.")
        return 0

    # Discover bundles by parsing folder names — keeps the script aligned
    # with how the webapp itself discovers them.
    candidates: list[Path] = []
    for p in sorted(REPORTS_ROOT.iterdir()):
        if not p.is_dir():
            continue
        if reports_mod.parse_bundle_name(p.name) is None:
            print(f"  skip non-bundle dir: {p.name}")
            continue
        candidates.append(p)

    if not candidates:
        print("No bundles found.")
        return 0

    print(f"Found {len(candidates)} bundle(s). Uploading to R2…")
    total_up = 0
    total_skip = 0
    for i, p in enumerate(candidates, start=1):
        print(f"[{i}/{len(candidates)}] {p.name} … ", end="", flush=True)
        up, skip = storage.put_bundle_dir(p, p.name)
        total_up += up
        total_skip += skip
        print(f"uploaded={up} skipped={skip}")

    storage.invalidate_list_cache()
    print(f"\nDone. {total_up} files uploaded, {total_skip} skipped/failed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
