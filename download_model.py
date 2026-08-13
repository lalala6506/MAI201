"""
download_model.py
------------------
Downloads model.pkl, scaler.pkl, and feature_columns.json directly from
the DVC S3 remote using boto3, without needing DVC or git present in the
container at all.

Why this exists: `dvc pull` requires being inside a git repository to
resolve which remote and revision to use. Docker excludes .git/ from the
build context, and even working around that to copy .git/ in is fragile
and bloats the image for no real benefit. This skips DVC entirely at
runtime and talks to S3 directly instead.

How it works: DVC never stores a file under its real name in the remote.
It stores every tracked file under a path built from its md5 hash:
<remote_prefix>/<first two hash characters>/<remaining hash characters>.
Those hashes already live in dvc.lock, a plain YAML file with no git
dependency. This script reads dvc.lock, finds the hash for each file we
need, and downloads it straight from S3 using that hash-based path.

Run this before starting the API, see entrypoint.sh.
"""

import os
import sys

import boto3
import yaml

BUCKET = "mlops-mai102-dvc"
REMOTE_PREFIX = "dvcdata/files/md5"
REGION = "us-east-2"

FILES_NEEDED = [
    "models/model.pkl",
    "models/scaler.pkl",
    "data/processed/feature_columns.json",
]


def find_hash(lock_data: dict, target_path: str):
    """Search every stage's deps and outs for a matching path, return
    its md5 hash if found."""
    for stage in lock_data.get("stages", {}).values():
        for section in ("deps", "outs"):
            for entry in stage.get(section, []):
                if entry.get("path") == target_path:
                    return entry.get("md5")
    return None


def main():
    if not os.path.exists("dvc.lock"):
        print("ERROR: dvc.lock not found. Run this from the project root.")
        sys.exit(1)

    with open("dvc.lock") as f:
        lock_data = yaml.safe_load(f)

    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        print(
            "ERROR: AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set "
            "as environment variables. Locally, put them in .env. On "
            "Render, add them under the service's Environment tab."
        )
        sys.exit(1)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=REGION,
    )

    for path in FILES_NEEDED:
        md5 = find_hash(lock_data, path)
        if md5 is None:
            print(f"ERROR: no hash found for '{path}' in dvc.lock. "
                  f"Has the pipeline been run since this file was added?")
            sys.exit(1)

        key = f"{REMOTE_PREFIX}/{md5[:2]}/{md5[2:]}"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        print(f"Downloading {path}  (s3://{BUCKET}/{key})")
        s3.download_file(BUCKET, key, path)

    print("All model artifacts downloaded successfully.")


if __name__ == "__main__":
    main()