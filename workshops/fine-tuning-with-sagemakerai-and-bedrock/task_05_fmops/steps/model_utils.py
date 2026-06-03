# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Shared utilities for downloading model artifacts from HuggingFace
and uploading them to S3. Designed to be called from multiple notebooks
with idempotent behavior (download and upload happen only once).
"""

import os

import boto3
from botocore.exceptions import ClientError
from huggingface_hub import snapshot_download


def s3_file_exists(s3_client: "boto3.client", bucket: str, key: str) -> bool:
    """Check if a file exists in S3."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def upload_directory_to_s3(
    local_dir: str,
    s3_bucket: str,
    s3_prefix: str,
    skip_existing: bool = True,
) -> tuple:
    """
    Upload a local directory to S3, optionally skipping files that already exist.

    Args:
        local_dir: Local directory containing files to upload.
        s3_bucket: S3 bucket name.
        s3_prefix: S3 prefix (folder path).
        skip_existing: Whether to skip files that already exist in S3.

    Returns:
        Tuple of (uploaded_files, skipped_files, failed_files).
    """
    s3_client = boto3.client("s3")
    uploaded_files = []
    skipped_files = []
    failed_files = []

    # Collect all local files
    local_files = []
    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(s3_prefix, rel_path).replace("\\", "/")
            local_files.append((local_path, s3_key))

    print(f"Found {len(local_files)} files in {local_dir}")

    for local_path, s3_key in local_files:
        try:
            if skip_existing and s3_file_exists(s3_client, s3_bucket, s3_key):
                print(f"Skipping {s3_key} (already exists in S3)")
                skipped_files.append(s3_key)
                continue

            print(f"Uploading {local_path} to s3://{s3_bucket}/{s3_key}")
            s3_client.upload_file(
                local_path,
                s3_bucket,
                s3_key,
                ExtraArgs={"ACL": "bucket-owner-full-control"},
            )
            uploaded_files.append(s3_key)

        except Exception as e:
            print(f"Failed to upload {local_path}: {str(e)}")
            failed_files.append((s3_key, str(e)))

    print(f"\nUpload Summary:")
    print(f"  - Uploaded: {len(uploaded_files)} files")
    print(f"  - Skipped: {len(skipped_files)} files")
    print(f"  - Failed: {len(failed_files)} files")

    return uploaded_files, skipped_files, failed_files


def ensure_model_on_s3(
    model_id: str,
    bucket_name: str,
    default_prefix: str = "",
) -> str:
    """
    Ensure model artifacts are downloaded locally and uploaded to S3.

    This function is idempotent:
    - HuggingFace snapshot_download skips already-downloaded files locally.
    - upload_directory_to_s3 skips files that already exist in S3.

    Args:
        model_id: HuggingFace model ID (e.g. "Qwen/Qwen3-4B-Instruct-2507").
        bucket_name: S3 bucket name.
        default_prefix: Optional S3 prefix from SageMaker session.

    Returns:
        The S3 URI where the model is stored (e.g. s3://bucket/prefix/models/...).
    """
    model_id_filesafe = model_id.replace("/", "_").replace(".", "_")

    # Local path: task_05_fmops/models/<model_id_filesafe>
    model_local_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        model_id_filesafe,
    )

    # S3 destination
    if default_prefix:
        s3_prefix = f"{default_prefix}/models/{model_id_filesafe}"
    else:
        s3_prefix = f"models/{model_id_filesafe}"

    model_s3_uri = f"s3://{bucket_name}/{s3_prefix}"

    # Step 1: Download from HuggingFace (idempotent — skips existing files)
    print(f"Ensuring model {model_id} is downloaded to {model_local_dir}")
    os.makedirs(model_local_dir, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=model_local_dir)
    print(f"Model {model_id} available at {model_local_dir}")

    # Step 2: Upload to S3 (idempotent — skips existing files)
    print(f"Ensuring model is uploaded to {model_s3_uri}")
    uploaded, skipped, failed = upload_directory_to_s3(
        local_dir=model_local_dir,
        s3_bucket=bucket_name,
        s3_prefix=s3_prefix,
        skip_existing=True,
    )

    if failed:
        raise RuntimeError(
            f"Failed to upload {len(failed)} file(s) to S3. "
            f"First failure: {failed[0]}"
        )

    print(f"Model ready at: {model_s3_uri}")
    return model_s3_uri
