#!/usr/local/bin/python3

import argparse
import logging
import os
from pathlib import Path

import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model_from_s3(
    bucket_name: str,
    s3_filename: str,
    destination: Path,
) -> None:
    """Load model from S3"""
    logger.info(f"Loading model from S3: {bucket_name}/{s3_filename}")

    s3 = boto3.client(
        "s3",
        os.environ.get("AWS_REGION"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    s3.download_file(bucket_name, s3_filename, str(destination))


def parse_args():
    parser = argparse.ArgumentParser(add_help=True, argument_default=True)
    parser.add_argument(
        "--bucket_name",
        type=str,
        default="fsdl-model-test",
        help="S3 bucket name",
    )

    parser.add_argument(
        "--s3_filename",
        type=str,
        default="traced-model.pt",
        help="S3 filename",
    )

    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/src/traced-model.pt"),
        help="Destination path",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(args)

    load_model_from_s3(args.bucket_name, args.s3_filename, args.destination)
