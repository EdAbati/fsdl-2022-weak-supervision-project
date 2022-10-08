import json
import logging
import os
from typing import Dict, Union

import boto3
import numpy as np
import torch
from transformers import AutoTokenizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

rng = np.random.default_rng()
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]


class NewsTextClassifier:
    """Classify news text."""

    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.tokenizer_checkpoint = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_checkpoint
        )

    @torch.no_grad()
    def predict(self, text: str) -> torch.Tensor:
        tokenized_text = self.tokenizer(
            text, truncation=True, return_tensors="pt"
        )
        # returns tokenized_text
        y_pred = self.model(**tokenized_text)[0].softmax(dim=-1)
        return y_pred


def load_model_from_s3(
    bucket_name: str = "fsdl-model-test",
    s3_filename: str = "model.pt",
) -> NewsTextClassifier:
    """Load model from S3"""
    logger.info(f"Loading model from S3: {bucket_name}/{s3_filename}")

    s3 = boto3.client(
        "s3",
        os.environ.get("AWS_REGION"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    # upload file from local directory to s3 bucket
    destination = "/tmp/model.pt"

    s3.download_file(bucket_name, s3_filename, destination)

    return NewsTextClassifier(destination)


def _json_str_to_dict(string_json: Union[dict, str]) -> dict:
    if isinstance(string_json, str):
        return json.loads(string_json)
    return string_json


def load_text(event_dict: dict) -> Union[str, None]:
    body = event_dict.get("body")
    if body is not None:
        logger.info("Getting 'text' from request body")
        body = _json_str_to_dict(body)
    else:
        body = event_dict
    text = body.get("text")
    return text


def get_predicted_labels(
    model: NewsTextClassifier, text: str
) -> Dict[str, float]:
    """Get predictions for each label"""
    predictions = model.predict(text=text).numpy()
    return dict(zip(CLASSES, predictions.reshape(-1)))


def lambda_handler(event, context):

    s3_model = load_model_from_s3()
    event_dict = _json_str_to_dict(event)
    text = load_text(event_dict)

    if text is None:
        return {
            "statusCode": 400,
            "body": {"message": "'text' not found in body of request"},
        }

    predictions_dict = get_predicted_labels(model=s3_model, text=text)

    return {
        "statusCode": 200,
        "body": json.dumps({"predicted_labels": predictions_dict}),
    }
