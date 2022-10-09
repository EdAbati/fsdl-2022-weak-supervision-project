import json
import logging
from typing import Dict, Union

import numpy as np
import torch
from transformers import AutoTokenizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

CLASSES = ["World", "Sports", "Business", "Sci/Tech"]


class NewsTextClassifier:
    """Classify news text."""

    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
        self.tokenizer_checkpoint = "distilbert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_checkpoint)

    @torch.no_grad()
    def predict(self, text: str) -> torch.Tensor:
        tokenized_text = self.tokenizer(text, truncation=True, return_tensors="pt")
        # return tokenized_text
        y_pred = self.model(**tokenized_text)[0].softmax(dim=-1)
        return y_pred


model = NewsTextClassifier("traced_model.pt")


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


def model_predict(text: str) -> np.ndarray:
    """Returns random predictions"""
    logger.info("Predicting text labels")
    return model.predict(text=text).numpy()


def get_predicted_labels(text: str) -> Dict[str, float]:
    """Get predictions for each label"""
    predictions = model_predict(text).reshape(-1)
    return {label: float(pred) for label, pred in zip(CLASSES, predictions)}


def lambda_handler(event, context):
    event_dict = _json_str_to_dict(event)
    text = load_text(event_dict)
    if text is None:
        return {"statusCode": 400, "body": {"message": "'text' not found in body of request"}}
    predictions_dict = get_predicted_labels(text=text)
    return {"statusCode": 200, "body": {"predicted_labels": predictions_dict}}
