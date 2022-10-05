import json
import logging
from typing import Dict, Union

import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

rng = np.random.default_rng()
CLASSES = ["World", "Sports", "Business", "Sci/Tech"]


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
    return rng.dirichlet(np.ones(4), size=1)


def get_predicted_labels(text: str) -> Dict[str, float]:
    """Get predictions for each label"""
    predictions = model_predict(text)
    return dict(zip(CLASSES, predictions.reshape(-1)))


def lambda_handler(event, context):
    event_dict = _json_str_to_dict(event)
    text = load_text(event_dict)
    if text is None:
        return {
            "statusCode": 400,
            "body": {"message": "'text' not found in body of request"},
        }
    predictions_dict = get_predicted_labels(text=text)
    return {
        "statusCode": 200,
        "body": json.dumps({"predicted_labels": predictions_dict}),
    }
