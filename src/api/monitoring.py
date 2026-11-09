"""
Basic input monitoring for the prediction API.

- Every prediction is logged as one JSON line (inputs, output, model version),
  so the logs (e.g. Render logs) can later be analysed for drift.
- Inputs far outside the range seen in training are logged as a WARNING. With
  this dataset the features arrive already scaled (by the intermediate backend),
  so a warning here usually means the values were not scaled as expected.

The training range comes from the champion run's feature_stats.json artifact
(logged by train.py). If it is missing (older models), the range check is skipped.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# How far (as a fraction of the training range) a value may go beyond the
# training min/max before it is flagged.
RANGE_TOLERANCE = 0.25


def load_feature_stats(path: Path) -> dict | None:
    """Return {feature: {"min": .., "max": ..}} from ``path``, or None if it does not exist."""

    if not path.exists():
        logger.warning("No %s for this model — input range check disabled.", path.name)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def out_of_range_features(features: dict, stats: dict | None) -> list[str]:
    """Return the names of the features whose value is far outside the training range."""

    if not stats:
        return []

    flagged = []
    for name, value in features.items():
        if name not in stats:
            continue
        low, high = stats[name]["min"], stats[name]["max"]
        margin = RANGE_TOLERANCE * (high - low)
        if not (low - margin <= value <= high + margin):
            flagged.append(name)
    return flagged


def log_prediction(features: dict, prediction: float, model_version: str | None, flagged: list[str]) -> None:
    """Log one prediction as a single JSON line (and a warning if inputs look wrong)."""

    if flagged:
        logger.warning("Input out of training range: %s", ", ".join(flagged))

    record = {
        "event": "prediction",
        "model_version": model_version,
        "features": features,
        "prediction": prediction,
        "out_of_range": flagged,
    }
    logger.info(json.dumps(record))
