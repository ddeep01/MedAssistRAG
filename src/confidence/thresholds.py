from enum import Enum
from typing import Dict, Any, Optional
from src.utils.config import load_confidence_config


class ConfidenceLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ConfidenceThresholds:
    """
    Manages configurable thresholds and weights for confidence evaluation.
    Initial engineering thresholds (not medically validated):
    - HIGH: >= 0.80
    - MEDIUM: 0.60 <= x < 0.80
    - LOW: < 0.60
    """

    def __init__(self, config_path: Optional[str] = None):
        cfg_full = load_confidence_config(config_path)
        c_cfg = cfg_full.get("confidence", {})

        self.retrieval_weight = float(c_cfg.get("retrieval_weight", 0.5))
        self.reranker_weight = float(c_cfg.get("reranker_weight", 0.5))
        self.high_threshold = float(c_cfg.get("high_threshold", 0.80))
        self.medium_threshold = float(c_cfg.get("medium_threshold", 0.60))

        raw_rank_weights = c_cfg.get("rank_weights", [0.5, 0.3, 0.2])
        self.rank_weights = [float(w) for w in raw_rank_weights]

        # Validate weights sum to 1.0
        total_w = self.retrieval_weight + self.reranker_weight
        if abs(total_w - 1.0) > 1e-5:
            if total_w > 0:
                self.retrieval_weight /= total_w
                self.reranker_weight /= total_w
            else:
                self.retrieval_weight = 0.5
                self.reranker_weight = 0.5

    def classify(self, score: float) -> ConfidenceLevel:
        """Classifies confidence score into HIGH, MEDIUM, or LOW level."""
        if score >= self.high_threshold:
            return ConfidenceLevel.HIGH
        elif score >= self.medium_threshold:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
