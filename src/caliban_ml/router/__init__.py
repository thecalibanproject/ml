"""Router training/eval: intent exemplars, kNN baseline, Stage-2 heads, UniRoute profiles."""

from caliban_ml.router.dataset import OOS_LABEL, IntentDataset, IntentSpec
from caliban_ml.router.knn import (
    KnnCalibrationResult,
    KnnIntentClassifier,
    OosThreshold,
    evaluate,
    fit_and_calibrate,
    select_label_thresholds,
    select_oos_threshold,
    write_knn_head_artifact,
)
from caliban_ml.router.profile import (
    ModelProfile,
    ProbeScore,
    ProfileCluster,
    RouterProfile,
    compute_profile,
    kmeans,
    load_probe_scores,
    write_profile_artifact,
)

__all__ = [
    "OOS_LABEL",
    "IntentDataset",
    "IntentSpec",
    "KnnCalibrationResult",
    "KnnIntentClassifier",
    "ModelProfile",
    "OosThreshold",
    "ProbeScore",
    "ProfileCluster",
    "RouterProfile",
    "compute_profile",
    "evaluate",
    "fit_and_calibrate",
    "kmeans",
    "load_probe_scores",
    "select_label_thresholds",
    "select_oos_threshold",
    "write_knn_head_artifact",
    "write_profile_artifact",
]
