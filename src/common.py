"""Shared constants and optional dependencies for the SSV Python workflow."""

from pathlib import Path
import math
import time

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False

ACTIVE_CLASSIFIER_METHODS = ("knn", "lasso", "rf", "svm", "xgb")

DEFAULT_CLASSIFIER_COLORS = {
    "knn": "#1f77b4",
    "lasso": "#d62728",
    "rf": "#2ca02c",
    "svm": "#8c564b",
    "xgb": "#ff7f0e",
}

DEFAULT_NORMALIZATION_LINESTYLES = {
    "non": "-",
    "qn": ":",
    "mn": "-.",
    "vsn": "--",
}

DEFAULT_ERROR_CURVE_LINEWIDTH = 1.2

PRO_DIRECTIONS = ("batch_1_to_batch_2", "batch_2_to_batch_1")
PRO_DIRECTION_BATCHES = {
    "batch_1_to_batch_2": ("batch_1", "batch_2"),
    "batch_2_to_batch_1": ("batch_2", "batch_1"),
}
PRO_METRICS = ("raw_error", "balanced_error")
