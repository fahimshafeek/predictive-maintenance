"""
config.py — Single source of truth for ALL constants, hyperparameters, paths, and feature lists.

Every tunable value in the entire DICSF pipeline is defined here. No magic numbers should exist
in any other file. If you need to change a hyperparameter, change it HERE and nowhere else.

This file is designed to be imported by every other module:
    from src.config import CFG
    print(CFG.RANDOM_SEED)
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional


# ══════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Auto-detect project root (2 levels up from src/config.py)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass(frozen=True)
class PathConfig:
    """All filesystem paths used by the project."""

    PROJECT_ROOT: str = _PROJECT_ROOT

    # Data paths
    RAW_DATA_DIR: str = os.path.join(_PROJECT_ROOT, "data", "raw")
    PROCESSED_DATA_DIR: str = os.path.join(_PROJECT_ROOT, "data", "processed")
    SPLITS_DIR: str = os.path.join(_PROJECT_ROOT, "data", "splits")
    RAW_CSV: str = os.path.join(_PROJECT_ROOT, "data", "raw", "ai4i2020.csv")

    # Output paths
    MODELS_DIR: str = os.path.join(_PROJECT_ROOT, "models")
    FIGURES_DIR: str = os.path.join(_PROJECT_ROOT, "figures")
    RESULTS_DIR: str = os.path.join(_PROJECT_ROOT, "results")

    # Figure subdirectories
    FIG_EDA: str = os.path.join(_PROJECT_ROOT, "figures", "eda")
    FIG_DEGRADATION: str = os.path.join(_PROJECT_ROOT, "figures", "degradation")
    FIG_SURVIVAL: str = os.path.join(_PROJECT_ROOT, "figures", "survival")
    FIG_CLASSIFICATION: str = os.path.join(_PROJECT_ROOT, "figures", "classification")
    FIG_CONFORMAL: str = os.path.join(_PROJECT_ROOT, "figures", "conformal")
    FIG_SHAP: str = os.path.join(_PROJECT_ROOT, "figures", "shap")

    # Saved model filenames
    MODEL_XGB_BINARY: str = os.path.join(_PROJECT_ROOT, "models", "xgboost_binary.json")
    MODEL_XGB_MULTICLASS: str = os.path.join(_PROJECT_ROOT, "models", "xgboost_multiclass.json")
    MODEL_RF: str = os.path.join(_PROJECT_ROOT, "models", "random_forest.pkl")
    PARAMS_WIENER: str = os.path.join(_PROJECT_ROOT, "models", "wiener_params.json")
    PARAMS_WEIBULL: str = os.path.join(_PROJECT_ROOT, "models", "weibull_params.json")
    PARAMS_CONFORMAL: str = os.path.join(_PROJECT_ROOT, "models", "conformal_quantile.json")

    # Results
    METRICS_CSV: str = os.path.join(_PROJECT_ROOT, "results", "metrics_comparison.csv")

    def create_all(self):
        """Create all output directories if they don't exist."""
        for d in [
            self.RAW_DATA_DIR, self.PROCESSED_DATA_DIR, self.SPLITS_DIR,
            self.MODELS_DIR, self.RESULTS_DIR,
            self.FIG_EDA, self.FIG_DEGRADATION, self.FIG_SURVIVAL,
            self.FIG_CLASSIFICATION, self.FIG_CONFORMAL, self.FIG_SHAP,
        ]:
            os.makedirs(d, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DataConfig:
    """Dataset schema, column names, and download URL."""

    # Download URL (UCI Machine Learning Repository)
    DOWNLOAD_URL: str = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"

    # Column renaming map (original → clean)
    COLUMN_RENAME: Dict[str, str] = field(default_factory=lambda: {
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rpm",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
        "Machine failure": "failure",
    })

    # Columns to drop (non-predictive identifiers)
    DROP_COLUMNS: List[str] = field(default_factory=lambda: ["UDI", "Product ID"])

    # Product type encoding: L=0 (Low, 60%), M=1 (Medium, 30%), H=2 (High, 10%)
    TYPE_ENCODING: Dict[str, int] = field(default_factory=lambda: {"L": 0, "M": 1, "H": 2})

    # ── Feature Groups ──────────────────────────────────────────────────────
    # These lists define EXACTLY which columns go into each category.
    # The augmented feature vector is the concatenation of all groups.

    RAW_FEATURES: List[str] = field(default_factory=lambda: [
        "air_temp", "process_temp", "rpm", "torque", "tool_wear"
    ])

    PHYSICS_FEATURES: List[str] = field(default_factory=lambda: [
        "delta_temp", "power", "power_wear", "torque_speed_ratio", "thermal_stress"
    ])

    DEGRADATION_FEATURES: List[str] = field(default_factory=lambda: [
        "health_index", "drift_mu", "diffusion_sigma"
    ])

    SURVIVAL_FEATURES: List[str] = field(default_factory=lambda: [
        "survival_prob", "hazard_rate", "conditional_reliability"
    ])

    # All 16 features in the augmented vector, IN ORDER
    ALL_FEATURES: List[str] = field(default_factory=lambda: [
        # Raw (5)
        "air_temp", "process_temp", "rpm", "torque", "tool_wear",
        # Physics (5)
        "delta_temp", "power", "power_wear", "torque_speed_ratio", "thermal_stress",
        # Degradation (3)
        "health_index", "drift_mu", "diffusion_sigma",
        # Survival (3)
        "survival_prob", "hazard_rate", "conditional_reliability",
    ])

    # Target columns
    BINARY_TARGET: str = "failure"
    FAILURE_MODES: List[str] = field(default_factory=lambda: [
        "TWF", "HDF", "PWF", "OSF", "RNF"
    ])

    # Pseudo-time variable for degradation/survival modeling
    TIME_VARIABLE: str = "tool_wear"

    # Minimum time value (avoid log(0) in Weibull; add epsilon if tool_wear == 0)
    TIME_EPSILON: float = 0.1


# ══════════════════════════════════════════════════════════════════════════════
# DATA SPLITTING
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SplitConfig:
    """Train / Calibration / Test split ratios."""

    RANDOM_SEED: int = 42

    # Split ratios (must sum to 1.0)
    TRAIN_RATIO: float = 0.60   # 6,000 samples
    CAL_RATIO: float = 0.20     # 2,000 samples (for conformal prediction)
    TEST_RATIO: float = 0.20    # 2,000 samples

    # Stratify splits by the target variable to preserve class balance
    STRATIFY: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — HEALTH INDEX CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class HealthIndexConfig:
    """Settings for PCA-based and MI-based Health Index construction."""

    # Features used to construct the Health Index
    # (all raw + physics features, before degradation/survival are computed)
    HI_INPUT_FEATURES: List[str] = field(default_factory=lambda: [
        "air_temp", "process_temp", "rpm", "torque", "tool_wear",
        "delta_temp", "power", "power_wear", "torque_speed_ratio", "thermal_stress",
    ])

    # PCA
    N_COMPONENTS: int = 1  # Only first principal component

    # Mutual Information
    MI_N_NEIGHBORS: int = 5  # k for KSG estimator in mutual_info_classif

    # Output range: HI is normalized to [0, 1] where 0 = healthy, 1 = degraded
    # If correlation with failure is negative, the HI is auto-flipped.


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — WIENER PROCESS DEGRADATION MODEL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WienerConfig:
    """Wiener process model settings."""

    # Covariates used in the covariate-dependent drift and diffusion
    COVARIATE_FEATURES: List[str] = field(default_factory=lambda: [
        "air_temp", "process_temp", "rpm", "torque"
    ])

    # Failure threshold for First Hitting Time
    # Set as the median Health Index among failed observations.
    # If None, computed dynamically during fitting.
    FAILURE_THRESHOLD: Optional[float] = None
    FAILURE_THRESHOLD_QUANTILE: float = 0.50  # Used if FAILURE_THRESHOLD is None

    # MLE optimizer settings
    OPTIMIZER_METHOD: str = "L-BFGS-B"
    OPTIMIZER_MAXITER: int = 1000
    OPTIMIZER_FTOL: float = 1e-12

    # RUL clipping (cap extreme values to prevent inf in downstream features)
    RUL_CLIP_MAX: float = 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — WEIBULL SURVIVAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WeibullConfig:
    """Weibull regression model settings."""

    # Covariates for the degradation-informed Weibull regression
    # First column MUST be health_index (the bridge from Wiener → Weibull)
    REGRESSION_FEATURES: List[str] = field(default_factory=lambda: [
        "health_index", "air_temp", "process_temp", "rpm", "torque"
    ])

    # Initial parameter guesses for MLE
    INIT_BETA: float = 2.0    # Shape > 1 assumes wear-out degradation
    INIT_ETA: float = 100.0   # Scale — will be optimized

    # Optimizer
    OPTIMIZER_METHOD: str = "L-BFGS-B"
    OPTIMIZER_MAXITER: int = 2000

    # Conditional reliability look-ahead (in tool_wear units)
    RELIABILITY_DELTA_T: float = 10.0

    # Maintenance threshold: recommend maintenance when R < R_MIN
    R_MIN: float = 0.80


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — MACHINE LEARNING CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class XGBoostConfig:
    """XGBoost hyperparameters, calibrated for RTX 4050 (6GB VRAM)."""

    # ── Tree structure ──
    max_depth: int = 6
    min_child_weight: int = 5
    gamma: float = 0.1

    # ── Learning ──
    learning_rate: float = 0.05
    n_estimators: int = 500          # Will use early stopping
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # ── Regularization ──
    reg_alpha: float = 0.1           # L1
    reg_lambda: float = 1.0          # L2

    # ── GPU (RTX 4050) ──
    # XGBoost 2.0+ syntax: gpu_hist is DEPRECATED
    tree_method: str = "hist"
    device: str = "cuda"             # Change to "cpu" if no GPU

    # ── Objective ──
    objective: str = "binary:logistic"
    eval_metric: List[str] = field(default_factory=lambda: ["logloss", "aucpr", "auc"])

    # ── Reproducibility ──
    random_state: int = 42
    verbosity: int = 1

    # ── Early stopping ──
    early_stopping_rounds: int = 50

    def to_dict(self) -> dict:
        """Convert to dict for XGBClassifier(**params). Handles list fields."""
        d = {}
        for k, v in self.__dict__.items():
            d[k] = v
        return d


@dataclass(frozen=True)
class RandomForestConfig:
    """Random Forest baseline hyperparameters."""

    n_estimators: int = 500
    max_depth: int = 8
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    class_weight: str = "balanced"
    max_features: str = "sqrt"
    oob_score: bool = True
    random_state: int = 42
    n_jobs: int = -1  # Use all CPU cores


@dataclass(frozen=True)
class HyperparamSearchConfig:
    """Hyperparameter search settings."""

    N_ITER: int = 50             # Number of random combinations
    CV_FOLDS: int = 5            # Stratified K-Fold
    SCORING: str = "average_precision"  # PR-AUC (better than ROC-AUC for imbalanced)
    RANDOM_STATE: int = 42

    # Search space (used by RandomizedSearchCV)
    PARAM_DISTRIBUTIONS: Dict = field(default_factory=lambda: {
        "max_depth": [4, 5, 6, 7, 8],
        "min_child_weight": [1, 3, 5, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "gamma": [0, 0.1, 0.3, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1.0],
        "reg_lambda": [0.5, 1.0, 2.0, 5.0],
    })


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — CONFORMAL PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ConformalConfig:
    """Conformal prediction settings."""

    # Significance level: α = 0.10 → 90% coverage guarantee
    ALPHA: float = 0.10

    # Adaptive Conformal Inference (ACI) learning rate
    ACI_GAMMA: float = 0.005

    # Multiple alpha levels to test for coverage-vs-alpha plot
    ALPHA_LEVELS: List[float] = field(default_factory=lambda: [
        0.01, 0.05, 0.10, 0.15, 0.20
    ])


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SHAPConfig:
    """SHAP analysis settings."""

    # Number of top features to show in dependence plots
    N_TOP_FEATURES: int = 4

    # Number of individual force plots to generate
    N_FORCE_PLOTS: int = 3  # For 3 different failure instances

    # Plot DPI
    DPI: int = 150


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8 — PROBABILITY CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CalibrationConfig:
    """Probability calibration settings."""

    # Number of bins for ECE and reliability diagrams
    N_BINS: int = 10

    # Platt scaling
    PLATT_MAX_ITER: int = 1000
    PLATT_SOLVER: str = "lbfgs"


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VizConfig:
    """Global visualization settings."""

    DPI: int = 150
    FIGURE_SIZE: tuple = (10, 6)
    STYLE: str = "seaborn-v0_8-whitegrid"  # matplotlib style
    FONT_SIZE: int = 12
    SAVE_FORMAT: str = "png"

    # Color palette for failure vs. no-failure
    COLOR_NO_FAILURE: str = "#2196F3"   # Blue
    COLOR_FAILURE: str = "#F44336"      # Red
    COLOR_UNCERTAIN: str = "#FF9800"    # Orange

    # Model comparison colors
    MODEL_COLORS: Dict[str, str] = field(default_factory=lambda: {
        "Logistic Regression": "#9E9E9E",
        "Random Forest (raw)": "#4CAF50",
        "XGBoost (raw)": "#FF9800",
        "XGBoost + SMOTE": "#9C27B0",
        "DICSF (Ours)": "#F44336",
    })


# ══════════════════════════════════════════════════════════════════════════════
# MASTER CONFIG — Single entry point
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Config:
    """
    Master configuration object. Import this in every module:

        from src.config import CFG
        print(CFG.data.RAW_FEATURES)
        print(CFG.xgboost.learning_rate)
    """

    RANDOM_SEED: int = 42

    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    health_index: HealthIndexConfig = field(default_factory=HealthIndexConfig)
    wiener: WienerConfig = field(default_factory=WienerConfig)
    weibull: WeibullConfig = field(default_factory=WeibullConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    hyperparam_search: HyperparamSearchConfig = field(default_factory=HyperparamSearchConfig)
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    shap: SHAPConfig = field(default_factory=SHAPConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    viz: VizConfig = field(default_factory=VizConfig)


# ── Singleton instance ───────────────────────────────────────────────────────
# Import this everywhere:  from src.config import CFG
CFG = Config()


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST (run: python -m src.config)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("DICSF Configuration Self-Test")
    print("=" * 60)
    print(f"Project root:       {CFG.paths.PROJECT_ROOT}")
    print(f"Random seed:        {CFG.RANDOM_SEED}")
    print(f"Raw features:       {CFG.data.RAW_FEATURES}")
    print(f"Physics features:   {CFG.data.PHYSICS_FEATURES}")
    print(f"All features (16):  {len(CFG.data.ALL_FEATURES)} → {CFG.data.ALL_FEATURES}")
    print(f"XGBoost device:     {CFG.xgboost.device}")
    print(f"XGBoost tree_method:{CFG.xgboost.tree_method}")
    print(f"Conformal α:        {CFG.conformal.ALPHA}")
    print(f"ACI γ:              {CFG.conformal.ACI_GAMMA}")
    print(f"Weibull init β:     {CFG.weibull.INIT_BETA}")
    print(f"Failure modes:      {CFG.data.FAILURE_MODES}")
    print()
    print("Creating output directories...")
    CFG.paths.create_all()
    print("✅ All directories created.")
    print("✅ Configuration valid.")
