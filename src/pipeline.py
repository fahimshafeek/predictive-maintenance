"""
pipeline.py — Full DICSF pipeline orchestrator.

This file chains all 9 phases in the correct execution order, managing data flow
between stages. Each phase is a function that takes the previous phase's output
and returns its own output + side effects (saved models, figures, CSVs).

USAGE:
    python -m src.pipeline              # Run the full pipeline
    python -m src.pipeline --phase 3    # Run only Phase 3 onward (loads prior phases from disk)

EXECUTION ORDER (arrows show data dependencies):
    Phase 1 (EDA + Feature Eng.)
        ↓ df with 10 features (5 raw + 5 physics)
    Phase 2 (Health Index)
        ↓ df + health_index column
    Phase 3 (Wiener Process)
        ↓ df + drift_mu, diffusion_sigma columns + saved wiener_params.json
    Phase 4 (Weibull Survival)
        ↓ df + survival_prob, hazard_rate, conditional_reliability + saved weibull_params.json
    Phase 5 (ML Classification)
        ↓ trained models (XGBoost, RF, LR, baselines) + saved model files
    Phase 6 (Conformal Prediction)
        ↓ prediction sets + coverage metrics + saved conformal_quantile.json
    Phase 7 (SHAP)
        ↓ SHAP values + figures
    Phase 8 (Calibration)
        ↓ calibrated probabilities + ECE metrics
    Phase 9 (Evaluation)
        ↓ final comparison table + all figures
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional

from src.config import CFG

warnings.filterwarnings("ignore", category=FutureWarning)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: DATA LOADING + EDA + PHYSICS FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def phase_1_eda_and_features(save: bool = True) -> pd.DataFrame:
    """
    Load the AI4I dataset, clean it, compute descriptive statistics,
    and engineer 5 physics-informed features.

    Returns:
        df: DataFrame with 10 features (5 raw + 5 physics) + targets
    """
    print("\n" + "=" * 70)
    print("PHASE 1: Exploratory Data Analysis + Feature Engineering")
    print("=" * 70)

    # ── 1.1 Load ──
    df = pd.read_csv(CFG.paths.RAW_CSV)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── 1.2 Clean ──
    df = df.drop(columns=CFG.data.DROP_COLUMNS)
    df["Type"] = df["Type"].map(CFG.data.TYPE_ENCODING)
    df = df.rename(columns=CFG.data.COLUMN_RENAME)

    # ── 1.3 Descriptive Statistics ──
    from scipy.stats import shapiro, normaltest
    print("\n── Descriptive Statistics ──")
    for col in CFG.data.RAW_FEATURES:
        s = df[col]
        print(f"  {col:20s}  mean={s.mean():.2f}  std={s.std():.2f}  "
              f"skew={s.skew():.3f}  kurt={s.kurtosis():.3f}")
        sample = s.sample(min(5000, len(s)), random_state=CFG.RANDOM_SEED)
        _, p_shapiro = shapiro(sample)
        print(f"  {'':20s}  Shapiro-Wilk p={p_shapiro:.4e} "
              f"({'Normal' if p_shapiro > 0.05 else 'Non-normal'})")

    # ── 1.4 Correlation ──
    corr_matrix = df[CFG.data.RAW_FEATURES].corr(method="pearson")
    print("\n── Pearson Correlation (key pairs) ──")
    print(f"  rpm ↔ torque: {corr_matrix.loc['rpm', 'torque']:.3f}")
    print(f"  air_temp ↔ process_temp: {corr_matrix.loc['air_temp', 'process_temp']:.3f}")

    # ── 1.5 Failure Mode Analysis ──
    print("\n── Failure Mode Frequencies ──")
    for mode in CFG.data.FAILURE_MODES:
        count = df[mode].sum()
        print(f"  {mode}: {count} ({count / len(df) * 100:.2f}%)")
    print(f"  Total failures: {df['failure'].sum()} ({df['failure'].mean() * 100:.2f}%)")

    # ── 1.6 Physics Feature Engineering ──
    df["delta_temp"] = df["process_temp"] - df["air_temp"]
    df["power"] = (2 * np.pi * df["rpm"] * df["torque"]) / 60
    df["power_wear"] = df["power"] * df["tool_wear"]
    df["torque_speed_ratio"] = df["torque"] / df["rpm"]
    df["thermal_stress"] = df["delta_temp"] * df["rpm"]
    print(f"\n✅ Engineered 5 physics features. Total columns: {len(df.columns)}")

    # ── 1.7 Save ──
    if save:
        df.to_csv(
            os.path.join(CFG.paths.PROCESSED_DATA_DIR, "features_engineered.csv"),
            index=False,
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: HEALTH INDEX CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def phase_2_health_index(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Construct the Health Index using PCA and MI methods, pick the better one.

    Returns:
        df: DataFrame with added 'health_index' column
    """
    print("\n" + "=" * 70)
    print("PHASE 2: Health Index Construction")
    print("=" * 70)

    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.metrics import roc_auc_score

    features = CFG.health_index.HI_INPUT_FEATURES
    Z = StandardScaler().fit_transform(df[features].values)

    # ── Method A: PCA ──
    pca = PCA(n_components=CFG.health_index.N_COMPONENTS)
    hi_pca = pca.fit_transform(Z).flatten()
    hi_pca = MinMaxScaler().fit_transform(hi_pca.reshape(-1, 1)).flatten()

    # Ensure positive correlation with failure
    if np.corrcoef(hi_pca, df["failure"])[0, 1] < 0:
        hi_pca = 1 - hi_pca

    print(f"PCA: PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
    print(f"PCA: Loadings = {dict(zip(features, pca.components_[0].round(3)))}")

    # ── Method B: MI-Weighted ──
    mi_weights = mutual_info_classif(
        df[features], df["failure"],
        discrete_features=False,
        random_state=CFG.RANDOM_SEED,
        n_neighbors=CFG.health_index.MI_N_NEIGHBORS,
    )
    mi_weights_norm = mi_weights / mi_weights.sum()
    hi_mi = Z @ mi_weights_norm
    hi_mi = MinMaxScaler().fit_transform(hi_mi.reshape(-1, 1)).flatten()

    if np.corrcoef(hi_mi, df["failure"])[0, 1] < 0:
        hi_mi = 1 - hi_mi

    print(f"MI:  Weights = {dict(zip(features, mi_weights_norm.round(3)))}")

    # ── Select Best ──
    auc_pca = roc_auc_score(df["failure"], hi_pca)
    auc_mi = roc_auc_score(df["failure"], hi_mi)
    print(f"\nHI (PCA) AUC-ROC: {auc_pca:.4f}")
    print(f"HI (MI)  AUC-ROC: {auc_mi:.4f}")

    if auc_mi >= auc_pca:
        df["health_index"] = hi_mi
        print("✅ Selected MI-based Health Index (higher AUC)")
    else:
        df["health_index"] = hi_pca
        print("✅ Selected PCA-based Health Index (higher AUC)")

    if save:
        df.to_csv(
            os.path.join(CFG.paths.PROCESSED_DATA_DIR, "health_index.csv"),
            index=False,
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: WIENER PROCESS DEGRADATION MODEL
# ══════════════════════════════════════════════════════════════════════════════

def phase_3_wiener_process(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Fit a Wiener process model with covariate-dependent drift and diffusion.
    Compute per-observation drift_mu and diffusion_sigma.

    Returns:
        df: DataFrame with added 'drift_mu', 'diffusion_sigma' columns
    """
    print("\n" + "=" * 70)
    print("PHASE 3: Wiener Process Degradation Modeling")
    print("=" * 70)

    from scipy.optimize import minimize
    from sklearn.preprocessing import StandardScaler

    cov_cols = CFG.wiener.COVARIATE_FEATURES
    time_col = CFG.data.TIME_VARIABLE

    # ── 3.1 Prepare degradation increments ──
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    hi_by_time = df_sorted.groupby(time_col).agg(
        {**{"health_index": "mean"}, **{c: "mean" for c in cov_cols}}
    ).reset_index()

    hi_by_time["delta_hi"] = hi_by_time["health_index"].diff()
    hi_by_time["delta_t"] = hi_by_time[time_col].diff()
    hi_by_time = hi_by_time.dropna()
    hi_by_time = hi_by_time[hi_by_time["delta_t"] > 0]

    delta_hi = hi_by_time["delta_hi"].values
    delta_t = hi_by_time["delta_t"].values

    # ── 3.2 Simple Wiener (no covariates) ──
    def neg_ll_simple(params, dhi, dt):
        mu, log_sigma = params
        sigma = np.exp(log_sigma)
        var = sigma**2 * dt
        res = dhi - mu * dt
        return 0.5 * np.sum(np.log(2 * np.pi * var) + res**2 / var)

    res_simple = minimize(neg_ll_simple, [0.001, np.log(0.01)],
                          args=(delta_hi, delta_t), method="L-BFGS-B")
    mu_simple = res_simple.x[0]
    sigma_simple = np.exp(res_simple.x[1])
    print(f"Simple Wiener: μ={mu_simple:.6f}, σ={sigma_simple:.6f}")

    # ── 3.3 Covariate-dependent Wiener ──
    cov_scaler = StandardScaler()
    X_cov_local = cov_scaler.fit_transform(hi_by_time[cov_cols].values)
    d = X_cov_local.shape[1]

    def neg_ll_full(params, dhi, dt, X):
        mu_0 = params[0]
        beta_mu = params[1:d + 1]
        log_sigma_0 = params[d + 1]
        beta_sigma = params[d + 2:2 * d + 2]

        mu = mu_0 + X @ beta_mu
        sigma = np.exp(log_sigma_0 + X @ beta_sigma)
        var = sigma**2 * dt
        res = dhi - mu * dt
        return 0.5 * np.sum(np.log(2 * np.pi * var) + res**2 / var)

    x0 = np.zeros(2 * d + 2)
    x0[0] = mu_simple
    x0[d + 1] = np.log(sigma_simple)

    res_full = minimize(neg_ll_full, x0,
                        args=(delta_hi, delta_t, X_cov_local),
                        method=CFG.wiener.OPTIMIZER_METHOD,
                        options={"maxiter": CFG.wiener.OPTIMIZER_MAXITER,
                                 "ftol": CFG.wiener.OPTIMIZER_FTOL})

    mu_0 = res_full.x[0]
    beta_mu = res_full.x[1:d + 1]
    sigma_0 = np.exp(res_full.x[d + 1])
    beta_sigma = res_full.x[d + 2:]

    print(f"Covariate Wiener: μ₀={mu_0:.6f}, σ₀={sigma_0:.6f}")
    print(f"  β_μ = {dict(zip(cov_cols, beta_mu.round(6)))}")
    print(f"  β_σ = {dict(zip(cov_cols, beta_sigma.round(6)))}")
    print(f"  Converged: {res_full.success}, Log-lik: {-res_full.fun:.2f}")

    # ── 3.4 Compute per-observation features ──
    # IMPORTANT: fit the scaler on the FULL dataset (not just grouped data)
    X_cov_all = cov_scaler.transform(df[cov_cols].values)
    df["drift_mu"] = mu_0 + X_cov_all @ beta_mu
    df["diffusion_sigma"] = sigma_0 * np.exp(X_cov_all @ beta_sigma)

    # ── 3.5 Failure threshold ──
    if CFG.wiener.FAILURE_THRESHOLD is not None:
        fail_thresh = CFG.wiener.FAILURE_THRESHOLD
    else:
        fail_thresh = df.loc[df["failure"] == 1, "health_index"].quantile(
            CFG.wiener.FAILURE_THRESHOLD_QUANTILE
        )
    print(f"  Failure threshold: {fail_thresh:.4f}")

    # ── 3.6 Save parameters ──
    if save:
        params = {
            "mu_0": float(mu_0),
            "beta_mu": beta_mu.tolist(),
            "sigma_0": float(sigma_0),
            "beta_sigma": beta_sigma.tolist(),
            "failure_threshold": float(fail_thresh),
            "covariate_cols": cov_cols,
            "scaler_mean": cov_scaler.mean_.tolist(),
            "scaler_scale": cov_scaler.scale_.tolist(),
            "log_likelihood": float(-res_full.fun),
        }
        with open(CFG.paths.PARAMS_WIENER, "w") as f:
            json.dump(params, f, indent=2)
        df.to_csv(
            os.path.join(CFG.paths.PROCESSED_DATA_DIR, "degradation_features.csv"),
            index=False,
        )

    print(f"✅ Added drift_mu and diffusion_sigma to feature set")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: WEIBULL SURVIVAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def phase_4_weibull_survival(df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Fit degradation-informed Weibull regression and compute survival features.

    Returns:
        df: DataFrame with added 'survival_prob', 'hazard_rate',
            'conditional_reliability' columns
    """
    print("\n" + "=" * 70)
    print("PHASE 4: Weibull Survival Analysis")
    print("=" * 70)

    from scipy.optimize import minimize
    from sklearn.preprocessing import StandardScaler

    feat_cols = CFG.weibull.REGRESSION_FEATURES
    time_col = CFG.data.TIME_VARIABLE

    times = df[time_col].values.astype(float)
    times = np.clip(times, CFG.data.TIME_EPSILON, None)  # Avoid log(0)
    events = df["failure"].values.astype(float)

    surv_scaler = StandardScaler()
    X_surv = surv_scaler.fit_transform(df[feat_cols].values)
    n_covariates = X_surv.shape[1]

    # ── 4.1 Simple Weibull MLE ──
    def weibull_neg_ll(params, t, e):
        beta = np.exp(params[0])
        eta = np.exp(params[1])
        ll = np.sum(
            e * (np.log(beta) - beta * np.log(eta) + (beta - 1) * np.log(t))
            - (t / eta) ** beta
        )
        return -ll

    res_simple = minimize(
        weibull_neg_ll,
        [np.log(CFG.weibull.INIT_BETA), np.log(CFG.weibull.INIT_ETA)],
        args=(times, events),
        method="L-BFGS-B",
    )
    beta_simple = np.exp(res_simple.x[0])
    eta_simple = np.exp(res_simple.x[1])
    print(f"Simple Weibull: β={beta_simple:.4f}, η={eta_simple:.4f}")

    # ── 4.2 Degradation-Informed Regression ──
    def weibull_reg_neg_ll(params, t, e, X):
        beta = np.exp(params[0])
        gamma = params[1:]  # [γ₀, γ₁, ..., γ_d]

        X_int = np.column_stack([np.ones(len(X)), X])
        log_eta = X_int @ gamma
        eta = np.exp(log_eta)

        ll = np.sum(
            e * (np.log(beta) - beta * log_eta + (beta - 1) * np.log(t))
            - (t / eta) ** beta
        )
        return -ll

    n_params = 1 + 1 + n_covariates  # log_beta + gamma_0 + gamma_vector
    x0 = np.zeros(n_params)
    x0[0] = np.log(beta_simple)
    x0[1] = np.log(eta_simple)

    res_reg = minimize(
        weibull_reg_neg_ll, x0,
        args=(times, events, X_surv),
        method=CFG.weibull.OPTIMIZER_METHOD,
        options={"maxiter": CFG.weibull.OPTIMIZER_MAXITER},
    )
    beta_reg = np.exp(res_reg.x[0])
    gamma_reg = res_reg.x[1:]
    print(f"Weibull Regression: β={beta_reg:.4f}")
    print(f"  γ = {gamma_reg.round(4).tolist()}")
    print(f"  Converged: {res_reg.success}, Log-lik: {-res_reg.fun:.2f}")

    # ── 4.3 Compute survival features ──
    X_int = np.column_stack([np.ones(len(X_surv)), X_surv])
    log_eta = X_int @ gamma_reg
    eta = np.exp(log_eta)

    # S(t) = exp(-(t/η)^β)
    df["survival_prob"] = np.exp(-(times / eta) ** beta_reg)

    # h(t) = (β/η)(t/η)^(β-1)
    df["hazard_rate"] = (beta_reg / eta) * (times / eta) ** (beta_reg - 1)

    # R(t+Δt|t) = S(t+Δt) / S(t)
    dt = CFG.weibull.RELIABILITY_DELTA_T
    S_future = np.exp(-((times + dt) / eta) ** beta_reg)
    S_now = np.clip(df["survival_prob"].values, 1e-10, None)
    df["conditional_reliability"] = S_future / S_now

    print(f"\n✅ Added survival_prob, hazard_rate, conditional_reliability")

    # ── 4.4 Lifelines verification (optional) ──
    try:
        from lifelines import WeibullAFTFitter
        lf_df = pd.DataFrame({
            "time": times, "event": events,
            **{c: X_surv[:, i] for i, c in enumerate(feat_cols)}
        })
        waft = WeibullAFTFitter()
        waft.fit(lf_df, duration_col="time", event_col="event")
        print("\n── Lifelines Verification ──")
        waft.print_summary()
    except ImportError:
        print("  (lifelines not installed — skipping verification)")

    # ── 4.5 Save ──
    if save:
        params = {
            "beta": float(beta_reg),
            "gamma": gamma_reg.tolist(),
            "feature_cols": feat_cols,
            "scaler_mean": surv_scaler.mean_.tolist(),
            "scaler_scale": surv_scaler.scale_.tolist(),
            "delta_t": dt,
            "log_likelihood": float(-res_reg.fun),
        }
        with open(CFG.paths.PARAMS_WEIBULL, "w") as f:
            json.dump(params, f, indent=2)
        df.to_csv(
            os.path.join(CFG.paths.PROCESSED_DATA_DIR, "survival_features.csv"),
            index=False,
        )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5: ML CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def phase_5_ml_classification(
    df: pd.DataFrame,
    save: bool = True,
) -> Dict[str, Any]:
    """
    Train XGBoost (DICSF), Random Forest, Logistic Regression, and baselines.

    Returns:
        dict with keys:
            'models': {name: trained_model}
            'X_train', 'X_cal', 'X_test': numpy arrays
            'y_train', 'y_cal', 'y_test': numpy arrays
            'train_idx', 'cal_idx', 'test_idx': index arrays
    """
    print("\n" + "=" * 70)
    print("PHASE 5: Machine Learning Classification")
    print("=" * 70)

    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
    from sklearn.metrics import roc_auc_score, average_precision_score
    import joblib

    features = CFG.data.ALL_FEATURES
    X = df[features].values.astype(np.float32)  # float32 saves GPU memory
    y = df[CFG.data.BINARY_TARGET].values

    # ── 5.1 Train / Cal / Test Split ──
    X_temp, X_test, y_temp, y_test, idx_temp, idx_test = train_test_split(
        X, y, np.arange(len(y)),
        test_size=CFG.split.TEST_RATIO,
        random_state=CFG.split.RANDOM_SEED,
        stratify=y if CFG.split.STRATIFY else None,
    )
    cal_ratio_of_temp = CFG.split.CAL_RATIO / (CFG.split.TRAIN_RATIO + CFG.split.CAL_RATIO)
    X_train, X_cal, y_train, y_cal, idx_train, idx_cal = train_test_split(
        X_temp, y_temp, idx_temp,
        test_size=cal_ratio_of_temp,
        random_state=CFG.split.RANDOM_SEED,
        stratify=y_temp if CFG.split.STRATIFY else None,
    )

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    spw = n_neg / max(n_pos, 1)
    print(f"Split sizes: train={len(y_train)}, cal={len(y_cal)}, test={len(y_test)}")
    print(f"Failure rate: train={y_train.mean():.3f}, cal={y_cal.mean():.3f}, test={y_test.mean():.3f}")
    print(f"scale_pos_weight: {spw:.1f}")

    models = {}

    # ── 5.2 XGBoost DICSF (all 16 features) ──
    print("\n── Training XGBoost DICSF (16 features, GPU) ──")
    xgb_params = CFG.xgboost.to_dict()
    xgb_params["scale_pos_weight"] = spw
    model_xgb = xgb.XGBClassifier(**xgb_params)
    try:
        model_xgb.fit(
            X_train, y_train,
            eval_set=[(X_cal, y_cal)],
            verbose=50,
        )
    except Exception as e:
        print(f"  GPU failed ({e}), falling back to CPU...")
        xgb_params["device"] = "cpu"
        model_xgb = xgb.XGBClassifier(**xgb_params)
        model_xgb.fit(X_train, y_train, eval_set=[(X_cal, y_cal)], verbose=50)

    models["DICSF (Ours)"] = model_xgb

    # ── 5.3 XGBoost raw features only (ablation) ──
    print("\n── Training XGBoost (raw features only) ──")
    n_raw = len(CFG.data.RAW_FEATURES)
    xgb_raw_params = xgb_params.copy()
    model_xgb_raw = xgb.XGBClassifier(**xgb_raw_params)
    model_xgb_raw.fit(
        X_train[:, :n_raw], y_train,
        eval_set=[(X_cal[:, :n_raw], y_cal)],
        verbose=0,
    )
    models["XGBoost (raw)"] = model_xgb_raw

    # ── 5.4 Random Forest ──
    print("\n── Training Random Forest ──")
    rf_cfg = CFG.random_forest
    model_rf = RandomForestClassifier(
        n_estimators=rf_cfg.n_estimators, max_depth=rf_cfg.max_depth,
        min_samples_split=rf_cfg.min_samples_split,
        min_samples_leaf=rf_cfg.min_samples_leaf,
        class_weight=rf_cfg.class_weight, max_features=rf_cfg.max_features,
        oob_score=rf_cfg.oob_score, random_state=rf_cfg.random_state,
        n_jobs=rf_cfg.n_jobs,
    )
    model_rf.fit(X_train[:, :n_raw], y_train)
    print(f"  OOB Score: {model_rf.oob_score_:.4f}")
    models["Random Forest (raw)"] = model_rf

    # ── 5.5 Logistic Regression ──
    print("\n── Training Logistic Regression ──")
    model_lr = LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="lbfgs",
        random_state=CFG.RANDOM_SEED,
    )
    model_lr.fit(X_train, y_train)
    models["Logistic Regression"] = model_lr

    # ── 5.6 XGBoost + SMOTE ──
    print("\n── Training XGBoost + SMOTE ──")
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=CFG.RANDOM_SEED)
        X_sm, y_sm = smote.fit_resample(X_train, y_train)
        smote_params = {k: v for k, v in xgb_params.items() if k != "scale_pos_weight"}
        model_smote = xgb.XGBClassifier(**smote_params)
        model_smote.fit(X_sm, y_sm, eval_set=[(X_cal, y_cal)], verbose=0)
        models["XGBoost + SMOTE"] = model_smote
    except ImportError:
        print("  (imbalanced-learn not installed — skipping SMOTE baseline)")

    # ── 5.7 Save models ──
    if save:
        model_xgb.save_model(CFG.paths.MODEL_XGB_BINARY)
        joblib.dump(model_rf, CFG.paths.MODEL_RF)
        print(f"\n✅ Trained {len(models)} models")

    return {
        "models": models,
        "X_train": X_train, "X_cal": X_cal, "X_test": X_test,
        "y_train": y_train, "y_cal": y_cal, "y_test": y_test,
        "idx_train": idx_train, "idx_cal": idx_cal, "idx_test": idx_test,
        "n_raw_features": n_raw,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6: CONFORMAL PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def phase_6_conformal_prediction(
    ml_results: Dict[str, Any],
    save: bool = True,
) -> Dict[str, Any]:
    """
    Implement Split Conformal Prediction and Adaptive Conformal Inference.

    Returns:
        dict with prediction_sets, coverage, quantile, ACI results
    """
    print("\n" + "=" * 70)
    print("PHASE 6: Conformal Prediction")
    print("=" * 70)

    model = ml_results["models"]["DICSF (Ours)"]
    X_cal = ml_results["X_cal"]
    y_cal = ml_results["y_cal"]
    X_test = ml_results["X_test"]
    y_test = ml_results["y_test"]
    alpha = CFG.conformal.ALPHA

    # ── 6.1 Split Conformal Prediction ──
    probs_cal = model.predict_proba(X_cal)[:, 1]
    scores = np.abs(y_cal - probs_cal)

    n_cal = len(y_cal)
    q_level = min(np.ceil((1 - alpha) * (n_cal + 1)) / n_cal, 1.0)
    q_hat = np.quantile(scores, q_level)
    print(f"Calibrated quantile q̂ = {q_hat:.4f} (α={alpha})")

    probs_test = model.predict_proba(X_test)[:, 1]
    prediction_sets = []
    for p in probs_test:
        ps = set()
        if abs(0 - p) <= q_hat:
            ps.add(0)
        if abs(1 - p) <= q_hat:
            ps.add(1)
        if len(ps) == 0:
            ps.add(1 if p >= 0.5 else 0)
        prediction_sets.append(ps)

    # ── 6.2 Evaluate ──
    coverage = np.mean([y_test[i] in prediction_sets[i] for i in range(len(y_test))])
    avg_size = np.mean([len(ps) for ps in prediction_sets])
    n_certain_0 = sum(1 for ps in prediction_sets if ps == {0})
    n_certain_1 = sum(1 for ps in prediction_sets if ps == {1})
    n_uncertain = sum(1 for ps in prediction_sets if ps == {0, 1})
    n_test = len(y_test)

    print(f"\n── Split CP Results ──")
    print(f"  Empirical Coverage:  {coverage:.4f}  (target ≥ {1 - alpha:.2f})")
    print(f"  Average Set Size:    {avg_size:.4f}  (ideal = 1.0)")
    print(f"  Confident {{0}}:      {n_certain_0} ({n_certain_0 / n_test:.1%})")
    print(f"  Confident {{1}}:      {n_certain_1} ({n_certain_1 / n_test:.1%})")
    print(f"  Uncertain {{0,1}}:    {n_uncertain} ({n_uncertain / n_test:.1%})")

    # ── 6.3 Adaptive Conformal Inference ──
    gamma_aci = CFG.conformal.ACI_GAMMA
    q_aci = 0.5
    aci_coverages, aci_sizes, aci_quantiles = [], [], []
    running_covered = 0

    for t in range(n_test):
        p_t = probs_test[t]
        ps = set()
        if abs(0 - p_t) <= q_aci:
            ps.add(0)
        if abs(1 - p_t) <= q_aci:
            ps.add(1)
        if len(ps) == 0:
            ps.add(1 if p_t >= 0.5 else 0)

        is_covered = y_test[t] in ps
        running_covered += is_covered
        q_aci = max(q_aci + gamma_aci * (alpha - (0 if is_covered else 1)), 0.0)

        aci_coverages.append(running_covered / (t + 1))
        aci_sizes.append(len(ps))
        aci_quantiles.append(q_aci)

    print(f"\n── ACI Results ──")
    print(f"  Final Coverage: {aci_coverages[-1]:.4f}")
    print(f"  Final Quantile: {aci_quantiles[-1]:.4f}")

    # ── 6.4 Multi-alpha analysis ──
    print(f"\n── Coverage at Multiple α Levels ──")
    for a in CFG.conformal.ALPHA_LEVELS:
        q = np.quantile(scores, min(np.ceil((1 - a) * (n_cal + 1)) / n_cal, 1.0))
        ps_a = []
        for p in probs_test:
            s = set()
            if abs(0 - p) <= q:
                s.add(0)
            if abs(1 - p) <= q:
                s.add(1)
            if len(s) == 0:
                s.add(1 if p >= 0.5 else 0)
            ps_a.append(s)
        cov = np.mean([y_test[i] in ps_a[i] for i in range(n_test)])
        print(f"  α={a:.2f}: coverage={cov:.4f} (target ≥ {1 - a:.2f})")

    # ── 6.5 Save ──
    if save:
        params = {
            "quantile": float(q_hat),
            "alpha": alpha,
            "n_calibration": int(n_cal),
            "coverage": float(coverage),
            "avg_set_size": float(avg_size),
            "aci_gamma": gamma_aci,
            "aci_final_quantile": float(aci_quantiles[-1]),
            "aci_final_coverage": float(aci_coverages[-1]),
        }
        with open(CFG.paths.PARAMS_CONFORMAL, "w") as f:
            json.dump(params, f, indent=2)

    print(f"\n✅ Conformal prediction complete")

    return {
        "prediction_sets": prediction_sets,
        "quantile": q_hat,
        "coverage": coverage,
        "aci_coverages": aci_coverages,
        "aci_quantiles": aci_quantiles,
        "aci_sizes": aci_sizes,
        "cal_scores": scores,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 7: SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def phase_7_shap(ml_results: Dict[str, Any], save: bool = True) -> np.ndarray:
    """Compute and visualize SHAP values for the DICSF model."""
    print("\n" + "=" * 70)
    print("PHASE 7: SHAP Explainability")
    print("=" * 70)

    import shap
    import matplotlib.pyplot as plt

    model = ml_results["models"]["DICSF (Ours)"]
    X_test = ml_results["X_test"]
    y_test = ml_results["y_test"]
    feature_names = CFG.data.ALL_FEATURES

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print(f"SHAP values shape: {shap_values.shape}")
    print(f"Base value: {explainer.expected_value:.4f}")

    # ── Plots ──
    if save:
        fig_dir = CFG.paths.FIG_SHAP
        dpi = CFG.shap.DPI

        # 1. Summary (beeswarm)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "summary_plot.png"), dpi=dpi, bbox_inches="tight")
        plt.close()

        # 2. Bar (mean |SHAP|)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, feature_names=feature_names,
                          plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "bar_plot.png"), dpi=dpi, bbox_inches="tight")
        plt.close()

        # 3. Force plot for failure instances
        failure_idx = np.where(y_test == 1)[0]
        for i, idx in enumerate(failure_idx[:CFG.shap.N_FORCE_PLOTS]):
            shap.force_plot(
                explainer.expected_value, shap_values[idx], X_test[idx],
                feature_names=feature_names, matplotlib=True, show=False,
            )
            plt.savefig(os.path.join(fig_dir, f"force_plot_failure_{i}.png"),
                        dpi=dpi, bbox_inches="tight")
            plt.close()

        # 4. Dependence plots for top features
        top_feat_idx = np.argsort(np.abs(shap_values).mean(axis=0))[::-1]
        for rank, fi in enumerate(top_feat_idx[:CFG.shap.N_TOP_FEATURES]):
            shap.dependence_plot(fi, shap_values, X_test,
                                 feature_names=feature_names, show=False)
            plt.savefig(os.path.join(fig_dir, f"dependence_{feature_names[fi]}.png"),
                        dpi=dpi, bbox_inches="tight")
            plt.close()

        # ── Report which feature GROUPS dominate ──
        raw_shap = np.abs(shap_values[:, :5]).mean()
        phys_shap = np.abs(shap_values[:, 5:10]).mean()
        deg_shap = np.abs(shap_values[:, 10:13]).mean()
        surv_shap = np.abs(shap_values[:, 13:16]).mean()
        print(f"\n── Mean |SHAP| by Feature Group ──")
        print(f"  Raw sensors:     {raw_shap:.4f}")
        print(f"  Physics:         {phys_shap:.4f}")
        print(f"  Degradation:     {deg_shap:.4f}")
        print(f"  Survival:        {surv_shap:.4f}")

    print(f"✅ SHAP analysis complete, {4 + CFG.shap.N_FORCE_PLOTS} plots saved")
    return shap_values


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 8: PROBABILITY CALIBRATION
# ══════════════════════════════════════════════════════════════════════════════

def phase_8_calibration(ml_results: Dict[str, Any], save: bool = True) -> Dict[str, float]:
    """Apply Platt scaling and isotonic regression, compute ECE."""
    print("\n" + "=" * 70)
    print("PHASE 8: Probability Calibration")
    print("=" * 70)

    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    model = ml_results["models"]["DICSF (Ours)"]
    X_cal, y_cal = ml_results["X_cal"], ml_results["y_cal"]
    X_test, y_test = ml_results["X_test"], ml_results["y_test"]

    raw_cal = model.predict_proba(X_cal)[:, 1]
    raw_test = model.predict_proba(X_test)[:, 1]

    # ── Platt Scaling ──
    platt = LogisticRegression(
        solver=CFG.calibration.PLATT_SOLVER,
        max_iter=CFG.calibration.PLATT_MAX_ITER,
    )
    platt.fit(raw_cal.reshape(-1, 1), y_cal)
    cal_platt = platt.predict_proba(raw_test.reshape(-1, 1))[:, 1]

    # ── Isotonic Regression ──
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal, y_cal)
    cal_iso = iso.predict(raw_test)

    # ── ECE ──
    def compute_ece(y_true, y_prob, n_bins):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
            n_bin = mask.sum()
            if n_bin == 0:
                continue
            ece += (n_bin / len(y_true)) * abs(y_true[mask].mean() - y_prob[mask].mean())
        return ece

    n_bins = CFG.calibration.N_BINS
    ece_raw = compute_ece(y_test, raw_test, n_bins)
    ece_platt = compute_ece(y_test, cal_platt, n_bins)
    ece_iso = compute_ece(y_test, cal_iso, n_bins)

    print(f"ECE (Raw):      {ece_raw:.4f}")
    print(f"ECE (Platt):    {ece_platt:.4f}")
    print(f"ECE (Isotonic): {ece_iso:.4f}")

    # ── Reliability Diagram ──
    if save:
        plt.figure(figsize=(8, 6))
        for label, probs in [("Raw", raw_test), ("Platt", cal_platt), ("Isotonic", cal_iso)]:
            frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=n_bins)
            plt.plot(mean_pred, frac_pos, marker="o", label=f"{label} (ECE={compute_ece(y_test, probs, n_bins):.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Perfect")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Reliability Diagram")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(CFG.paths.FIG_CLASSIFICATION, "reliability_diagram.png"),
                    dpi=CFG.viz.DPI, bbox_inches="tight")
        plt.close()

    print(f"✅ Calibration complete")
    return {"ece_raw": ece_raw, "ece_platt": ece_platt, "ece_iso": ece_iso}


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 9: FULL EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def phase_9_evaluation(
    ml_results: Dict[str, Any],
    conformal_results: Dict[str, Any],
    calibration_results: Dict[str, float],
    save: bool = True,
) -> pd.DataFrame:
    """Evaluate all models, produce comparison table and classification plots."""
    print("\n" + "=" * 70)
    print("PHASE 9: Full Evaluation")
    print("=" * 70)

    from sklearn.metrics import (
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
        confusion_matrix, classification_report,
        roc_curve, precision_recall_curve,
    )
    import matplotlib.pyplot as plt

    models = ml_results["models"]
    X_test = ml_results["X_test"]
    y_test = ml_results["y_test"]
    n_raw = ml_results["n_raw_features"]

    results = []
    fig_dir = CFG.paths.FIG_CLASSIFICATION

    # ── Evaluate each model ──
    plt.figure(figsize=(10, 6))  # ROC curves
    fig_pr, ax_pr = plt.subplots(figsize=(10, 6))  # PR curves

    for name, model in models.items():
        # Select features based on model type
        if "(raw)" in name:
            X = X_test[:, :n_raw]
        else:
            X = X_test

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        metrics = {
            "Model": name,
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "AUC-ROC": roc_auc_score(y_test, y_prob),
            "PR-AUC": average_precision_score(y_test, y_prob),
        }
        results.append(metrics)

        print(f"\n{'─' * 50}")
        print(f"  {name}")
        print(f"{'─' * 50}")
        for k, v in metrics.items():
            if k != "Model":
                print(f"    {k:12s}: {v:.4f}")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        color = CFG.viz.MODEL_COLORS.get(name, "#000000")
        lw = 3 if "DICSF" in name else 1.5
        plt.plot(fpr, tpr, label=f"{name} (AUC={metrics['AUC-ROC']:.3f})",
                 color=color, linewidth=lw)

        # PR curve
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_prob)
        ax_pr.plot(rec_arr, prec_arr, label=f"{name} (AP={metrics['PR-AUC']:.3f})",
                   color=color, linewidth=lw)

    # Save ROC plot
    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(fig_dir, "roc_curves_all_models.png"),
                    dpi=CFG.viz.DPI, bbox_inches="tight")
    plt.close()

    # Save PR plot
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curves — All Models")
    ax_pr.legend(loc="upper right")
    fig_pr.tight_layout()
    if save:
        fig_pr.savefig(os.path.join(fig_dir, "pr_curves_all_models.png"),
                       dpi=CFG.viz.DPI, bbox_inches="tight")
    plt.close(fig_pr)

    # ── Confusion matrix for DICSF ──
    if save and "DICSF (Ours)" in models:
        from sklearn.metrics import ConfusionMatrixDisplay
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_estimator(
            models["DICSF (Ours)"], X_test, y_test,
            display_labels=["No Failure", "Failure"],
            cmap="Blues", ax=ax_cm,
        )
        ax_cm.set_title("DICSF Confusion Matrix")
        fig_cm.tight_layout()
        fig_cm.savefig(os.path.join(fig_dir, "confusion_matrix_dicsf.png"),
                       dpi=CFG.viz.DPI, bbox_inches="tight")
        plt.close(fig_cm)

    # ── Results table ──
    results_df = pd.DataFrame(results)
    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON TABLE")
    print(f"{'=' * 70}")
    print(results_df.to_string(index=False, float_format="%.4f"))

    if save:
        results_df.to_csv(CFG.paths.METRICS_CSV, index=False)
        print(f"\n✅ Results saved to {CFG.paths.METRICS_CSV}")

    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(start_phase: int = 1):
    """
    Run the full DICSF pipeline from the specified starting phase.

    Args:
        start_phase: Phase number to start from (1-9). Phases before this
                     will be loaded from saved intermediate files on disk.
    """
    print("╔" + "═" * 68 + "╗")
    print("║   DICSF — Degradation-Informed Conformal Survival Framework      ║")
    print("║   Full Pipeline Execution                                         ║")
    print("╚" + "═" * 68 + "╝")

    # Create output directories
    CFG.paths.create_all()

    # ── Phase 1 ──
    if start_phase <= 1:
        df = phase_1_eda_and_features()
    else:
        df = pd.read_csv(os.path.join(CFG.paths.PROCESSED_DATA_DIR, "features_engineered.csv"))
        print(f"[Loaded Phase 1 output: {len(df)} rows]")

    # ── Phase 2 ──
    if start_phase <= 2:
        df = phase_2_health_index(df)
    else:
        df = pd.read_csv(os.path.join(CFG.paths.PROCESSED_DATA_DIR, "health_index.csv"))
        print(f"[Loaded Phase 2 output]")

    # ── Phase 3 ──
    if start_phase <= 3:
        df = phase_3_wiener_process(df)
    else:
        df = pd.read_csv(os.path.join(CFG.paths.PROCESSED_DATA_DIR, "degradation_features.csv"))
        print(f"[Loaded Phase 3 output]")

    # ── Phase 4 ──
    if start_phase <= 4:
        df = phase_4_weibull_survival(df)
    else:
        df = pd.read_csv(os.path.join(CFG.paths.PROCESSED_DATA_DIR, "survival_features.csv"))
        print(f"[Loaded Phase 4 output]")

    # ── Phase 5 ──
    ml_results = phase_5_ml_classification(df)

    # ── Phase 6 ──
    conformal_results = phase_6_conformal_prediction(ml_results)

    # ── Phase 7 ──
    shap_values = phase_7_shap(ml_results)

    # ── Phase 8 ──
    calibration_results = phase_8_calibration(ml_results)

    # ── Phase 9 ──
    results_df = phase_9_evaluation(ml_results, conformal_results, calibration_results)

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║   ✅  PIPELINE COMPLETE                                           ║")
    print("╚" + "═" * 68 + "╝")

    return results_df


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DICSF Pipeline")
    parser.add_argument("--phase", type=int, default=1,
                        help="Start from this phase (1-9). Earlier phases loaded from disk.")
    args = parser.parse_args()

    run_pipeline(start_phase=args.phase)
