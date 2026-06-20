"""
model.py — 5-fold CV training with LightGBM + XGBoost + CatBoost ensemble.
Includes leak-free risk score recomputation inside each fold.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
from scipy.stats import skew
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from preprocess import (
    build_features, ALL_FEATURES, CATEGORICAL_FEATURES,
    CAT_ENCODED_FEATURES, RISK_SCORE_FEATURES
)

TARGET_COL = 'resolution_duration_mins'
N_FOLDS    = 5
SEED       = 42

os.makedirs('models', exist_ok=True)
os.makedirs('data/artifacts', exist_ok=True)
os.makedirs('data/encoders', exist_ok=True)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ── LABEL ENCODE CATEGORICALS ─────────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame,
                         le_dict: dict = None,
                         fit: bool = True) -> tuple:
    df = df.copy()
    if le_dict is None:
        le_dict = {}
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = '__missing__'
        df[col] = df[col].fillna('__missing__').astype(str)
        enc_col = col + '_enc'
        if fit:
            le = LabelEncoder()
            df[enc_col] = le.fit_transform(df[col])
            le_dict[col] = le
        else:
            le = le_dict.get(col)
            if le is not None:
                # Handle unseen labels
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else '__missing__')
                if '__missing__' not in known:
                    le.classes_ = np.append(le.classes_, '__missing__')
                df[enc_col] = le.transform(df[col])
            else:
                df[enc_col] = -1
    return df, le_dict


# ── RECOMPUTE RISK SCORES INSIDE FOLD (leak-free) ────────────────────────────
def recompute_risk_scores_fold(df_train: pd.DataFrame,
                                df_val:   pd.DataFrame,
                                risk_cols: list) -> tuple:
    global_mean = df_train[TARGET_COL].mean()
    for col in risk_cols:
        base = col.replace('_risk_score', '')
        if base in df_train.columns:
            risk_map = df_train.groupby(base)[TARGET_COL].mean()
            df_train[col] = df_train[base].map(risk_map).fillna(global_mean)
            df_val[col]   = df_val[base].map(risk_map).fillna(global_mean)
        else:
            df_train[col] = global_mean
            df_val[col]   = global_mean
    return df_train, df_val


# ── TRAIN ─────────────────────────────────────────────────────────────────────
def train(df_feat: pd.DataFrame):
    """
    Full 5-fold training pipeline.
    df_feat: output of build_features(is_training=True)
    """
    # Quality filter
    valid = (df_feat[TARGET_COL].notna() &
             (df_feat[TARGET_COL] > 0) &
             (df_feat[TARGET_COL] < 1440))
    df_model = df_feat[valid].copy().reset_index(drop=True)
    print(f"[train] Modeling dataset: {df_model.shape}")

    # Encode categoricals
    df_model, le_dict = encode_categoricals(df_model, fit=True)
    joblib.dump(le_dict, 'data/encoders/le_dict.pkl')

    # Log-transform target if skewed
    sk = skew(df_model[TARGET_COL])
    print(f"[train] Target skewness: {sk:.2f}")
    use_log = sk > 1.5
    if use_log:
        print("[train] Applying log1p transform to target")
        y = np.log1p(df_model[TARGET_COL].values)
    else:
        y = df_model[TARGET_COL].values

    # Ensure all features exist
    available_features = [f for f in ALL_FEATURES if f in df_model.columns]
    print(f"[train] Using {len(available_features)} features")
    X = df_model[available_features].copy()
    X = X.fillna(-999)

    risk_cols_present = [c for c in RISK_SCORE_FEATURES if c in available_features]

    # Storage
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    lgb_models, xgb_models, cat_models = [], [], []

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n─── Fold {fold+1}/{N_FOLDS} ───")

        X_tr = X.iloc[tr_idx].copy()
        X_val= X.iloc[val_idx].copy()
        y_tr = y[tr_idx]
        y_val= y[val_idx]

        # Leak-free risk score recompute
        df_tr_slice  = df_model.iloc[tr_idx].copy()
        df_val_slice = df_model.iloc[val_idx].copy()
        df_tr_slice, df_val_slice = recompute_risk_scores_fold(
            df_tr_slice, df_val_slice, risk_cols_present)
        for c in risk_cols_present:
            X_tr[c]  = df_tr_slice[c].values
            X_val[c] = df_val_slice[c].values

        # ── LightGBM ──────────────────────────────────────────────
        lgb_params = dict(
            n_estimators=3000, learning_rate=0.02,
            num_leaves=63, max_depth=-1,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=SEED, n_jobs=-1, verbose=-1
        )
        lgb_m = lgb.LGBMRegressor(**lgb_params)
        lgb_m.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(150, verbose=False),
                              lgb.log_evaluation(500)])
        oof_lgb[val_idx] = lgb_m.predict(X_val)
        lgb_models.append(lgb_m)

        # ── XGBoost ───────────────────────────────────────────────
        xgb_m = XGBRegressor(
            n_estimators=3000, learning_rate=0.02,
            max_depth=6, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=SEED, n_jobs=-1, verbosity=0,
            early_stopping_rounds=150, eval_metric='rmse'
        )
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb_m.predict(X_val)
        xgb_models.append(xgb_m)

        # ── CatBoost ──────────────────────────────────────────────
        cat_m = CatBoostRegressor(
            iterations=3000, learning_rate=0.02,
            depth=6, l2_leaf_reg=3,
            random_seed=SEED, verbose=0,
            early_stopping_rounds=150,
            eval_metric='RMSE'
        )
        cat_m.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        oof_cat[val_idx] = cat_m.predict(X_val)
        cat_models.append(cat_m)

        # Fold metrics (original scale)
        y_val_orig = np.expm1(y_val) if use_log else y_val
        for name, oof in [('LGB', oof_lgb), ('XGB', oof_xgb), ('CAT', oof_cat)]:
            pred_orig = np.expm1(oof[val_idx]) if use_log else oof[val_idx]
            r = rmse(y_val_orig, pred_orig)
            m = mean_absolute_error(y_val_orig, pred_orig)
            r2= r2_score(y_val_orig, pred_orig)
            print(f"  {name}  RMSE={r:.2f}min  MAE={m:.2f}min  R²={r2:.4f}")

    # ── Ensemble ──────────────────────────────────────────────────
    y_orig = np.expm1(y) if use_log else y
    if use_log:
        oof_lgb_o = np.expm1(oof_lgb)
        oof_xgb_o = np.expm1(oof_xgb)
        oof_cat_o = np.expm1(oof_cat)
    else:
        oof_lgb_o, oof_xgb_o, oof_cat_o = oof_lgb, oof_xgb, oof_cat

    def neg_rmse(w):
        w = np.abs(w); w /= w.sum()
        return rmse(y_orig, w[0]*oof_lgb_o + w[1]*oof_xgb_o + w[2]*oof_cat_o)

    res = minimize(neg_rmse, [1/3,1/3,1/3], method='Nelder-Mead')
    best_w = np.abs(res.x); best_w /= best_w.sum()
    print(f"\n[ensemble] Optimal weights → LGB:{best_w[0]:.3f}  XGB:{best_w[1]:.3f}  CAT:{best_w[2]:.3f}")

    oof_blend = best_w[0]*oof_lgb_o + best_w[1]*oof_xgb_o + best_w[2]*oof_cat_o
    print(f"[ensemble] Blend  RMSE={rmse(y_orig,oof_blend):.2f}min  "
          f"MAE={mean_absolute_error(y_orig,oof_blend):.2f}min  "
          f"R²={r2_score(y_orig,oof_blend):.4f}")

    # ── Save everything ───────────────────────────────────────────
    joblib.dump(lgb_models, 'models/lgb_models.pkl')
    joblib.dump(xgb_models, 'models/xgb_models.pkl')
    joblib.dump(cat_models, 'models/cat_models.pkl')

    meta = {
        'use_log': bool(use_log),
        'ensemble_weights': best_w.tolist(),
        'feature_list': available_features,
        'n_folds': N_FOLDS
    }
    with open('data/artifacts/model_meta.json','w') as f:
        json.dump(meta, f)

    print("\n[train] All artifacts saved.")
    return lgb_models, xgb_models, cat_models, best_w, available_features, use_log


# ── INFERENCE ─────────────────────────────────────────────────────────────────
def load_artifacts():
    lgb_models  = joblib.load('models/lgb_models.pkl')
    xgb_models  = joblib.load('models/xgb_models.pkl')
    cat_models  = joblib.load('models/cat_models.pkl')
    le_dict     = joblib.load('data/encoders/le_dict.pkl')
    with open('data/artifacts/model_meta.json') as f:
        meta = json.load(f)
    with open('data/metadata/risk_maps.json') as f:
        risk_maps = json.load(f)
    return lgb_models, xgb_models, cat_models, le_dict, meta, risk_maps


# def predict_duration(event_dict: dict,
#                      lgb_models, xgb_models, cat_models,
#                      le_dict, meta, risk_maps) -> float:
#     """
#     Predict resolution duration in minutes for a single event dict.
#     """
#     from preprocess import build_features, ALL_FEATURES, CATEGORICAL_FEATURES
#     import pandas as pd
#     import numpy as np

#     row = pd.DataFrame([event_dict])
#     for col in ['start_datetime','created_date','end_datetime']:
#         if col in row.columns:
#             row[col] = pd.to_datetime(row[col], errors='coerce')

#     row = build_features(row, risk_maps=risk_maps, is_training=False)
#     row, _ = encode_categoricals(row, le_dict=le_dict, fit=False)

#     features = meta['feature_list']
#     X_in = row.reindex(columns=features, fill_value=-999).fillna(-999)

#     use_log = meta['use_log']
#     w = meta['ensemble_weights']

#     preds_lgb = np.mean([m.predict(X_in) for m in lgb_models], axis=0)
#     preds_xgb = np.mean([m.predict(X_in) for m in xgb_models], axis=0)
#     preds_cat = np.mean([m.predict(X_in) for m in cat_models], axis=0)

#     pred_raw = w[0]*preds_lgb + w[1]*preds_xgb + w[2]*preds_cat
#     pred = float(np.expm1(pred_raw[0]) if use_log else pred_raw[0])
#     return max(1.0, pred)

def predict_duration(event_dict: dict,
                     lgb_models, xgb_models, cat_models,
                     le_dict, meta, risk_maps) -> float:

    import pandas as pd
    import numpy as np
    from preprocess import CATEGORICAL_FEATURES

    row = pd.DataFrame([event_dict])

    # Parse datetimes
    for col in ['start_datetime', 'created_date', 'end_datetime']:
        if col in row.columns:
            row[col] = pd.to_datetime(row[col], errors='coerce')

    # ── TEMPORAL ──────────────────────────────────────────────────
    dt = row['start_datetime']
    row['hour']            = dt.dt.hour
    row['day_of_week']     = dt.dt.dayofweek
    row['month']           = dt.dt.month
    row['week_of_year']    = dt.dt.isocalendar().week.astype(float)
    row['quarter']         = dt.dt.quarter
    row['is_weekend']      = row['day_of_week'].isin([5, 6]).astype(int)
    row['is_peak_morning'] = row['hour'].isin([8, 9, 10]).astype(int)
    row['is_peak_evening'] = row['hour'].isin([17, 18, 19, 20]).astype(int)
    row['is_peak_hour']    = (row['is_peak_morning'] | row['is_peak_evening']).astype(int)
    row['is_night']        = row['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
    row['is_midday']       = row['hour'].isin([11, 12, 13, 14]).astype(int)
    row['hour_sin']        = np.sin(2 * np.pi * row['hour'] / 24)
    row['hour_cos']        = np.cos(2 * np.pi * row['hour'] / 24)
    row['dow_sin']         = np.sin(2 * np.pi * row['day_of_week'] / 7)
    row['dow_cos']         = np.cos(2 * np.pi * row['day_of_week'] / 7)
    row['month_sin']       = np.sin(2 * np.pi * row['month'] / 12)
    row['month_cos']       = np.cos(2 * np.pi * row['month'] / 12)

    # ── RESPONSE LAG ──────────────────────────────────────────────
    row['create_to_start_mins'] = 0.0
    row['is_reactive']          = 0

    # ── SPATIAL ───────────────────────────────────────────────────
    row['event_span_km']             = 0.0
    row['is_moving_event']           = 0
    row['resolution_displacement_km']= 0.0

    # ── VEHICLE ───────────────────────────────────────────────────
    row['truck_age_bucket'] = 0.0
    row['has_cargo']        = int(event_dict.get('has_cargo', 0))
    row['has_vehicle']      = int(event_dict.get('has_vehicle', 0))
    row['is_breakdown']     = 0
    row['road_closure_flag']= int(event_dict.get('road_closure_flag', 0))
    row['is_authenticated'] = 1

    # ── RISK SCORES from precomputed maps ─────────────────────────
    global_mean = risk_maps.get('__global_mean__', 60.0)
    risk_cols = ['zone', 'junction', 'corridor', 'event_type',
                 'event_cause', 'police_station']
    for col in risk_cols:
        val = str(event_dict.get(col, '')).strip().lower()
        score = risk_maps.get(col, {}).get(val, global_mean)
        row[f'{col}_risk_score'] = score

    # ── FREQUENCY ─────────────────────────────────────────────────
    row['zone_event_count']     = risk_maps.get('zone_count', {}).get(
        str(event_dict.get('zone', '')).lower(), 1)
    row['junction_event_count'] = 1
    row['corridor_event_count'] = 1

    # ── INTERACTION ───────────────────────────────────────────────
    high_risk_types = ['political_rally', 'rally', 'festival', 'procession',
                       'sports', 'strike', 'protest', 'march']
    et = str(event_dict.get('event_type', '')).lower()
    row['is_high_risk_type']   = int(et in high_risk_types)
    row['high_risk_x_peak']    = row['is_high_risk_type'] * row['is_peak_hour']
    row['high_risk_x_weekend'] = row['is_high_risk_type'] * row['is_weekend']

    # ── ENCODE CATEGORICALS ───────────────────────────────────────
    for col in CATEGORICAL_FEATURES:
        val = str(event_dict.get(col, '__missing__')).strip().lower()
        le  = le_dict.get(col)
        enc_col = col + '_enc'
        if le is not None:
            known = set(le.classes_)
            val = val if val in known else '__missing__'
            if '__missing__' not in known:
                import numpy as np
                le.classes_ = np.append(le.classes_, '__missing__')
            row[enc_col] = le.transform([val])[0]
        else:
            row[enc_col] = -1

    # ── PREDICT ───────────────────────────────────────────────────
    features = meta['feature_list']
    X_in     = row.reindex(columns=features, fill_value=-999).fillna(-999)

    use_log  = meta['use_log']
    w        = meta['ensemble_weights']

    preds_lgb = np.mean([m.predict(X_in) for m in lgb_models], axis=0)
    preds_xgb = np.mean([m.predict(X_in) for m in xgb_models], axis=0)
    preds_cat = np.mean([m.predict(X_in) for m in cat_models], axis=0)

    pred_raw = w[0]*preds_lgb + w[1]*preds_xgb + w[2]*preds_cat
    pred     = float(np.expm1(pred_raw[0]) if use_log else pred_raw[0])
    return max(1.0, pred)


def classify_severity(duration_mins: float) -> dict:
    if duration_mins < 30:
        return {'label':'🟢 LOW',      'color':'#388E3C','officers':2, 'barricades':0,'diversions':0,
                'action':'Standard monitoring. No special deployment needed.'}
    elif duration_mins < 90:
        return {'label':'🟡 MODERATE', 'color':'#FBC02D','officers':6, 'barricades':2,'diversions':1,
                'action':'Deploy officers at event start. Monitor every 30 mins.'}
    elif duration_mins < 180:
        return {'label':'🟠 HIGH',     'color':'#F57C00','officers':12,'barricades':5,'diversions':2,
                'action':'Pre-position 2 hrs before. Set barricades at key junctions.'}
    else:
        return {'label':'🔴 CRITICAL', 'color':'#D32F2F','officers':20,'barricades':8,'diversions':3,
                'action':'Immediate deployment. Close alternate routes. Activate emergency response.'}
