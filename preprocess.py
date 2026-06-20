"""
preprocess.py — Data loading, cleaning, target construction, feature engineering
Adapted to actual Astram columns.
"""

import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import json, os, warnings
warnings.filterwarnings('ignore')

DATETIME_COLS = ['start_datetime','end_datetime','resolved_datetime',
                 'closed_datetime','created_date','modified_datetime']

CATEGORICAL_FEATURES = ['event_type','event_cause','zone','junction',
                         'corridor','direction','veh_type','police_station']

DROP_COLS = ['map_file','route_path','meta_data','client_id','created_by_id',
             'last_modified_by_id','kgid','gba_identifier','description',
             'comment','citizen_accident_id','assigned_to_police_id',
             'resolved_by_id','closed_by_id','address','end_address',
             'resolved_at_address','veh_no']

# ── LOAD & CLEAN ──────────────────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"[load] Raw shape: {df.shape}")
    df.columns = df.columns.str.strip().str.lower().str.replace(' ','_')

    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    drop_existing = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_existing)

    if 'requires_road_closure' in df.columns:
        rc = df['requires_road_closure'].astype(str).str.strip().str.lower()
        df['road_closure_flag'] = rc.map({'yes':1,'true':1,'1':1,'no':0,'false':0,'0':0}).fillna(0).astype(int)

    if 'authenticated' in df.columns:
        df['is_authenticated'] = df['authenticated'].astype(str).str.lower().map(
            {'yes':1,'true':1,'1':1,'no':0,'false':0,'0':0}).fillna(0).astype(int)

    for col in ['latitude','longitude','endlatitude','endlongitude',
                'resolved_at_latitude','resolved_at_longitude','age_of_truck']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()
            df[col] = df[col].replace({'nan':np.nan,'none':np.nan,'':np.nan})

    print(f"[clean] After cleaning: {df.shape}")
    return df

# ── TARGET CONSTRUCTION ───────────────────────────────────────────────────────
def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if 'resolved_datetime' in df.columns and 'start_datetime' in df.columns:
        df['resolution_duration_mins'] = (
            df['resolved_datetime'] - df['start_datetime']
        ).dt.total_seconds() / 60

    if 'end_datetime' in df.columns and 'start_datetime' in df.columns:
        df['event_duration_mins'] = (
            df['end_datetime'] - df['start_datetime']
        ).dt.total_seconds() / 60
        df['event_duration_mins'] = df['event_duration_mins'].clip(0, 1440)

    if 'priority' in df.columns:
        pv = df['priority'].astype(str).str.strip().str.lower()
        print(f"[targets] Priority values: {pv.value_counts().to_dict()}")
        pm = {'low':0,'1':0,'p4':0,'medium':1,'2':1,'p3':1,
              'high':2,'3':2,'p2':2,'critical':3,'4':3,'p1':3,'emergency':3}
        df['priority_encoded'] = pv.map(pm)

    if 'resolution_duration_mins' in df.columns:
        valid = (df['resolution_duration_mins'].notna() &
                 (df['resolution_duration_mins'] > 0) &
                 (df['resolution_duration_mins'] < 1440))
        print(f"[targets] Valid duration rows: {valid.sum()} / {len(df)}")
        print(df.loc[valid,'resolution_duration_mins'].describe().round(2))

    return df

# ── HAVERSINE ─────────────────────────────────────────────────────────────────
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    try:
        lat1,lon1,lat2,lon2 = map(float,[lat1,lon1,lat2,lon2])
        if any(np.isnan(v) for v in [lat1,lon1,lat2,lon2]):
            return np.nan
        lat1,lon1,lat2,lon2 = map(radians,[lat1,lon1,lat2,lon2])
        dlat,dlon = lat2-lat1, lon2-lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        return R*2*atan2(sqrt(a),sqrt(1-a))
    except:
        return np.nan

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, risk_maps: dict = None,
                   is_training: bool = True) -> pd.DataFrame:
    df = df.copy()
    target_col = 'resolution_duration_mins'

    # TEMPORAL
    if 'start_datetime' in df.columns:
        dt = df['start_datetime']
        df['hour']         = dt.dt.hour
        df['day_of_week']  = dt.dt.dayofweek
        df['month']        = dt.dt.month
        df['week_of_year'] = dt.dt.isocalendar().week.astype(float)
        df['quarter']      = dt.dt.quarter
        df['is_weekend']      = df['day_of_week'].isin([5,6]).astype(int)
        df['is_peak_morning'] = df['hour'].isin([8,9,10]).astype(int)
        df['is_peak_evening'] = df['hour'].isin([17,18,19,20]).astype(int)
        df['is_peak_hour']    = (df['is_peak_morning']|df['is_peak_evening']).astype(int)
        df['is_night']        = df['hour'].isin([22,23,0,1,2,3,4,5]).astype(int)
        df['is_midday']       = df['hour'].isin([11,12,13,14]).astype(int)
        # Cyclical
        df['hour_sin']   = np.sin(2*np.pi*df['hour']/24)
        df['hour_cos']   = np.cos(2*np.pi*df['hour']/24)
        df['dow_sin']    = np.sin(2*np.pi*df['day_of_week']/7)
        df['dow_cos']    = np.cos(2*np.pi*df['day_of_week']/7)
        df['month_sin']  = np.sin(2*np.pi*df['month']/12)
        df['month_cos']  = np.cos(2*np.pi*df['month']/12)

    # Response lag
    if 'created_date' in df.columns and 'start_datetime' in df.columns:
        lag = (df['start_datetime'] - df['created_date']).dt.total_seconds() / 60
        df['create_to_start_mins'] = lag
        df['is_reactive'] = (lag < 0).astype(int)
    else:
        df['create_to_start_mins'] = 0.0
        df['is_reactive'] = 0

    # SPATIAL
    has_start = 'latitude' in df.columns and 'longitude' in df.columns
    has_end   = 'endlatitude' in df.columns and 'endlongitude' in df.columns
    has_res   = 'resolved_at_latitude' in df.columns

    if has_start and has_end:
        df['event_span_km']  = df.apply(lambda r: haversine_km(
            r['latitude'],r['longitude'],r['endlatitude'],r['endlongitude']),axis=1)
        df['is_moving_event'] = (df['event_span_km']>0.5).astype(int)
    else:
        df['event_span_km']   = 0.0
        df['is_moving_event'] = 0

    if has_start and has_res:
        df['resolution_displacement_km'] = df.apply(lambda r: haversine_km(
            r['latitude'],r['longitude'],
            r['resolved_at_latitude'],r['resolved_at_longitude']),axis=1)
    else:
        df['resolution_displacement_km'] = 0.0

    # VEHICLE
    if 'age_of_truck' in df.columns:
        df['truck_age_bucket'] = pd.cut(
            df['age_of_truck'].fillna(-1),
            bins=[-2,0,3,7,15,100], labels=[0,1,2,3,4]).astype(float)
    else:
        df['truck_age_bucket'] = 0.0

    df['has_cargo']    = df['cargo_material'].notna().astype(int) if 'cargo_material' in df.columns else 0
    df['has_vehicle']  = df['veh_type'].notna().astype(int) if 'veh_type' in df.columns else 0
    df['is_breakdown'] = df['reason_breakdown'].notna().astype(int) if 'reason_breakdown' in df.columns else 0
    if 'road_closure_flag' not in df.columns:
        df['road_closure_flag'] = 0

    # HISTORICAL RISK SCORES (leak-free — recompute inside CV folds)
    risk_cols = ['zone','junction','corridor','event_type','event_cause','police_station']
    if is_training and target_col in df.columns:
        computed_maps = {}
        global_mean = df[target_col].mean()
        computed_maps['__global_mean__'] = global_mean
        for col in risk_cols:
            if col in df.columns:
                risk = df.groupby(col)[target_col].mean()
                computed_maps[col] = risk.to_dict()
                df[f'{col}_risk_score'] = df[col].map(risk).fillna(global_mean)
            else:
                df[f'{col}_risk_score'] = global_mean
        os.makedirs('data/metadata', exist_ok=True)
        with open('data/metadata/risk_maps.json','w') as f:
            json.dump(computed_maps, f)
    elif not is_training and risk_maps is not None:
        global_mean = risk_maps.get('__global_mean__', 60.0)
        for col in risk_cols:
            if col in df.columns and col in risk_maps:
                df[f'{col}_risk_score'] = df[col].map(risk_maps[col]).fillna(global_mean)
            else:
                df[f'{col}_risk_score'] = global_mean
    else:
        for col in risk_cols:
            df[f'{col}_risk_score'] = 0.0

    # FREQUENCY
    for col in ['zone','junction','corridor']:
        if col in df.columns:
            df[f'{col}_event_count'] = df.groupby(col)['id'].transform('count') \
                if 'id' in df.columns else 1
        else:
            df[f'{col}_event_count'] = 1

    # INTERACTION
    high_risk_types = ['political_rally','rally','festival','procession',
                       'sports','strike','protest','march']
    if 'event_type' in df.columns:
        df['is_high_risk_type']    = df['event_type'].isin(high_risk_types).astype(int)
        df['high_risk_x_peak']     = df['is_high_risk_type'] * df.get('is_peak_hour', pd.Series(0,index=df.index))
        df['high_risk_x_weekend']  = df['is_high_risk_type'] * df.get('is_weekend',  pd.Series(0,index=df.index))
    else:
        df['is_high_risk_type']   = 0
        df['high_risk_x_peak']    = 0
        df['high_risk_x_weekend'] = 0

    return df

# ── FEATURE LISTS ─────────────────────────────────────────────────────────────
TEMPORAL_FEATURES = [
    'hour','day_of_week','month','week_of_year','quarter',
    'is_weekend','is_peak_hour','is_peak_morning','is_peak_evening',
    'is_night','is_midday',
    'hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos'
]
SPATIAL_FEATURES = [
    'event_span_km','resolution_displacement_km','is_moving_event',
    'latitude','longitude'
]
OPERATIONAL_FEATURES = [
    'road_closure_flag','has_cargo','has_vehicle',
    'is_breakdown','truck_age_bucket',
    'create_to_start_mins','is_reactive','is_authenticated'
]
RISK_SCORE_FEATURES = [
    'zone_risk_score','junction_risk_score','corridor_risk_score',
    'event_type_risk_score','event_cause_risk_score',
    'police_station_risk_score'
]
COUNT_FEATURES = [
    'zone_event_count','junction_event_count','corridor_event_count'
]
INTERACTION_FEATURES = [
    'is_high_risk_type','high_risk_x_peak','high_risk_x_weekend'
]
CAT_ENCODED_FEATURES = [c+'_enc' for c in CATEGORICAL_FEATURES]

ALL_FEATURES = (TEMPORAL_FEATURES + SPATIAL_FEATURES + OPERATIONAL_FEATURES +
                RISK_SCORE_FEATURES + COUNT_FEATURES + INTERACTION_FEATURES +
                CAT_ENCODED_FEATURES)
