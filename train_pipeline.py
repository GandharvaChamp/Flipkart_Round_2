"""
train_pipeline.py — Run this once to train all models.
Usage: cd event_congestion && python src/train_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from preprocess import load_and_clean, build_targets, build_features
from model import train

DATA_PATH = 'data/raw/astram_event_data.csv'

def main():
    print("=" * 55)
    print("TrafficSense — Training Pipeline")
    print("=" * 55)

    print("\n[1/3] Loading and cleaning data...")
    df = load_and_clean(DATA_PATH)

    print("\n[2/3] Building targets and features...")
    df = build_targets(df)
    df = build_features(df, is_training=True)

    if 'resolution_duration_mins' not in df.columns:
        print("\n[ERROR] resolution_duration_mins not found.")
        print("Check that resolved_datetime and start_datetime exist.")
        return

    valid = (df['resolution_duration_mins'].notna() &
             (df['resolution_duration_mins'] > 0) &
             (df['resolution_duration_mins'] < 1440))
    if valid.sum() < 50:
        print(f"\n[WARNING] Only {valid.sum()} valid target rows.")
        print("Switching to priority_encoded as target...")
        # Fallback: if priority exists, train on that instead
        if 'priority_encoded' in df.columns:
            df['resolution_duration_mins'] = df['priority_encoded'] * 60
        else:
            print("[ERROR] No valid target found. Check dataset.")
            return

    print("\n[3/3] Training ensemble (LightGBM + XGBoost + CatBoost)...")
    lgb_m, xgb_m, cat_m, weights, features, use_log = train(df)

    print("\n" + "=" * 55)
    print("Training complete. Artifacts saved to models/ and data/")
    print("Run: streamlit run app/streamlit_app.py")
    print("=" * 55)

if __name__ == '__main__':
    main()
