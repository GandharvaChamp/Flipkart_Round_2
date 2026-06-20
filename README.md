# TrafficSense — Event-Driven Congestion Forecaster

## Setup
```bash
git clone <repo>
cd event_congestion
pip install -r requirements.txt
cp path/to/astram_event_data.csv data/raw/astram_event_data.csv
# export GEMINI_API_KEY=your_key_here
```

## Train Models
```bash
python src/train_pipeline.py
```

## Run Dashboard
```bash
streamlit run app/streamlit_app.py
```

## Features
- LightGBM + XGBoost + CatBoost ensemble with optimal blending
- 5-fold CV with leak-free risk score computation
- AI-generated officer deployment briefings (Claude API)
- Real-time what-if simulator with gauge chart
- Folium congestion heatmap
- Post-event accuracy tracking and learning loop
- SQLite feedback logging
