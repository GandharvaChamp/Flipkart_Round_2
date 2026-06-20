"""
heatmap.py — Folium congestion heatmap generation.
"""
import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np

HYDERABAD_LOCATIONS = {
    'charminar':     (17.3616, 78.4747), 'lal darwaza':   (17.3616, 78.4740),
    'hussain sagar': (17.4239, 78.4738), 'tank bund':     (17.4239, 78.4738),
    'necklace road': (17.4126, 78.4657), 'abids':         (17.3950, 78.4667),
    'secunderabad':  (17.4399, 78.4983), 'ameerpet':      (17.4375, 78.4482),
    'jubilee hills': (17.4325, 78.4072), 'banjara hills': (17.4156, 78.4483),
    'hitech city':   (17.4435, 78.3772), 'gachibowli':    (17.4401, 78.3489),
    'lb nagar':      (17.3483, 78.5513), 'dilsukhnagar':  (17.3688, 78.5267),
    'kukatpally':    (17.4849, 78.4138), 'uppal':         (17.4051, 78.5597),
}

def location_lookup(addr: str) -> tuple:
    if not isinstance(addr, str):
        return (17.3850, 78.4867)
    addr_lower = addr.lower().strip()
    for key, coords in HYDERABAD_LOCATIONS.items():
        if key in addr_lower:
            return coords
    return (17.3850, 78.4867)

def generate_heatmap(df: pd.DataFrame,
                     lat_col='latitude', lon_col='longitude',
                     intensity_col='predicted_duration_mins') -> folium.Map:
    # Fallback lat/lon from address if coordinates missing
    if lat_col not in df.columns or df[lat_col].isna().mean() > 0.5:
        addr_col = 'address' if 'address' in df.columns else None
        if addr_col:
            coords = df[addr_col].apply(
                lambda x: pd.Series(location_lookup(str(x)),
                                    index=['latitude','longitude']))
            df = df.copy()
            df['latitude']  = df.get('latitude',  pd.Series(dtype=float)).fillna(coords['latitude'])
            df['longitude'] = df.get('longitude', pd.Series(dtype=float)).fillna(coords['longitude'])

    df = df.dropna(subset=[lat_col, lon_col])
    center_lat = df[lat_col].median()
    center_lon = df[lon_col].median()

    m = folium.Map(location=[center_lat, center_lon],
                   zoom_start=12, tiles='CartoDB positron')

    if intensity_col in df.columns:
        heat_data = df[[lat_col, lon_col, intensity_col]].dropna().values.tolist()
    else:
        heat_data = df[[lat_col, lon_col]].dropna().assign(w=1).values.tolist()

    HeatMap(heat_data, radius=20, blur=15, min_opacity=0.4,
            max_zoom=13).add_to(m)

    # Markers for top 50 events
    sample = df.nlargest(50, intensity_col) if intensity_col in df.columns else df.head(50)
    for _, row in sample.iterrows():
        val = row.get(intensity_col, 60)
        color = 'red' if val > 180 else 'orange' if val > 90 else 'green'
        popup_text = (
            f"<b>{row.get('event_type','Event')}</b><br>"
            f"Zone: {row.get('zone','N/A')}<br>"
            f"Duration: {val:.0f} min<br>"
            f"Status: {row.get('status','N/A')}"
        )
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=8, color=color, fill=True, fill_opacity=0.7,
            popup=folium.Popup(popup_text, max_width=200)
        ).add_to(m)

    return m
