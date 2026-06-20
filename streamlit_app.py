# """
# streamlit_app.py — TrafficSense: Event-Driven Congestion Forecaster
# """
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# import sys, os, json, warnings
# warnings.filterwarnings('ignore')

# sys.path.append(os.path.dirname(__file__))

# from preprocess import load_and_clean, build_targets, build_features, CATEGORICAL_FEATURES
# from model import load_artifacts, predict_duration, classify_severity, encode_categoricals
# from database import init_db, log_forecast, update_actual, get_all_forecasts, get_accuracy_df
# from heatmap import generate_heatmap
# from llm_briefing import generate_briefing, generate_post_event_report

# st.set_page_config(page_title="TrafficSense", page_icon="🚦", layout="wide")
# init_db()

# @st.cache_resource(show_spinner="Loading trained models...")
# def get_models():
#     try:
#         return load_artifacts()
#     except Exception as e:
#         return None

# @st.cache_data(show_spinner="Loading dataset...")
# def get_dataset():
#     try:
#         df = load_and_clean('data/raw/astram_event_data.csv')
#         df = build_targets(df)
#         df = build_features(df, is_training=True)
#         return df
#     except Exception as e:
#         return None

# models_bundle = get_models()
# df_full = get_dataset()

# st.sidebar.markdown("## 🚦 TrafficSense")
# st.sidebar.caption("Event-Driven Congestion Forecaster")
# st.sidebar.markdown("---")
# page = st.sidebar.radio("Navigate", [
#     "📊 Dashboard", "🔮 Forecast Event", "🎛️ What-If Simulator",
#     "📋 Officer Briefing", "🗺️ Heatmap", "📈 Post-Event Analytics"
# ])
# model_status = "✅ Models Loaded" if models_bundle else "⚠️ Train models first"
# data_status  = f"✅ {len(df_full)} events" if df_full is not None else "⚠️ No dataset"
# st.sidebar.markdown(f"**Status:**  \n{model_status}  \n{data_status}")

# # ─── PAGE 1: DASHBOARD ────────────────────────────────────────────────────────
# if page == "📊 Dashboard":
#     st.title("📊 Event Congestion Overview")
#     if df_full is None:
#         st.error("Place CSV at data/raw/astram_event_data.csv"); st.stop()
#     df = df_full.copy()
#     c1,c2,c3,c4,c5 = st.columns(5)
#     c1.metric("Total Events", f"{len(df):,}")
#     if 'road_closure_flag' in df.columns:
#         c2.metric("Road Closures", f"{df['road_closure_flag'].sum():,}")
#     if 'resolution_duration_mins' in df.columns:
#         v = df[df['resolution_duration_mins'].between(0,1440)]
#         c3.metric("Avg Resolution", f"{v['resolution_duration_mins'].mean():.0f} min")
#         c4.metric("Median Resolution", f"{v['resolution_duration_mins'].median():.0f} min")
#     if 'zone' in df.columns:
#         c5.metric("Unique Zones", df['zone'].nunique())
#     st.markdown("---")
#     col1, col2 = st.columns(2)
#     if 'event_type' in df.columns:
#         et = df['event_type'].value_counts().head(15)
#         fig1 = px.bar(x=et.values, y=et.index, orientation='h',
#                       title="Top 15 Event Types", color=et.values,
#                       color_continuous_scale='RdYlGn_r')
#         fig1.update_layout(showlegend=False, height=380)
#         col1.plotly_chart(fig1, use_container_width=True)
#     if 'event_type' in df.columns and 'resolution_duration_mins' in df.columns:
#         v = df[df['resolution_duration_mins'].between(0,1440)]
#         dur = v.groupby('event_type')['resolution_duration_mins'].median().sort_values().tail(15)
#         fig2 = px.bar(x=dur.values, y=dur.index, orientation='h',
#                       title="Median Duration by Event Type (min)", color=dur.values,
#                       color_continuous_scale='YlOrRd')
#         fig2.update_layout(showlegend=False, height=380)
#         col2.plotly_chart(fig2, use_container_width=True)
#     if all(c in df.columns for c in ['day_of_week','hour','resolution_duration_mins']):
#         v = df[df['resolution_duration_mins'].between(0,1440)]
#         pivot = v.pivot_table(values='resolution_duration_mins',
#                                index='day_of_week', columns='hour', aggfunc='median')
#         pivot.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][:len(pivot)]
#         fig3 = go.Figure(go.Heatmap(z=pivot.values,
#             x=[str(h) for h in pivot.columns], y=pivot.index.tolist(),
#             colorscale='YlOrRd', colorbar=dict(title='Min')))
#         fig3.update_layout(title='Median Duration: Day × Hour',
#                            xaxis_title='Hour', yaxis_title='Day', height=320)
#         st.plotly_chart(fig3, use_container_width=True)
#     show_cols = [c for c in ['event_type','event_cause','zone','junction',
#                               'corridor','status','priority',
#                               'resolution_duration_mins','road_closure_flag']
#                  if c in df.columns]
#     st.dataframe(df[show_cols].head(50), use_container_width=True)

# # ─── PAGE 2: FORECAST ─────────────────────────────────────────────────────────
# elif page == "🔮 Forecast Event":
#     st.title("🔮 Forecast Event Impact")
#     if models_bundle is None:
#         st.warning("Run `python src/train_pipeline.py` first."); st.stop()
#     lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps = models_bundle
#     zone_opts  = sorted(df_full['zone'].dropna().unique().tolist())  if df_full is not None and 'zone'       in df_full.columns else ['unknown']
#     junc_opts  = sorted(df_full['junction'].dropna().unique().tolist()) if df_full is not None and 'junction'  in df_full.columns else ['unknown']
#     corr_opts  = sorted(df_full['corridor'].dropna().unique().tolist()) if df_full is not None and 'corridor'  in df_full.columns else ['unknown']
#     etype_opts = sorted(df_full['event_type'].dropna().unique().tolist()) if df_full is not None and 'event_type' in df_full.columns else ['festival']
#     ecause_opts= sorted(df_full['event_cause'].dropna().unique().tolist()) if df_full is not None and 'event_cause' in df_full.columns else ['traffic']
#     with st.form("forecast_form"):
#         c1,c2,c3 = st.columns(3)
#         event_type  = c1.selectbox("Event Type",   etype_opts)
#         event_cause = c2.selectbox("Event Cause",  ecause_opts)
#         zone        = c3.selectbox("Zone",         zone_opts)
#         junction    = c1.selectbox("Junction",     junc_opts)
#         corridor    = c2.selectbox("Corridor",     corr_opts)
#         direction   = c3.selectbox("Direction",    ['north','south','east','west','both'])
#         address     = c1.text_input("Address", "Charminar, Hyderabad")
#         start_dt    = c2.date_input("Event Date")
#         start_time  = c3.time_input("Start Time")
#         road_closure= c1.checkbox("Road Closure Required?")
#         has_vehicle = c2.checkbox("Vehicle Involved?")
#         has_cargo   = c3.checkbox("Cargo Present?")
#         submitted   = st.form_submit_button("🔮 Generate Forecast")
#     if submitted:
#         import datetime
#         sdt = datetime.datetime.combine(start_dt, start_time)
#         ev = {'event_type':event_type,'event_cause':event_cause,'zone':zone,
#               'junction':junction,'corridor':corridor,'direction':direction,
#               'address':address,'start_datetime':sdt,'created_date':sdt,
#               'latitude':np.nan,'longitude':np.nan,'endlatitude':np.nan,
#               'endlongitude':np.nan,'resolved_at_latitude':np.nan,
#               'resolved_at_longitude':np.nan,'road_closure_flag':int(road_closure),
#               'has_vehicle':int(has_vehicle),'has_cargo':int(has_cargo),
#               'is_authenticated':1,'veh_type':np.nan,'police_station':np.nan,
#               'cargo_material':None,'reason_breakdown':None,'age_of_truck':np.nan}
#         with st.spinner("Running ensemble prediction..."):
#             predicted = predict_duration(ev,lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps)
#         severity = classify_severity(predicted)
#         st.markdown("---")
#         st.markdown(f"### Predicted Resolution Time: `{predicted:.0f} minutes`")
#         st.markdown(f"<div style='background:{severity['color']}22;border-left:5px solid "
#                     f"{severity['color']};padding:14px;border-radius:6px;'>"
#                     f"<b>{severity['label']}</b> — {severity['action']}</div>",
#                     unsafe_allow_html=True)
#         m1,m2,m3,m4 = st.columns(4)
#         m1.metric("👮 Officers",   severity['officers'])
#         m2.metric("🚧 Barricades", severity['barricades'])
#         m3.metric("↩️ Diversions", severity['diversions'])
#         m4.metric("⏱️ Duration",   f"{predicted:.0f} min")
#         st.session_state.update({'last_event':ev,'last_severity':severity,'last_predicted':predicted})
#         log_forecast(ev, predicted, severity)
#         st.success("✅ Logged. Go to Officer Briefing for deployment order.")

# # ─── PAGE 3: WHAT-IF ─────────────────────────────────────────────────────────
# elif page == "🎛️ What-If Simulator":
#     st.title("🎛️ What-If Simulator")
#     if models_bundle is None:
#         st.warning("Models not trained yet."); st.stop()
#     lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps = models_bundle
#     zone_opts  = sorted(df_full['zone'].dropna().unique().tolist()) if df_full is not None and 'zone' in df_full.columns else ['zone_a']
#     etype_opts = sorted(df_full['event_type'].dropna().unique().tolist()) if df_full is not None and 'event_type' in df_full.columns else ['festival']
#     c1,c2 = st.columns(2)
#     event_type   = c1.selectbox("Event Type", etype_opts)
#     zone         = c2.selectbox("Zone",        zone_opts)
#     hour         = c1.slider("Start Hour", 0, 23, 18)
#     is_weekend   = c1.checkbox("Weekend?", value=True)
#     road_closure = c2.checkbox("Road Closure?")
#     import datetime
#     bdt = datetime.datetime(2024,1,(6 if is_weekend else 2),hour,0,0)
#     ev = {'event_type':event_type,'event_cause':'traffic','zone':zone,
#           'junction':'unknown','corridor':'unknown','direction':'both',
#           'address':'Hyderabad','start_datetime':bdt,'created_date':bdt,
#           'latitude':np.nan,'longitude':np.nan,'endlatitude':np.nan,
#           'endlongitude':np.nan,'resolved_at_latitude':np.nan,
#           'resolved_at_longitude':np.nan,'road_closure_flag':int(road_closure),
#           'has_vehicle':0,'has_cargo':0,'is_authenticated':1,
#           'veh_type':np.nan,'police_station':np.nan,
#           'cargo_material':None,'reason_breakdown':None,'age_of_truck':np.nan}
#     predicted = predict_duration(ev,lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps)
#     severity  = classify_severity(predicted)
#     st.markdown("---")
#     ga,gb = st.columns([1,2])
#     ga.metric("🚦 Predicted Duration", f"{predicted:.0f} min")
#     ga.metric("Severity",              severity['label'])
#     ga.metric("Officers Needed",       severity['officers'])
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number", value=min(predicted,300),
#         gauge={'axis':{'range':[0,300]},'bar':{'color':severity['color']},
#                'steps':[{'range':[0,30],'color':'#E8F5E9'},
#                         {'range':[30,90],'color':'#FFF9C4'},
#                         {'range':[90,180],'color':'#FFE0B2'},
#                         {'range':[180,300],'color':'#FFCDD2'}]},
#         title={'text':'Est. Resolution (min)'}))
#     gb.plotly_chart(fig, use_container_width=True)

# # ─── PAGE 4: OFFICER BRIEFING ─────────────────────────────────────────────────
# elif page == "📋 Officer Briefing":
#     st.title("📋 AI-Generated Officer Deployment Briefing")
#     if 'last_event' not in st.session_state:
#         st.warning("Please forecast an event first."); st.stop()
#     ev  = st.session_state['last_event']
#     sev = st.session_state['last_severity']
#     pred= st.session_state['last_predicted']
#     st.subheader(f"Event: {str(ev.get('event_type','')).title()}")
#     st.caption(f"{ev.get('address','')} | Zone: {ev.get('zone','')} | Junction: {ev.get('junction','')}")
#     c1,c2,c3 = st.columns(3)
#     c1.metric("Severity",   sev['label'])
#     c2.metric("Officers",   sev['officers'])
#     c3.metric("Predicted",  f"{pred:.0f} min")
#     # if st.button("🤖 Generate Deployment Briefing (Claude AI)"):
#     #     if not os.environ.get("ANTHROPIC_API_KEY",""):
#     #         st.error("Set ANTHROPIC_API_KEY environment variable.")
#     #     else:
#     #         with st.spinner("Generating briefing..."):
#     #             briefing = generate_briefing(ev, sev, pred)
#     #         st.markdown("---")
#     #         st.markdown(briefing)
#     #         st.download_button("📥 Download Briefing", data=briefing,
#     #                            file_name="deployment_briefing.txt", mime="text/plain")
#     if st.button("📋 Generate Deployment Briefing"):
#         with st.spinner("Generating briefing..."):
#             briefing = generate_briefing(ev, sev, pred)
#         st.markdown("---")
#         st.code(briefing, language=None)
#         st.download_button("📥 Download Briefing (.txt)", data=briefing,
#                        file_name="deployment_briefing.txt", mime="text/plain")

# # ─── PAGE 5: HEATMAP ─────────────────────────────────────────────────────────
# elif page == "🗺️ Heatmap":
#     st.title("🗺️ Congestion Heatmap")
#     from streamlit_folium import st_folium
#     if df_full is None:
#         st.error("Dataset not loaded."); st.stop()
#     df = df_full.copy()
#     if 'resolution_duration_mins' not in df.columns:
#         df['resolution_duration_mins'] = 60.0
#     m = generate_heatmap(df, intensity_col='resolution_duration_mins')
#     st_folium(m, width=950, height=520)

# # ─── PAGE 6: ANALYTICS ───────────────────────────────────────────────────────
# elif page == "📈 Post-Event Analytics":
#     st.title("📈 Post-Event Learning Analytics")
#     df_log = get_all_forecasts()
#     if df_log.empty:
#         st.info("No forecasts logged yet."); st.stop()
#     st.dataframe(df_log, use_container_width=True)
#     with st.form("actual_form"):
#         fid         = st.number_input("Forecast ID to update", min_value=1, step=1)
#         actual_mins = st.number_input("Actual Duration (minutes)", min_value=1.0)
#         if st.form_submit_button("Update Actual"):
#             update_actual(int(fid), actual_mins)
#             st.success(f"Updated #{fid}")
#     acc_df = get_accuracy_df()
#     if not acc_df.empty:
#         st.metric("Mean Absolute Error", f"{acc_df['abs_error'].mean():.1f} min")
#         fig = px.scatter(acc_df, x='predicted_duration_mins', y='actual_duration_mins',
#                          color='event_type', title='Predicted vs Actual Duration')
#         mx = max(acc_df[['predicted_duration_mins','actual_duration_mins']].max())
#         fig.add_shape(type='line',x0=0,y0=0,x1=mx,y1=mx,
#                       line=dict(dash='dash',color='gray'))
#         st.plotly_chart(fig, use_container_width=True)


#---------------------------------------------------New-----------------------------------------

# """
# streamlit_app.py — TrafficSense: Event-Driven Congestion Forecaster
# """
# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.graph_objects as go
# import plotly.express as px
# import sys, os, json, warnings
# warnings.filterwarnings('ignore')
 
# sys.path.append(os.path.dirname(__file__))
 
# from preprocess import load_and_clean, build_targets, build_features, CATEGORICAL_FEATURES
# from model import load_artifacts, predict_duration, classify_severity, encode_categoricals
# from database import (init_db, log_forecast, update_actual,
#                       get_all_forecasts, get_accuracy_df,
#                       log_event_memory, get_similar_past_events,
#                       get_all_event_memories, get_memory_stats)
# from heatmap import generate_heatmap
# from llm_briefing import generate_briefing, generate_post_event_report
 
# st.set_page_config(page_title="TrafficSense", page_icon="🚦", layout="wide")
# init_db()
 
# @st.cache_resource(show_spinner="Loading trained models...")
# def get_models():
#     try:
#         return load_artifacts()
#     except Exception as e:
#         return None
 
# @st.cache_data(show_spinner="Loading dataset...")
# def get_dataset():
#     try:
#         df = load_and_clean('data/raw/astram_event_data.csv')
#         df = build_targets(df)
#         df = build_features(df, is_training=True)
#         return df
#     except Exception as e:
#         return None
 
# models_bundle = get_models()
# df_full = get_dataset()
 
# st.sidebar.markdown("## 🚦 TrafficSense")
# st.sidebar.caption("Event-Driven Congestion Forecaster")
# st.sidebar.markdown("---")
# page = st.sidebar.radio("Navigate", [
#     "📊 Dashboard", "🔮 Forecast Event", "🎛️ What-If Simulator",
#     "📋 Officer Briefing", "🗺️ Heatmap",
#     "📝 Post-Event Feedback", "📈 Post-Event Analytics"
# ])
# model_status = "✅ Models Loaded" if models_bundle else "⚠️ Train models first"
# data_status  = f"✅ {len(df_full)} events" if df_full is not None else "⚠️ No dataset"
# st.sidebar.markdown(f"**Status:**  \n{model_status}  \n{data_status}")
 
 
# # ═══════════════════════════════════════════════════════════════════════════════
# # HELPER: Render a single past event memory card
# # ═══════════════════════════════════════════════════════════════════════════════
# def render_memory_card(m: dict, idx: int):
#     """Render one past event memory as a styled expander card."""
#     match_quality = []
#     pred  = m.get('predicted_duration_mins', 0) or 0
#     actual = m.get('actual_duration_mins', 0) or 0
#     error  = abs(pred - actual) if actual else None
#     rating = m.get('overall_rating', None)
#     stars  = "⭐" * int(rating) if rating else "—"
 
#     with st.expander(
#         f"📁 Past Event #{idx+1} — {str(m.get('event_type','')).replace('_',' ').title()} | "
#         f"{m.get('zone','?')} | {m.get('event_date','?')[:10] if m.get('event_date') else '?'}  |  Rating: {stars}",
#         expanded=(idx == 0)
#     ):
#         col1, col2, col3 = st.columns(3)
#         col1.metric("Predicted", f"{pred:.0f} min")
#         col2.metric("Actual",    f"{actual:.0f} min" if actual else "—")
#         col3.metric("Error",     f"{error:.0f} min"  if error  else "—")
 
#         st.markdown("---")
#         c1, c2 = st.columns(2)
 
#         # Officers
#         off_ok = m.get('officers_were_sufficient')
#         off_icon = "✅" if off_ok == 1 else ("❌" if off_ok == 0 else "—")
#         c1.markdown(f"**👮 Officers deployed:** {m.get('officers_deployed','—')} {off_icon}")
 
#         # Barricades
#         bar_ok = m.get('barricades_were_sufficient')
#         bar_icon = "✅" if bar_ok == 1 else ("❌" if bar_ok == 0 else "—")
#         c2.markdown(f"**🚧 Barricades deployed:** {m.get('barricades_deployed','—')} {bar_icon}")
 
#         # Diversions
#         div_eff = m.get('diversion_effective', -1)
#         div_icon = "✅" if div_eff == 1 else ("❌" if div_eff == 0 else "N/A")
#         c1.markdown(f"**↩️ Diversion effective:** {div_icon}")
#         c1.markdown(f"**Routes used:** {m.get('diversion_routes_used','—')}")
 
#         c2.markdown(f"**👥 Crowd density:** {m.get('crowd_density','—').title() if m.get('crowd_density') else '—'}")
#         c2.markdown(f"**🌦️ Weather impact:** {m.get('weather_impact','—').title() if m.get('weather_impact') else '—'}")
 
#         st.markdown("---")
#         if m.get('what_worked_well'):
#             st.success(f"**✅ What worked well:**\n\n{m['what_worked_well']}")
#         if m.get('what_didnt_work'):
#             st.error(f"**❌ What didn't work:**\n\n{m['what_didnt_work']}")
#         if m.get('unexpected_factors'):
#             st.warning(f"**⚠️ Unexpected factors:**\n\n{m['unexpected_factors']}")
 
 
# # ═══════════════════════════════════════════════════════════════════════════════
# # PAGE 1: DASHBOARD
# # ═══════════════════════════════════════════════════════════════════════════════
# if page == "📊 Dashboard":
#     st.title("📊 Event Congestion Overview")
#     if df_full is None:
#         st.error("Place CSV at data/raw/astram_event_data.csv"); st.stop()
#     df = df_full.copy()
#     c1,c2,c3,c4,c5 = st.columns(5)
#     c1.metric("Total Events", f"{len(df):,}")
#     if 'road_closure_flag' in df.columns:
#         c2.metric("Road Closures", f"{df['road_closure_flag'].sum():,}")
#     if 'resolution_duration_mins' in df.columns:
#         v = df[df['resolution_duration_mins'].between(0,1440)]
#         c3.metric("Avg Resolution", f"{v['resolution_duration_mins'].mean():.0f} min")
#         c4.metric("Median Resolution", f"{v['resolution_duration_mins'].median():.0f} min")
#     if 'zone' in df.columns:
#         c5.metric("Unique Zones", df['zone'].nunique())
#     st.markdown("---")
#     col1, col2 = st.columns(2)
#     if 'event_type' in df.columns:
#         et = df['event_type'].value_counts().head(15)
#         fig1 = px.bar(x=et.values, y=et.index, orientation='h',
#                       title="Top 15 Event Types", color=et.values,
#                       color_continuous_scale='RdYlGn_r')
#         fig1.update_layout(showlegend=False, height=380)
#         col1.plotly_chart(fig1, use_container_width=True)
#     if 'event_type' in df.columns and 'resolution_duration_mins' in df.columns:
#         v = df[df['resolution_duration_mins'].between(0,1440)]
#         dur = v.groupby('event_type')['resolution_duration_mins'].median().sort_values().tail(15)
#         fig2 = px.bar(x=dur.values, y=dur.index, orientation='h',
#                       title="Median Duration by Event Type (min)", color=dur.values,
#                       color_continuous_scale='YlOrRd')
#         fig2.update_layout(showlegend=False, height=380)
#         col2.plotly_chart(fig2, use_container_width=True)
#     if all(c in df.columns for c in ['day_of_week','hour','resolution_duration_mins']):
#         v = df[df['resolution_duration_mins'].between(0,1440)]
#         pivot = v.pivot_table(values='resolution_duration_mins',
#                                index='day_of_week', columns='hour', aggfunc='median')
#         pivot.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][:len(pivot)]
#         fig3 = go.Figure(go.Heatmap(z=pivot.values,
#             x=[str(h) for h in pivot.columns], y=pivot.index.tolist(),
#             colorscale='YlOrRd', colorbar=dict(title='Min')))
#         fig3.update_layout(title='Median Duration: Day × Hour',
#                            xaxis_title='Hour', yaxis_title='Day', height=320)
#         st.plotly_chart(fig3, use_container_width=True)
#     show_cols = [c for c in ['event_type','event_cause','zone','junction',
#                               'corridor','status','priority',
#                               'resolution_duration_mins','road_closure_flag']
#                  if c in df.columns]
#     st.dataframe(df[show_cols].head(50), use_container_width=True)
 
 
# # ═══════════════════════════════════════════════════════════════════════════════
# # PAGE 2: FORECAST EVENT  (+ Institutional Memory lookup)
# # ═══════════════════════════════════════════════════════════════════════════════
# # elif page == "🔮 Forecast Event":
# #     st.title("🔮 Forecast Event Impact")
# #     if models_bundle is None:
# #         st.warning("Run `python src/train_pipeline.py` first."); st.stop()
# #     lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps = models_bundle
# #     zone_opts  = sorted(df_full['zone'].dropna().unique().tolist())  if df_full is not None and 'zone'       in df_full.columns else ['unknown']
# #     junc_opts  = sorted(df_full['junction'].dropna().unique().tolist()) if df_full is not None and 'junction'  in df_full.columns else ['unknown']
# #     corr_opts  = sorted(df_full['corridor'].dropna().unique().tolist()) if df_full is not None and 'corridor'  in df_full.columns else ['unknown']
# #     etype_opts = sorted(df_full['event_type'].dropna().unique().tolist()) if df_full is not None and 'event_type' in df_full.columns else ['festival']
# #     ecause_opts= sorted(df_full['event_cause'].dropna().unique().tolist()) if df_full is not None and 'event_cause' in df_full.columns else ['traffic']
 
# #     with st.form("forecast_form"):
# #         c1,c2,c3 = st.columns(3)
# #         event_type  = c1.selectbox("Event Type",   etype_opts)
# #         event_cause = c2.selectbox("Event Cause",  ecause_opts)
# #         zone        = c3.selectbox("Zone",         zone_opts)
# #         junction    = c1.selectbox("Junction",     junc_opts)
# #         corridor    = c2.selectbox("Corridor",     corr_opts)
# #         direction   = c3.selectbox("Direction",    ['north','south','east','west','both'])
# #         address     = c1.text_input("Address", "Charminar, Hyderabad")
# #         start_dt    = c2.date_input("Event Date")
# #         start_time  = c3.time_input("Start Time")
# #         road_closure= c1.checkbox("Road Closure Required?")
# #         has_vehicle = c2.checkbox("Vehicle Involved?")
# #         has_cargo   = c3.checkbox("Cargo Present?")
# #         submitted   = st.form_submit_button("🔮 Generate Forecast")
 
# #     if submitted:
# #         import datetime
# #         sdt = datetime.datetime.combine(start_dt, start_time)
# #         ev = {'event_type':event_type,'event_cause':event_cause,'zone':zone,
# #               'junction':junction,'corridor':corridor,'direction':direction,
# #               'address':address,'start_datetime':sdt,'created_date':sdt,
# #               'latitude':np.nan,'longitude':np.nan,'endlatitude':np.nan,
# #               'endlongitude':np.nan,'resolved_at_latitude':np.nan,
# #               'resolved_at_longitude':np.nan,'road_closure_flag':int(road_closure),
# #               'has_vehicle':int(has_vehicle),'has_cargo':int(has_cargo),
# #               'is_authenticated':1,'veh_type':np.nan,'police_station':np.nan,
# #               'cargo_material':None,'reason_breakdown':None,'age_of_truck':np.nan}
 
# #         # ── INSTITUTIONAL MEMORY LOOKUP ────────────────────────────────────
# #         past_events = get_similar_past_events(event_type, zone, junction, limit=5)
 
# #         if past_events:
# #             st.markdown("---")
# #             st.markdown(
# #                 f"### 🧠 Institutional Memory — {len(past_events)} Similar Past Event(s) Found",
# #             )
# #             st.caption(
# #                 f"The following records match **{event_type.replace('_',' ').title()}** "
# #                 f"events in zone **{zone}**. Review learnings before accepting the ML forecast."
# #             )
 
# #             # Summary strip from past events
# #             avg_actual = np.mean([p['actual_duration_mins'] for p in past_events
# #                                   if p.get('actual_duration_mins')])
# #             off_issues = sum(1 for p in past_events if p.get('officers_were_sufficient') == 0)
# #             bar_issues = sum(1 for p in past_events if p.get('barricades_were_sufficient') == 0)
# #             avg_rating = np.mean([p['overall_rating'] for p in past_events
# #                                   if p.get('overall_rating')])
 
# #             mc1, mc2, mc3, mc4 = st.columns(4)
# #             mc1.metric("Avg Actual Duration (past)", f"{avg_actual:.0f} min" if avg_actual else "—")
# #             mc2.metric("Officer shortfall cases",    f"{off_issues}/{len(past_events)}")
# #             mc3.metric("Barricade shortfall cases",  f"{bar_issues}/{len(past_events)}")
# #             mc4.metric("Avg officer rating",         f"{avg_rating:.1f}/5" if avg_rating else "—")
 
# #             for i, mem in enumerate(past_events):
# #                 render_memory_card(mem, i)
 
# #             st.markdown("---")
# #             st.markdown("### 🔮 ML Model Forecast")
# #             st.caption("This is the model's fresh prediction. Compare it against historical actuals above.")
# #         else:
# #             st.markdown("---")
# #             st.info(
# #                 "🆕 **No past records found** for this event type + zone combination. "
# #                 "This is a first-time event — forecast is purely ML-based. "
# #                 "After the event, submit feedback via **Post-Event Feedback** so future "
# #                 "planners benefit from your experience."
# #             )
# #             st.markdown("### 🔮 ML Model Forecast")
 
# #         # ── RUN ML PREDICTION ─────────────────────────────────────────────
# #         with st.spinner("Running ensemble prediction..."):
# #             predicted = predict_duration(ev,lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps)
# #         severity = classify_severity(predicted)
 
# #         st.markdown(f"### Predicted Resolution Time: `{predicted:.0f} minutes`")
# #         st.markdown(f"<div style='background:{severity['color']}22;border-left:5px solid "
# #                     f"{severity['color']};padding:14px;border-radius:6px;'>"
# #                     f"<b>{severity['label']}</b> — {severity['action']}</div>",
# #                     unsafe_allow_html=True)
# #         m1,m2,m3,m4 = st.columns(4)
# #         m1.metric("👮 Officers",   severity['officers'])
# #         m2.metric("🚧 Barricades", severity['barricades'])
# #         m3.metric("↩️ Diversions", severity['diversions'])
# #         m4.metric("⏱️ Duration",   f"{predicted:.0f} min")
 
# #         # If past events exist, show a delta vs historical average
# #         if past_events and avg_actual:
# #             delta = predicted - avg_actual
# #             st.caption(
# #                 f"ℹ️ Model predicts **{abs(delta):.0f} min {'more' if delta > 0 else 'less'}** "
# #                 f"than the historical average actual duration of {avg_actual:.0f} min."
# #             )
 
# #         st.session_state.update({'last_event':ev,'last_severity':severity,'last_predicted':predicted})
# #         log_forecast(ev, predicted, severity)
# #         st.success("✅ Logged. Go to Officer Briefing for deployment order, or submit feedback after the event.")


# # elif page == "🔮 Forecast Event":
# #     st.title("🔮 Forecast Event Impact")
# #     if models_bundle is None:
# #         st.warning("Run `python src/train_pipeline.py` first."); st.stop()
# #     lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps = models_bundle
# #     zone_opts  = sorted(df_full['zone'].dropna().unique().tolist())  if df_full is not None and 'zone'       in df_full.columns else ['unknown']
# #     junc_opts  = sorted(df_full['junction'].dropna().unique().tolist()) if df_full is not None and 'junction'  in df_full.columns else ['unknown']
# #     corr_opts  = sorted(df_full['corridor'].dropna().unique().tolist()) if df_full is not None and 'corridor'  in df_full.columns else ['unknown']
# #     etype_opts = sorted(df_full['event_type'].dropna().unique().tolist()) if df_full is not None and 'event_type' in df_full.columns else ['festival']
# #     ecause_opts= sorted(df_full['event_cause'].dropna().unique().tolist()) if df_full is not None and 'event_cause' in df_full.columns else ['traffic']

# #     with st.form("forecast_form"):
# #         c1,c2,c3 = st.columns(3)
# #         event_type  = c1.selectbox("Event Type",   etype_opts)
# #         event_cause = c2.selectbox("Event Cause",  ecause_opts)
# #         zone        = c3.selectbox("Zone",         zone_opts)
# #         junction    = c1.selectbox("Junction",     junc_opts)
# #         corridor    = c2.selectbox("Corridor",     corr_opts)
# #         direction   = c3.selectbox("Direction",    ['north','south','east','west','both'])
# #         address     = c1.text_input("Address", "Charminar, Hyderabad")
# #         start_dt    = c2.date_input("Event Date")
# #         start_time  = c3.time_input("Start Time")
# #         road_closure= c1.checkbox("Road Closure Required?")
# #         has_vehicle = c2.checkbox("Vehicle Involved?")
# #         has_cargo   = c3.checkbox("Cargo Present?")
# #         submitted   = st.form_submit_button("🔮 Generate Forecast")

# #     if submitted:
# #         import datetime
# #         sdt = datetime.datetime.combine(start_dt, start_time)
# #         ev = {'event_type':event_type,'event_cause':event_cause,'zone':zone,
# #               'junction':junction,'corridor':corridor,'direction':direction,
# #               'address':address,'start_datetime':sdt,'created_date':sdt,
# #               'latitude':np.nan,'longitude':np.nan,'endlatitude':np.nan,
# #               'endlongitude':np.nan,'resolved_at_latitude':np.nan,
# #               'resolved_at_longitude':np.nan,'road_closure_flag':int(road_closure),
# #               'has_vehicle':int(has_vehicle),'has_cargo':int(has_cargo),
# #               'is_authenticated':1,'veh_type':np.nan,'police_station':np.nan,
# #               'cargo_material':None,'reason_breakdown':None,'age_of_truck':np.nan}

# #         # Store submitted params in session so results are tied to THIS submission
# #         st.session_state['forecast_params'] = {
# #             'event_type': event_type, 'zone': zone, 'junction': junction
# #         }
# #         st.session_state['forecast_ev']        = ev
# #         st.session_state['forecast_submitted']  = True

# #         # Run memory lookup with submitted values
# #         past_events = get_similar_past_events(event_type, zone, junction, limit=5)
# #         st.session_state['past_events'] = past_events

# #         # Run ML prediction
# #         with st.spinner("Running ensemble prediction..."):
# #             predicted = predict_duration(ev,lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps)
# #         severity = classify_severity(predicted)

# #         st.session_state.update({
# #             'last_event':     ev,
# #             'last_severity':  severity,
# #             'last_predicted': predicted
# #         })
# #         log_forecast(ev, predicted, severity)

# #     # ── Render results only if a forecast has been submitted ─────────────────
# #     if st.session_state.get('forecast_submitted'):
# #         ev        = st.session_state['last_event']
# #         severity  = st.session_state['last_severity']
# #         predicted = st.session_state['last_predicted']
# #         past_events = st.session_state.get('past_events', [])
# #         params    = st.session_state.get('forecast_params', {})

# #         st.markdown("---")

# #         if past_events:
# #             st.markdown(f"### 🧠 Institutional Memory — {len(past_events)} Similar Past Event(s) Found")
# #             st.caption(
# #                 f"Matching **{params.get('event_type','').replace('_',' ').title()}** "
# #                 f"events in zone **{params.get('zone','')}**. "
# #                 f"Review before accepting the ML forecast."
# #             )
# #             avg_actual = np.mean([p['actual_duration_mins'] for p in past_events
# #                                   if p.get('actual_duration_mins')])
# #             off_issues = sum(1 for p in past_events if p.get('officers_were_sufficient') == 0)
# #             bar_issues = sum(1 for p in past_events if p.get('barricades_were_sufficient') == 0)
# #             avg_rating = np.mean([p['overall_rating'] for p in past_events
# #                                   if p.get('overall_rating')])

# #             mc1,mc2,mc3,mc4 = st.columns(4)
# #             mc1.metric("Avg Actual Duration (past)", f"{avg_actual:.0f} min" if avg_actual else "—")
# #             mc2.metric("Officer shortfall cases",    f"{off_issues}/{len(past_events)}")
# #             mc3.metric("Barricade shortfall cases",  f"{bar_issues}/{len(past_events)}")
# #             mc4.metric("Avg officer rating",         f"{avg_rating:.1f}/5" if avg_rating else "—")

# #             for i, mem in enumerate(past_events):
# #                 render_memory_card(mem, i)

# #             st.markdown("---")
# #             st.markdown("### 🔮 ML Model Forecast")
# #             st.caption("Fresh model prediction — compare against historical actuals above.")
# #         else:
# #             st.info(
# #                 "🆕 **No past records found** for this event type + zone combination. "
# #                 "Forecast is purely ML-based. After the event, submit feedback via "
# #                 "**Post-Event Feedback** so future planners benefit."
# #             )
# #             st.markdown("### 🔮 ML Model Forecast")

# #         st.markdown(f"### Predicted Resolution Time: `{predicted:.0f} minutes`")
# #         st.markdown(
# #             f"<div style='background:{severity['color']}22;border-left:5px solid "
# #             f"{severity['color']};padding:14px;border-radius:6px;'>"
# #             f"<b>{severity['label']}</b> — {severity['action']}</div>",
# #             unsafe_allow_html=True
# #         )
# #         m1,m2,m3,m4 = st.columns(4)
# #         m1.metric("👮 Officers",   severity['officers'])
# #         m2.metric("🚧 Barricades", severity['barricades'])
# #         m3.metric("↩️ Diversions", severity['diversions'])
# #         m4.metric("⏱️ Duration",   f"{predicted:.0f} min")

# #         if past_events and avg_actual:
# #             delta = predicted - avg_actual
# #             st.caption(
# #                 f"ℹ️ Model predicts **{abs(delta):.0f} min {'more' if delta > 0 else 'less'}** "
# #                 f"than the historical average actual of {avg_actual:.0f} min."
# #             )

# #         st.success("✅ Logged. Go to Officer Briefing for deployment order.")

# elif page == "🔮 Forecast Event":
#     st.title("🔮 Forecast Event Impact")
#     if models_bundle is None:
#         st.warning("Run `python src/train_pipeline.py` first."); st.stop()
#     lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps = models_bundle
#     zone_opts  = sorted(df_full['zone'].dropna().unique().tolist())  if df_full is not None and 'zone'       in df_full.columns else ['unknown']
#     junc_opts  = sorted(df_full['junction'].dropna().unique().tolist()) if df_full is not None and 'junction'  in df_full.columns else ['unknown']
#     corr_opts  = sorted(df_full['corridor'].dropna().unique().tolist()) if df_full is not None and 'corridor'  in df_full.columns else ['unknown']
#     etype_opts = sorted(df_full['event_type'].dropna().unique().tolist()) if df_full is not None and 'event_type' in df_full.columns else ['festival']
#     ecause_opts= sorted(df_full['event_cause'].dropna().unique().tolist()) if df_full is not None and 'event_cause' in df_full.columns else ['traffic']

#     with st.form("forecast_form"):
#         c1,c2,c3 = st.columns(3)
#         event_type  = c1.selectbox("Event Type",   etype_opts)
#         event_cause = c2.selectbox("Event Cause",  ecause_opts)
#         zone        = c3.selectbox("Zone",         zone_opts)
#         junction    = c1.selectbox("Junction",     junc_opts)
#         corridor    = c2.selectbox("Corridor",     corr_opts)
#         direction   = c3.selectbox("Direction",    ['north','south','east','west','both'])
#         address     = c1.text_input("Address", "Charminar, Hyderabad")
#         start_dt    = c2.date_input("Event Date")
#         start_time  = c3.time_input("Start Time")
#         road_closure= c1.checkbox("Road Closure Required?")
#         has_vehicle = c2.checkbox("Vehicle Involved?")
#         has_cargo   = c3.checkbox("Cargo Present?")
#         submitted   = st.form_submit_button("🔮 Generate Forecast")

#     if submitted:
#         import datetime
#         sdt = datetime.datetime.combine(start_dt, start_time)
#         ev = {'event_type':event_type,'event_cause':event_cause,'zone':zone,
#               'junction':junction,'corridor':corridor,'direction':direction,
#               'address':address,'start_datetime':sdt,'created_date':sdt,
#               'latitude':np.nan,'longitude':np.nan,'endlatitude':np.nan,
#               'endlongitude':np.nan,'resolved_at_latitude':np.nan,
#               'resolved_at_longitude':np.nan,'road_closure_flag':int(road_closure),
#               'has_vehicle':int(has_vehicle),'has_cargo':int(has_cargo),
#               'is_authenticated':1,'veh_type':np.nan,'police_station':np.nan,
#               'cargo_material':None,'reason_breakdown':None,'age_of_truck':np.nan}

#         # Run ML prediction first and log it to get a forecast_id
#         with st.spinner("Running ensemble prediction..."):
#             predicted = predict_duration(ev,lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps)
#         severity = classify_severity(predicted)
#         log_forecast(ev, predicted, severity)

#         # Get the forecast_id just created
#         all_fc = get_all_forecasts()
#         latest_id = int(all_fc.iloc[0]['id']) if not all_fc.empty else None

#         # Memory lookup based on forecast_id
#         # past_events = get_similar_past_events(forecast_id=latest_id, limit=5)

#         # Derive hour and weekend from submitted datetime
#         submitted_hour    = sdt.hour
#         submitted_weekend = 1 if sdt.weekday() >= 5 else 0

#         past_events = get_similar_past_events(
#             event_type=event_type,
#             zone=zone,
#             junction=junction,
#             corridor=corridor,
#             road_closure_flag=int(road_closure),
#             hour=submitted_hour,
#             is_weekend=submitted_weekend,
#             limit=5
#         )

#         # Store everything in session
#         st.session_state['forecast_params']   = {'event_type': event_type, 'zone': zone, 'junction': junction}
#         st.session_state['past_events']       = past_events
#         st.session_state['last_event']        = ev
#         st.session_state['last_severity']     = severity
#         st.session_state['last_predicted']    = predicted
#         st.session_state['forecast_submitted'] = True

#     # ── Render results only after a submission ────────────────────────────────
#     if st.session_state.get('forecast_submitted'):
#         ev        = st.session_state['last_event']
#         severity  = st.session_state['last_severity']
#         predicted = st.session_state['last_predicted']
#         past_events = st.session_state.get('past_events', [])
#         params    = st.session_state.get('forecast_params', {})

#         st.markdown("---")

#         if past_events:
#             st.markdown(f"### 🧠 Institutional Memory — {len(past_events)} Similar Past Event(s) Found")
#             st.caption(
#                 f"Matching **{params.get('event_type','').replace('_',' ').title()}** "
#                 f"in zone **{params.get('zone','')}**. Review before accepting the ML forecast."
#             )
#             avg_actual = np.mean([p['actual_duration_mins'] for p in past_events
#                                   if p.get('actual_duration_mins')])
#             off_issues = sum(1 for p in past_events if p.get('officers_were_sufficient') == 0)
#             bar_issues = sum(1 for p in past_events if p.get('barricades_were_sufficient') == 0)
#             avg_rating = np.mean([p['overall_rating'] for p in past_events
#                                   if p.get('overall_rating')])

#             mc1,mc2,mc3,mc4 = st.columns(4)
#             mc1.metric("Avg Actual Duration (past)", f"{avg_actual:.0f} min" if avg_actual else "—")
#             mc2.metric("Officer shortfall cases",    f"{off_issues}/{len(past_events)}")
#             mc3.metric("Barricade shortfall cases",  f"{bar_issues}/{len(past_events)}")
#             mc4.metric("Avg officer rating",         f"{avg_rating:.1f}/5" if avg_rating else "—")

#             for i, mem in enumerate(past_events):
#                 render_memory_card(mem, i)

#             st.markdown("---")
#             st.markdown("### 🔮 ML Model Forecast")
#             st.caption("Fresh model prediction — compare against historical actuals above.")
#         else:
#             st.info(
#                 "🆕 **No past records found** for this forecast. "
#                 "Prediction is purely ML-based. After the event, submit feedback via "
#                 "**Post-Event Feedback** so future planners benefit."
#             )
#             st.markdown("### 🔮 ML Model Forecast")

#         st.markdown(f"### Predicted Resolution Time: `{predicted:.0f} minutes`")
#         st.markdown(
#             f"<div style='background:{severity['color']}22;border-left:5px solid "
#             f"{severity['color']};padding:14px;border-radius:6px;'>"
#             f"<b>{severity['label']}</b> — {severity['action']}</div>",
#             unsafe_allow_html=True
#         )
#         m1,m2,m3,m4 = st.columns(4)
#         m1.metric("👮 Officers",   severity['officers'])
#         m2.metric("🚧 Barricades", severity['barricades'])
#         m3.metric("↩️ Diversions", severity['diversions'])
#         m4.metric("⏱️ Duration",   f"{predicted:.0f} min")

#         if past_events:
#             avg_actual = np.mean([p['actual_duration_mins'] for p in past_events
#                                   if p.get('actual_duration_mins')])
#             if avg_actual:
#                 delta = predicted - avg_actual
#                 st.caption(
#                     f"ℹ️ Model predicts **{abs(delta):.0f} min {'more' if delta > 0 else 'less'}** "
#                     f"than the historical average actual of {avg_actual:.0f} min."
#                 )

#         st.success("✅ Logged. Go to Officer Briefing for deployment order.")
 
 
# # ═══════════════════════════════════════════════════════════════════════════════
# # PAGE 3: WHAT-IF
# # ═══════════════════════════════════════════════════════════════════════════════
# elif page == "🎛️ What-If Simulator":
#     st.title("🎛️ What-If Simulator")
#     if models_bundle is None:
#         st.warning("Models not trained yet."); st.stop()
#     lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps = models_bundle
#     zone_opts  = sorted(df_full['zone'].dropna().unique().tolist()) if df_full is not None and 'zone' in df_full.columns else ['zone_a']
#     etype_opts = sorted(df_full['event_type'].dropna().unique().tolist()) if df_full is not None and 'event_type' in df_full.columns else ['festival']
#     c1,c2 = st.columns(2)
#     event_type   = c1.selectbox("Event Type", etype_opts)
#     zone         = c2.selectbox("Zone",        zone_opts)
#     hour         = c1.slider("Start Hour", 0, 23, 18)
#     is_weekend   = c1.checkbox("Weekend?", value=True)
#     road_closure = c2.checkbox("Road Closure?")
#     import datetime
#     bdt = datetime.datetime(2024,1,(6 if is_weekend else 2),hour,0,0)
#     ev = {'event_type':event_type,'event_cause':'traffic','zone':zone,
#           'junction':'unknown','corridor':'unknown','direction':'both',
#           'address':'Hyderabad','start_datetime':bdt,'created_date':bdt,
#           'latitude':np.nan,'longitude':np.nan,'endlatitude':np.nan,
#           'endlongitude':np.nan,'resolved_at_latitude':np.nan,
#           'resolved_at_longitude':np.nan,'road_closure_flag':int(road_closure),
#           'has_vehicle':0,'has_cargo':0,'is_authenticated':1,
#           'veh_type':np.nan,'police_station':np.nan,
#           'cargo_material':None,'reason_breakdown':None,'age_of_truck':np.nan}
#     predicted = predict_duration(ev,lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps)
#     severity  = classify_severity(predicted)
#     st.markdown("---")
#     ga,gb = st.columns([1,2])
#     ga.metric("🚦 Predicted Duration", f"{predicted:.0f} min")
#     ga.metric("Severity",              severity['label'])
#     ga.metric("Officers Needed",       severity['officers'])
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number", value=min(predicted,300),
#         gauge={'axis':{'range':[0,300]},'bar':{'color':severity['color']},
#                'steps':[{'range':[0,30],'color':'#E8F5E9'},
#                         {'range':[30,90],'color':'#FFF9C4'},
#                         {'range':[90,180],'color':'#FFE0B2'},
#                         {'range':[180,300],'color':'#FFCDD2'}]},
#         title={'text':'Est. Resolution (min)'}))
#     gb.plotly_chart(fig, use_container_width=True)


# # elif page == "🎛️ What-If Simulator":
# #     st.title("🎛️ What-If Simulator")
# #     if models_bundle is None:
# #         st.warning("Models not trained yet."); st.stop()
# #     lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps = models_bundle
# #     zone_opts  = sorted(df_full['zone'].dropna().unique().tolist()) if df_full is not None and 'zone' in df_full.columns else ['zone_a']
# #     etype_opts = sorted(df_full['event_type'].dropna().unique().tolist()) if df_full is not None and 'event_type' in df_full.columns else ['festival']
# #     corr_opts  = sorted(df_full['corridor'].dropna().unique().tolist()) if df_full is not None and 'corridor' in df_full.columns else ['unknown']
# #     junc_opts  = sorted(df_full['junction'].dropna().unique().tolist()) if df_full is not None and 'junction' in df_full.columns else ['unknown']

# #     st.markdown("### ⚙️ Adjust Parameters")
# #     c1, c2, c3 = st.columns(3)
# #     event_type   = c1.selectbox("Event Type",  etype_opts, key="wi_etype")
# #     zone         = c2.selectbox("Zone",         zone_opts,  key="wi_zone")
# #     junction     = c3.selectbox("Junction",     junc_opts,  key="wi_junc")
# #     corridor     = c1.selectbox("Corridor",     corr_opts,  key="wi_corr")
# #     hour         = c2.slider("Start Hour", 0, 23, 18,       key="wi_hour")
# #     is_weekend   = c3.checkbox("Weekend?", value=True,      key="wi_wknd")
# #     road_closure = c1.checkbox("Road Closure?",             key="wi_rc")
# #     has_vehicle  = c2.checkbox("Vehicle Involved?",         key="wi_veh")
# #     has_cargo    = c3.checkbox("Cargo Present?",            key="wi_cargo")

# #     import datetime
# #     bdt = datetime.datetime(2024, 1, (6 if is_weekend else 2), hour, 0, 0)
# #     ev = {
# #         'event_type': event_type, 'event_cause': 'traffic', 'zone': zone,
# #         'junction': junction, 'corridor': corridor, 'direction': 'both',
# #         'address': 'Hyderabad', 'start_datetime': bdt, 'created_date': bdt,
# #         'latitude': np.nan, 'longitude': np.nan, 'endlatitude': np.nan,
# #         'endlongitude': np.nan, 'resolved_at_latitude': np.nan,
# #         'resolved_at_longitude': np.nan, 'road_closure_flag': int(road_closure),
# #         'has_vehicle': int(has_vehicle), 'has_cargo': int(has_cargo),
# #         'is_authenticated': 1, 'veh_type': np.nan, 'police_station': np.nan,
# #         'cargo_material': None, 'reason_breakdown': None, 'age_of_truck': np.nan
# #     }

# #     predicted = predict_duration(ev, lgb_m, xgb_m, cat_m, le_dict, meta, risk_maps)
# #     severity  = classify_severity(predicted)

# #     st.markdown("---")
# #     st.markdown("### 📊 Live Prediction")

# #     m1, m2, m3, m4 = st.columns(4)
# #     m1.metric("⏱️ Duration",    f"{predicted:.0f} min")
# #     m2.metric("🚨 Severity",    severity['label'])
# #     m3.metric("👮 Officers",    severity['officers'])
# #     m4.metric("🚧 Barricades",  severity['barricades'])

# #     col_gauge, col_bar = st.columns([1, 1])

# #     fig_gauge = go.Figure(go.Indicator(
# #         mode="gauge+number",
# #         value=min(predicted, 300),
# #         gauge={
# #             'axis': {'range': [0, 300]},
# #             'bar':  {'color': severity['color']},
# #             'steps': [
# #                 {'range': [0,   30],  'color': '#E8F5E9'},
# #                 {'range': [30,  90],  'color': '#FFF9C4'},
# #                 {'range': [90,  180], 'color': '#FFE0B2'},
# #                 {'range': [180, 300], 'color': '#FFCDD2'},
# #             ],
# #             'threshold': {
# #                 'line': {'color': 'red', 'width': 4},
# #                 'thickness': 0.75,
# #                 'value': predicted
# #             }
# #         },
# #         title={'text': 'Est. Resolution (min)'}
# #     ))
# #     col_gauge.plotly_chart(fig_gauge, use_container_width=True)

# #     # Sensitivity bar — show impact of each boolean factor
# #     factors = {
# #         'Road Closure': road_closure,
# #         'Weekend':      is_weekend,
# #         'Vehicle':      has_vehicle,
# #         'Cargo':        has_cargo,
# #     }
# #     deltas = {}
# #     for fname, fval in factors.items():
# #         if not fval:
# #             ev_on = {**ev}
# #             key_map = {
# #                 'Road Closure': 'road_closure_flag',
# #                 'Weekend':      None,
# #                 'Vehicle':      'has_vehicle',
# #                 'Cargo':        'has_cargo',
# #             }
# #             k = key_map[fname]
# #             if k:
# #                 ev_on[k] = 1
# #             else:
# #                 alt_dt = datetime.datetime(2024, 1, 6, hour, 0, 0)
# #                 ev_on['start_datetime'] = alt_dt
# #                 ev_on['created_date']   = alt_dt
# #             pred_on = predict_duration(ev_on, lgb_m, xgb_m, cat_m, le_dict, meta, risk_maps)
# #             deltas[fname] = round(pred_on - predicted, 1)
# #         else:
# #             ev_off = {**ev}
# #             key_map = {
# #                 'Road Closure': 'road_closure_flag',
# #                 'Weekend':      None,
# #                 'Vehicle':      'has_vehicle',
# #                 'Cargo':        'has_cargo',
# #             }
# #             k = key_map[fname]
# #             if k:
# #                 ev_off[k] = 0
# #             else:
# #                 alt_dt = datetime.datetime(2024, 1, 2, hour, 0, 0)
# #                 ev_off['start_datetime'] = alt_dt
# #                 ev_off['created_date']   = alt_dt
# #             pred_off = predict_duration(ev_off, lgb_m, xgb_m, cat_m, le_dict, meta, risk_maps)
# #             deltas[fname] = round(predicted - pred_off, 1)

# #     fig_bar = go.Figure(go.Bar(
# #         x=list(deltas.values()),
# #         y=list(deltas.keys()),
# #         orientation='h',
# #         marker_color=['#EF5350' if v > 0 else '#66BB6A' for v in deltas.values()],
# #         text=[f"+{v} min" if v > 0 else f"{v} min" for v in deltas.values()],
# #         textposition='outside'
# #     ))
# #     fig_bar.update_layout(
# #         title="Factor Impact on Duration (vs current toggle state)",
# #         xaxis_title="Minutes added if toggled ON",
# #         height=280, margin=dict(l=10, r=60, t=40, b=10)
# #     )
# #     col_bar.plotly_chart(fig_bar, use_container_width=True)
 
 
# # ═══════════════════════════════════════════════════════════════════════════════
# # PAGE 4: OFFICER BRIEFING
# # ═══════════════════════════════════════════════════════════════════════════════
# elif page == "📋 Officer Briefing":
#     st.title("📋 AI-Generated Officer Deployment Briefing")
#     if 'last_event' not in st.session_state:
#         st.warning("Please forecast an event first."); st.stop()
#     ev  = st.session_state['last_event']
#     sev = st.session_state['last_severity']
#     pred= st.session_state['last_predicted']
#     st.subheader(f"Event: {str(ev.get('event_type','')).title()}")
#     st.caption(f"{ev.get('address','')} | Zone: {ev.get('zone','')} | Junction: {ev.get('junction','')}")
#     c1,c2,c3 = st.columns(3)
#     c1.metric("Severity",   sev['label'])
#     c2.metric("Officers",   sev['officers'])
#     c3.metric("Predicted",  f"{pred:.0f} min")
#     if st.button("📋 Generate Deployment Briefing"):
#         with st.spinner("Generating briefing..."):
#             briefing = generate_briefing(ev, sev, pred)
#         st.markdown("---")
#         st.code(briefing, language=None)
#         st.download_button("📥 Download Briefing (.txt)", data=briefing,
#                        file_name="deployment_briefing.txt", mime="text/plain")
 
 
# # ═══════════════════════════════════════════════════════════════════════════════
# # PAGE 5: HEATMAP
# # ═══════════════════════════════════════════════════════════════════════════════
# elif page == "🗺️ Heatmap":
#     st.title("🗺️ Congestion Heatmap")
#     from streamlit_folium import st_folium
#     if df_full is None:
#         st.error("Dataset not loaded."); st.stop()
#     df = df_full.copy()
#     if 'resolution_duration_mins' not in df.columns:
#         df['resolution_duration_mins'] = 60.0
#     m = generate_heatmap(df, intensity_col='resolution_duration_mins')
#     st_folium(m, width=950, height=520)




#     # Hour sensitivity curve
#     st.markdown("### 📈 Duration by Hour of Day  *(all other params fixed)*")
#     hour_preds = []
#     for h in range(24):
#         ev_h = {**ev, 'start_datetime': datetime.datetime(2024, 1, (6 if is_weekend else 2), h, 0, 0),
#                 'created_date': datetime.datetime(2024, 1, (6 if is_weekend else 2), h, 0, 0)}
#         hour_preds.append(predict_duration(ev_h, lgb_m, xgb_m, cat_m, le_dict, meta, risk_maps))

#     fig_line = go.Figure()
#     fig_line.add_trace(go.Scatter(
#         x=list(range(24)), y=hour_preds, mode='lines+markers',
#         line=dict(color='#1976D2', width=2),
#         marker=dict(size=6),
#         name='Predicted Duration'
#     ))
#     fig_line.add_vline(x=hour, line_dash='dash', line_color='red',
#                        annotation_text=f"Current: {hour}:00")
#     fig_line.update_layout(
#         xaxis=dict(title='Hour of Day', tickvals=list(range(24))),
#         yaxis_title='Predicted Duration (min)',
#         height=300, margin=dict(t=20, b=20)
#     )
#     st.plotly_chart(fig_line, use_container_width=True)
 
 
# # ═══════════════════════════════════════════════════════════════════════════════
# # PAGE 6: POST-EVENT FEEDBACK  (NEW)
# # ═══════════════════════════════════════════════════════════════════════════════
# elif page == "📝 Post-Event Feedback":
#     st.title("📝 Post-Event Feedback")
#     st.markdown(
#         "Fill this form **after an event is resolved**. "
#         "Your inputs are stored as institutional memory and will automatically "
#         "surface when a similar event is forecasted in future."
#     )
#     st.markdown("---")
 
#     # Let officer either link to a logged forecast or fill manually
#     all_fc = get_all_forecasts()
#     use_logged = st.radio(
#         "Link to a logged forecast?",
#         ["Yes — pick from forecast log", "No — enter manually"],
#         horizontal=True
#     )
 
#     forecast_id   = None
#     prefill       = {}
 
#     if use_logged == "Yes — pick from forecast log" and not all_fc.empty:
#         fc_options = {
#             f"#{row['id']} | {row['event_type']} | {row['zone']} | {str(row['forecast_ts'])[:10]}": row
#             for _, row in all_fc.iterrows()
#         }
#         chosen_label = st.selectbox("Select Forecast", list(fc_options.keys()))
#         prefill = fc_options[chosen_label].to_dict()
#         forecast_id = int(prefill.get('id', 0))
#         st.info(f"Pre-filling from forecast #{forecast_id}")
#     elif use_logged == "Yes — pick from forecast log":
#         st.warning("No forecasts logged yet. Fill manually.")
 
#     st.markdown("### 📋 Event Details")
#     col1, col2, col3 = st.columns(3)
 
#     zone_opts_fb  = sorted(df_full['zone'].dropna().unique().tolist())  if df_full is not None and 'zone'       in df_full.columns else []
#     junc_opts_fb  = sorted(df_full['junction'].dropna().unique().tolist()) if df_full is not None and 'junction'  in df_full.columns else []
#     etype_opts_fb = sorted(df_full['event_type'].dropna().unique().tolist()) if df_full is not None and 'event_type' in df_full.columns else []
#     ecause_opts_fb= sorted(df_full['event_cause'].dropna().unique().tolist()) if df_full is not None and 'event_cause' in df_full.columns else []
 
#     def _idx(opts, val):
#         try: return opts.index(str(val).lower()) if str(val).lower() in opts else 0
#         except: return 0
 
#     fb_event_type  = col1.selectbox("Event Type",  etype_opts_fb,
#                                      index=_idx(etype_opts_fb, prefill.get('event_type','')))
#     fb_event_cause = col2.selectbox("Event Cause", ecause_opts_fb,
#                                      index=_idx(ecause_opts_fb, prefill.get('event_cause','')))
#     fb_zone        = col3.selectbox("Zone",        zone_opts_fb,
#                                      index=_idx(zone_opts_fb, prefill.get('zone','')))
#     fb_junction    = col1.selectbox("Junction",    junc_opts_fb,
#                                      index=_idx(junc_opts_fb, prefill.get('junction','')))
#     fb_address     = col2.text_input("Address",    value=str(prefill.get('address','')) if prefill.get('address') else '')
#     fb_event_date  = col3.date_input("Event Date")
#     fb_road_closure= col1.checkbox("Road Closure was active?",
#                                     value=bool(prefill.get('road_closure', 0)))
 
#     st.markdown("### ⏱️ Duration")
#     col4, col5 = st.columns(2)
#     fb_predicted = col4.number_input(
#         "Predicted Duration (min)",
#         min_value=0.0, value=float(prefill.get('predicted_duration_mins', 60) or 60))
#     fb_actual    = col5.number_input(
#         "Actual Duration (min) *",
#         min_value=1.0, value=float(prefill.get('actual_duration_mins', 60) or 60))
 
#     st.markdown("### 👮 Resources")
#     col6, col7, col8, col9 = st.columns(4)
#     fb_officers_dep  = col6.number_input("Officers Deployed", min_value=0,
#                                           value=int(prefill.get('officers', 5) or 5))
#     fb_officers_ok   = col7.radio("Officers Sufficient?",    ["Yes", "No"], horizontal=True)
#     fb_barricades    = col8.number_input("Barricades Deployed", min_value=0,
#                                           value=int(prefill.get('barricades', 0) or 0))
#     fb_barricades_ok = col9.radio("Barricades Sufficient?", ["Yes", "No"], horizontal=True)
 
#     st.markdown("### ↩️ Diversions")
#     col10, col11 = st.columns(2)
#     fb_diversion_routes = col10.text_area(
#         "Diversion Routes Used (describe briefly)",
#         placeholder="e.g. Rerouted via MG Road, alternate via Tank Bund..."
#     )
#     fb_diversion_eff = col11.radio(
#         "Were diversions effective?", ["Yes", "No", "Not Applicable"], horizontal=True)
#     div_eff_map = {"Yes": 1, "No": 0, "Not Applicable": -1}
 
#     st.markdown("### 🌍 Context")
#     col12, col13 = st.columns(2)
#     fb_crowd    = col12.selectbox("Crowd Density",   ["low","medium","high","very_high"])
#     fb_weather  = col13.selectbox("Weather Impact",  ["none","mild","significant"])
 
#     st.markdown("### 📝 Learnings  *(most important — this is what future planners will read)*")
#     fb_worked   = st.text_area("✅ What worked well?",
#                                 placeholder="e.g. Early barricading at Charminar junction worked perfectly. Officers at both ends managed flow well.")
#     fb_didnt    = st.text_area("❌ What didn't work / what went wrong?",
#                                 placeholder="e.g. Not enough officers for crowd spill on south side. Diversion signs not visible at night.")
#     fb_unexpected = st.text_area("⚠️ Unexpected factors encountered?",
#                                   placeholder="e.g. Second procession merged unexpectedly. Rain caused crowd shelter at junction.")
 
#     st.markdown("### 🏁 Summary")
#     col14, col15 = st.columns(2)
#     fb_rating      = col14.slider("Overall operation rating (1=poor, 5=excellent)", 1, 5, 3)
#     fb_submitted_by= col15.text_input("Officer / Submitter name (optional)")
 
#     st.markdown("---")
#     if st.button("💾 Save Post-Event Feedback", type="primary"):
#         if not fb_worked and not fb_didnt:
#             st.warning("Please fill in at least one learning (what worked or what didn't).")
#         else:
#             memory = {
#                 'forecast_id':                  forecast_id,
#                 'event_type':                   fb_event_type,
#                 'event_cause':                  fb_event_cause,
#                 'zone':                         fb_zone,
#                 'junction':                     fb_junction,
#                 'corridor':                     prefill.get('corridor',''),
#                 'address':                      fb_address,
#                 'event_date':                   str(fb_event_date),
#                 'predicted_duration_mins':      fb_predicted,
#                 'actual_duration_mins':         fb_actual,
#                 'officers_deployed':            fb_officers_dep,
#                 'officers_were_sufficient':     1 if fb_officers_ok == "Yes" else 0,
#                 'barricades_deployed':          fb_barricades,
#                 'barricades_were_sufficient':   1 if fb_barricades_ok == "Yes" else 0,
#                 'diversion_routes_used':        fb_diversion_routes,
#                 'diversion_effective':          div_eff_map[fb_diversion_eff],
#                 'road_closure_flag':            int(fb_road_closure),
#                 'what_worked_well':             fb_worked,
#                 'what_didnt_work':              fb_didnt,
#                 'crowd_density':                fb_crowd,
#                 'weather_impact':               fb_weather,
#                 'unexpected_factors':           fb_unexpected,
#                 'overall_rating':               fb_rating,
#                 'submitted_by':                 fb_submitted_by,
#             }
#             # Also update actual duration in forecasts table if linked
#             if forecast_id:
#                 update_actual(forecast_id, fb_actual)
 
#             new_id = log_event_memory(memory)
#             st.success(f"✅ Feedback saved as Memory #{new_id}. Future forecasts for "
#                        f"**{fb_event_type.replace('_',' ').title()}** in **{fb_zone}** will surface this.")
#             st.balloons()
 
 
# # ═══════════════════════════════════════════════════════════════════════════════
# # PAGE 7: POST-EVENT ANALYTICS  (enhanced from original)
# # ═══════════════════════════════════════════════════════════════════════════════
# elif page == "📈 Post-Event Analytics":
#     st.title("📈 Post-Event Analytics")
 
#     tab1, tab2 = st.tabs(["📊 Forecast Accuracy", "🧠 Institutional Memory"])
 
#     # ── Tab 1: Original accuracy analytics ───────────────────────────────────
#     with tab1:
#         df_log = get_all_forecasts()
#         if df_log.empty:
#             st.info("No forecasts logged yet."); 
#         else:
#             st.dataframe(df_log, use_container_width=True)
#             with st.form("actual_form"):
#                 fid         = st.number_input("Forecast ID to update", min_value=1, step=1)
#                 actual_mins = st.number_input("Actual Duration (minutes)", min_value=1.0)
#                 if st.form_submit_button("Update Actual"):
#                     update_actual(int(fid), actual_mins)
#                     st.success(f"Updated #{fid}")
#             acc_df = get_accuracy_df()
#             if not acc_df.empty:
#                 st.metric("Mean Absolute Error", f"{acc_df['abs_error'].mean():.1f} min")
#                 fig = px.scatter(acc_df, x='predicted_duration_mins', y='actual_duration_mins',
#                                  color='event_type', title='Predicted vs Actual Duration')
#                 mx = max(acc_df[['predicted_duration_mins','actual_duration_mins']].max())
#                 fig.add_shape(type='line',x0=0,y0=0,x1=mx,y1=mx,
#                               line=dict(dash='dash',color='gray'))
#                 st.plotly_chart(fig, use_container_width=True)
 
#     # ── Tab 2: Institutional Memory analytics ─────────────────────────────────
#     with tab2:
#         mem_df = get_all_event_memories()
#         if mem_df.empty:
#             st.info("No post-event feedback recorded yet. Use **📝 Post-Event Feedback** to add entries.")
#         else:
#             # KPI row
#             k1,k2,k3,k4,k5 = st.columns(5)
#             k1.metric("Total Memories",         len(mem_df))
#             k2.metric("Avg Rating",             f"{mem_df['overall_rating'].mean():.1f}/5"
#                        if mem_df['overall_rating'].notna().any() else "—")
#             off_suff = mem_df['officers_were_sufficient'].mean()
#             k3.metric("Officer Sufficiency",    f"{off_suff*100:.0f}%" if not np.isnan(off_suff) else "—")
#             bar_suff = mem_df['barricades_were_sufficient'].mean()
#             k4.metric("Barricade Sufficiency",  f"{bar_suff*100:.0f}%" if not np.isnan(bar_suff) else "—")
#             err_series = (mem_df['actual_duration_mins'] - mem_df['predicted_duration_mins']).dropna().abs()
#             k5.metric("Mean Actual Error",      f"{err_series.mean():.0f} min" if len(err_series) > 0 else "—")
 
#             st.markdown("---")
 
#             col_a, col_b = st.columns(2)
 
#             # Events by type
#             if 'event_type' in mem_df.columns:
#                 et_cnt = mem_df['event_type'].value_counts().head(12)
#                 fig_et = px.bar(et_cnt, orientation='h',
#                                 title="Memory Count by Event Type",
#                                 color=et_cnt.values, color_continuous_scale='Blues')
#                 fig_et.update_layout(showlegend=False, height=340)
#                 col_a.plotly_chart(fig_et, use_container_width=True)
 
#             # Rating distribution
#             if mem_df['overall_rating'].notna().any():
#                 fig_rat = px.histogram(mem_df.dropna(subset=['overall_rating']),
#                                        x='overall_rating', nbins=5,
#                                        title="Operation Rating Distribution",
#                                        color_discrete_sequence=['#4CAF50'])
#                 col_b.plotly_chart(fig_rat, use_container_width=True)
 
#             # Predicted vs Actual scatter from memory
#             valid_mem = mem_df.dropna(subset=['predicted_duration_mins','actual_duration_mins'])
#             if not valid_mem.empty:
#                 fig_sc = px.scatter(valid_mem,
#                                     x='predicted_duration_mins', y='actual_duration_mins',
#                                     color='event_type', symbol='zone',
#                                     title="Predicted vs Actual (from Field Feedback)",
#                                     hover_data=['zone','junction','crowd_density','overall_rating'])
#                 mx = max(valid_mem[['predicted_duration_mins','actual_duration_mins']].max())
#                 fig_sc.add_shape(type='line',x0=0,y0=0,x1=mx,y1=mx,
#                                  line=dict(dash='dash',color='gray'))
#                 st.plotly_chart(fig_sc, use_container_width=True)
 
#             # Officer sufficiency by event type
#             off_by_type = mem_df.groupby('event_type')['officers_were_sufficient'].mean().sort_values()
#             if not off_by_type.empty:
#                 fig_off = px.bar(off_by_type, orientation='h',
#                                  title="Officer Sufficiency Rate by Event Type",
#                                  color=off_by_type.values, color_continuous_scale='RdYlGn',
#                                  range_color=[0,1])
#                 fig_off.update_layout(height=320)
#                 st.plotly_chart(fig_off, use_container_width=True)
 
#             st.markdown("### 📋 All Feedback Records")
#             st.dataframe(mem_df, use_container_width=True)
 


#----------------------------------------------new---------------------------------------

"""
streamlit_app.py — TrafficSense: Event-Driven Congestion Forecaster
6 pages + Event Memory System (similarity-based past event lookup + feedback)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys, os, warnings
warnings.filterwarnings('ignore')
 
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
 
from preprocess import load_and_clean, build_targets, build_features, CATEGORICAL_FEATURES
from model import load_artifacts, predict_duration, classify_severity, encode_categoricals
from database import (init_db, log_forecast, update_actual, get_all_forecasts,
                      get_accuracy_df, log_event_memory, get_all_memory,
                      find_similar_events, delete_forecast, delete_memory)
from heatmap import generate_heatmap
from llm_briefing import generate_briefing, generate_post_event_report
 
st.set_page_config(page_title="TrafficSense", page_icon="🚦", layout="wide")
init_db()
 
@st.cache_resource(show_spinner="Loading trained models...")
def get_models():
    try:
        return load_artifacts()
    except Exception:
        return None
 
@st.cache_data(show_spinner="Loading dataset...")
def get_dataset():
    try:
        df = load_and_clean('data/raw/astram_event_data.csv')
        df = build_targets(df)
        df = build_features(df, is_training=True)
        return df
    except Exception:
        return None
 
models_bundle = get_models()
df_full = get_dataset()
 
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🚦 TrafficSense")
st.sidebar.caption("Event-Driven Congestion Forecaster")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "🔮 Forecast Event",
    "🎛️ What-If Simulator",
    "📋 Officer Briefing",
    "🗺️ Heatmap",
    "📈 Post-Event Analytics",
    "🧠 Event Feedback & Memory"
])
model_status = "✅ Models Loaded" if models_bundle else "⚠️ Train models first"
data_status  = f"✅ {len(df_full)} events" if df_full is not None else "⚠️ No dataset"
st.sidebar.markdown(f"**Status:**  \n{model_status}  \n{data_status}")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1: DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("📊 Event Congestion Overview")
    if df_full is None:
        st.error("Place CSV at data/raw/astram_event_data.csv"); st.stop()
    df = df_full.copy()
 
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Events", f"{len(df):,}")
    if 'road_closure_flag' in df.columns:
        c2.metric("Road Closures", f"{df['road_closure_flag'].sum():,}")
    if 'resolution_duration_mins' in df.columns:
        v = df[df['resolution_duration_mins'].between(0,1440)]
        c3.metric("Avg Resolution",    f"{v['resolution_duration_mins'].mean():.0f} min")
        c4.metric("Median Resolution", f"{v['resolution_duration_mins'].median():.0f} min")
    if 'zone' in df.columns:
        c5.metric("Unique Zones", df['zone'].nunique())
    st.markdown("---")
 
    col1, col2 = st.columns(2)
    if 'event_type' in df.columns:
        et = df['event_type'].value_counts().head(15)
        fig1 = px.bar(x=et.values, y=et.index, orientation='h',
                      title="Top 15 Event Types", color=et.values,
                      color_continuous_scale='RdYlGn_r')
        fig1.update_layout(showlegend=False, height=380)
        col1.plotly_chart(fig1, use_container_width=True)
 
    if 'event_type' in df.columns and 'resolution_duration_mins' in df.columns:
        v = df[df['resolution_duration_mins'].between(0,1440)]
        dur = v.groupby('event_type')['resolution_duration_mins'].median().sort_values().tail(15)
        fig2 = px.bar(x=dur.values, y=dur.index, orientation='h',
                      title="Median Duration by Event Type (min)", color=dur.values,
                      color_continuous_scale='YlOrRd')
        fig2.update_layout(showlegend=False, height=380)
        col2.plotly_chart(fig2, use_container_width=True)
 
    if all(c in df.columns for c in ['day_of_week','hour','resolution_duration_mins']):
        v = df[df['resolution_duration_mins'].between(0,1440)]
        pivot = v.pivot_table(values='resolution_duration_mins',
                               index='day_of_week', columns='hour', aggfunc='median')
        pivot.index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][:len(pivot)]
        fig3 = go.Figure(go.Heatmap(z=pivot.values,
            x=[str(h) for h in pivot.columns], y=pivot.index.tolist(),
            colorscale='YlOrRd', colorbar=dict(title='Min')))
        fig3.update_layout(title='Median Duration: Day × Hour',
                           xaxis_title='Hour', yaxis_title='Day', height=320)
        st.plotly_chart(fig3, use_container_width=True)
 
    show_cols = [c for c in ['event_type','event_cause','zone','junction',
                              'corridor','status','priority',
                              'resolution_duration_mins','road_closure_flag']
                 if c in df.columns]
    st.dataframe(df[show_cols].head(50), use_container_width=True)

# ── CORRIDOR RISK RANKING ─────────────────────────────────────
    st.markdown("---")
    st.subheader("🛣️ Corridor Risk Ranking")
    st.caption("Which corridors are structurally the most dangerous — regardless of specific event.")

    if all(c in df.columns for c in ['corridor', 'resolution_duration_mins', 'road_closure_flag']):
        valid = df[df['resolution_duration_mins'].between(0, 1440)].copy()

        corridor_risk = valid.groupby('corridor').agg(
            Incidents         = ('corridor', 'count'),
            Avg_Resolution    = ('resolution_duration_mins', 'mean'),
            Road_Closure_Rate = ('road_closure_flag', 'mean')
        ).reset_index()

        corridor_risk = corridor_risk[corridor_risk['Incidents'] >= 5]
        corridor_risk = corridor_risk.sort_values('Incidents', ascending=False).reset_index(drop=True)
        corridor_risk.index += 1

        corridor_risk['Avg_Resolution']    = corridor_risk['Avg_Resolution'].round(0).astype(int).astype(str) + ' mins'
        corridor_risk['Road_Closure_Rate'] = (corridor_risk['Road_Closure_Rate'] * 100).round(0).astype(int).astype(str) + '%'
        corridor_risk.columns = ['Corridor', 'Incidents', 'Avg Resolution', 'Road Closure Rate']
        corridor_risk.index.name = 'Rank'

        st.dataframe(corridor_risk.head(15), use_container_width=True)

        worst = corridor_risk.iloc[0]
        st.info(
            f"⚠️ **{worst['Corridor'].title()}** needs permanent pre-positioned resources — "
            f"it has the most incidents ({worst['Incidents']}), "
            f"takes {worst['Avg Resolution']} on average to clear, "
            f"and has a road closure rate of {worst['Road Closure Rate']}."
        )
    else:
        st.warning("Corridor or duration data not available in dataset.")





# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2: FORECAST EVENT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮 Forecast Event":
    st.title("🔮 Forecast Event Impact")
 
    if models_bundle is None:
        st.warning("Run `python train_pipeline.py` first."); st.stop()
 
    lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps = models_bundle
 
    zone_opts  = sorted(df_full['zone'].dropna().unique().tolist())       if df_full is not None and 'zone'       in df_full.columns else ['unknown']
    junc_opts  = sorted(df_full['junction'].dropna().unique().tolist())   if df_full is not None and 'junction'   in df_full.columns else ['unknown']
    corr_opts  = sorted(df_full['corridor'].dropna().unique().tolist())   if df_full is not None and 'corridor'   in df_full.columns else ['unknown']
    etype_opts = sorted(df_full['event_type'].dropna().unique().tolist()) if df_full is not None and 'event_type' in df_full.columns else ['festival']
    ecause_opts= sorted(df_full['event_cause'].dropna().unique().tolist())if df_full is not None and 'event_cause'in df_full.columns else ['traffic']
 
    with st.form("forecast_form"):
        c1,c2,c3 = st.columns(3)
        event_type  = c1.selectbox("Event Type",   etype_opts)
        event_cause = c2.selectbox("Event Cause",  ecause_opts)
        zone        = c3.selectbox("Zone",         zone_opts)
        junction    = c1.selectbox("Junction",     junc_opts)
        corridor    = c2.selectbox("Corridor",     corr_opts)
        direction   = c3.selectbox("Direction",    ['north','south','east','west','both'])
        address     = c1.text_input("Address", "Charminar, Hyderabad")
        start_dt    = c2.date_input("Event Date")
        start_time  = c3.time_input("Start Time")
        road_closure= c1.checkbox("Road Closure Required?")
        has_vehicle = c2.checkbox("Vehicle Involved?")
        has_cargo   = c3.checkbox("Cargo Present?")
        submitted   = st.form_submit_button("🔮 Generate Forecast")
 
    if submitted:
        import datetime
        sdt = datetime.datetime.combine(start_dt, start_time)
        hour       = sdt.hour
        dow        = sdt.weekday()
 
        ev = {
            'event_type':event_type,'event_cause':event_cause,'zone':zone,
            'junction':junction,'corridor':corridor,'direction':direction,
            'address':address,'start_datetime':sdt,'created_date':sdt,
            'latitude':np.nan,'longitude':np.nan,'endlatitude':np.nan,
            'endlongitude':np.nan,'resolved_at_latitude':np.nan,
            'resolved_at_longitude':np.nan,'road_closure_flag':int(road_closure),
            'has_vehicle':int(has_vehicle),'has_cargo':int(has_cargo),
            'is_authenticated':1,'veh_type':np.nan,'police_station':np.nan,
            'cargo_material':None,'reason_breakdown':None,'age_of_truck':np.nan
        }
 
        # ── PAST SIMILAR EVENTS LOOKUP ────────────────────────────
        similar = find_similar_events(event_type, zone, hour, dow, top_n=3)
 
        if not similar.empty:
            st.markdown("---")
            st.subheader("🧠 Similar Past Events Found")
            st.caption("The system found past events that match your input. "
                       "Review what happened before making deployment decisions.")
 
            for _, row in similar.iterrows():
                score = int(row.get('similarity_score', 0))
                match_pct = min(100, score)
                with st.expander(
                    f"📁 {str(row.get('event_type','')).title()} | "
                    f"Zone: {row.get('zone','?')} | "
                    f"Actual: {row.get('actual_duration_mins','?'):.0f} min | "
                    f"Match: {match_pct}%",
                    expanded=True
                ):
                    mc1, mc2, mc3, mc4 = st.columns(4)
                    mc1.metric("Predicted (then)", f"{row.get('predicted_duration_mins',0):.0f} min")
                    mc2.metric("Actual (then)",    f"{row.get('actual_duration_mins',0):.0f} min")
                    mc3.metric("Officers Used",    row.get('officers_actual', row.get('officers_recommended','?')))
                    mc4.metric("Outcome",          row.get('outcome','Unknown'))
 
                    if row.get('what_worked'):
                        st.success(f"✅ **What worked:** {row['what_worked']}")
                    if row.get('what_didnt_work'):
                        st.error(f"❌ **What didn't work:** {row['what_didnt_work']}")
                    if row.get('unexpected_issues'):
                        st.warning(f"⚠️ **Unexpected issues:** {row['unexpected_issues']}")
                    if row.get('officer_notes'):
                        st.info(f"📝 **Officer notes:** {row['officer_notes']}")
                    if row.get('plan_followed'):
                        st.write(f"**Plan followed:** {row['plan_followed']}")
        else:
            st.info("🔍 No similar past events found in memory. "
                    "After this event, submit feedback so future forecasts can learn from it.")
 
        # ── ML PREDICTION ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔮 ML Prediction")
 
        with st.spinner("Running ensemble prediction..."):
            predicted = predict_duration(ev, lgb_m, xgb_m, cat_m,
                                         le_dict, meta, risk_maps)
        severity = classify_severity(predicted)
 
        st.markdown(f"### Predicted Resolution Time: `{predicted:.0f} minutes`")
        st.markdown(
            f"<div style='background:{severity['color']}22;border-left:5px solid "
            f"{severity['color']};padding:14px;border-radius:6px;'>"
            f"<b>{severity['label']}</b> — {severity['action']}</div>",
            unsafe_allow_html=True)
 
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("👮 Officers",   severity['officers'])
        m2.metric("🚧 Barricades", severity['barricades'])
        m3.metric("↩️ Diversions", severity['diversions'])
        m4.metric("⏱️ Duration",   f"{predicted:.0f} min")
 
        # Store in session for briefing page and feedback
        st.session_state.update({
            'last_event':     ev,
            'last_severity':  severity,
            'last_predicted': predicted,
            'last_forecast_id': log_forecast(ev, predicted, severity)
        })
        st.success("✅ Forecast logged. Go to Officer Briefing or submit feedback after the event.")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3: WHAT-IF SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
# elif page == "🎛️ What-If Simulator":
#     st.title("🎛️ What-If Simulator")
#     st.caption("Adjust parameters and see forecast update in real-time.")
 
#     if models_bundle is None:
#         st.warning("Models not trained yet."); st.stop()
 
#     lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps = models_bundle
 
#     zone_opts  = sorted(df_full['zone'].dropna().unique().tolist())       if df_full is not None and 'zone'       in df_full.columns else ['zone_a']
#     etype_opts = sorted(df_full['event_type'].dropna().unique().tolist()) if df_full is not None and 'event_type' in df_full.columns else ['festival']
 
#     c1,c2 = st.columns(2)
#     event_type   = c1.selectbox("Event Type", etype_opts)
#     zone         = c2.selectbox("Zone",        zone_opts)
#     hour         = c1.slider("Start Hour", 0, 23, 18)
#     is_weekend   = c1.checkbox("Weekend?", value=True)
#     road_closure = c2.checkbox("Road Closure?")
 
#     import datetime
#     bdt = datetime.datetime(2024,1,(6 if is_weekend else 2),hour,0,0)
#     ev = {
#         'event_type':event_type,'event_cause':'traffic','zone':zone,
#         'junction':'unknown','corridor':'unknown','direction':'both',
#         'address':'Hyderabad','start_datetime':bdt,'created_date':bdt,
#         'latitude':np.nan,'longitude':np.nan,'endlatitude':np.nan,
#         'endlongitude':np.nan,'resolved_at_latitude':np.nan,
#         'resolved_at_longitude':np.nan,'road_closure_flag':int(road_closure),
#         'has_vehicle':0,'has_cargo':0,'is_authenticated':1,
#         'veh_type':np.nan,'police_station':np.nan,
#         'cargo_material':None,'reason_breakdown':None,'age_of_truck':np.nan
#     }
 
#     predicted = predict_duration(ev, lgb_m, xgb_m, cat_m, le_dict, meta, risk_maps)
#     severity  = classify_severity(predicted)
 
#     st.markdown("---")
#     ga, gb = st.columns([1,2])
#     ga.metric("🚦 Predicted Duration", f"{predicted:.0f} min")
#     ga.metric("Severity",              severity['label'])
#     ga.metric("Officers Needed",       severity['officers'])
 
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number", value=min(predicted,300),
#         gauge={
#             'axis':  {'range':[0,300]},
#             'bar':   {'color': severity['color']},
#             'steps': [
#                 {'range':[0,  30],  'color':'#E8F5E9'},
#                 {'range':[30, 90],  'color':'#FFF9C4'},
#                 {'range':[90, 180], 'color':'#FFE0B2'},
#                 {'range':[180,300], 'color':'#FFCDD2'}
#             ]
#         },
#         title={'text':'Est. Resolution (min)'}
#     ))
#     gb.plotly_chart(fig, use_container_width=True)


elif page == "🎛️ What-If Simulator":
    st.title("🎛️ What-If Simulator")
    if models_bundle is None:
        st.warning("Models not trained yet."); st.stop()
    lgb_m,xgb_m,cat_m,le_dict,meta,risk_maps = models_bundle
    zone_opts  = sorted(df_full['zone'].dropna().unique().tolist()) if df_full is not None and 'zone' in df_full.columns else ['zone_a']
    etype_opts = sorted(df_full['event_type'].dropna().unique().tolist()) if df_full is not None and 'event_type' in df_full.columns else ['festival']
    corr_opts  = sorted(df_full['corridor'].dropna().unique().tolist()) if df_full is not None and 'corridor' in df_full.columns else ['unknown']
    junc_opts  = sorted(df_full['junction'].dropna().unique().tolist()) if df_full is not None and 'junction' in df_full.columns else ['unknown']

    st.markdown("### ⚙️ Adjust Parameters")
    c1, c2, c3 = st.columns(3)
    event_type   = c1.selectbox("Event Type",  etype_opts, key="wi_etype")
    zone         = c2.selectbox("Zone",         zone_opts,  key="wi_zone")
    junction     = c3.selectbox("Junction",     junc_opts,  key="wi_junc")
    corridor     = c1.selectbox("Corridor",     corr_opts,  key="wi_corr")
    hour         = c2.slider("Start Hour", 0, 23, 18,       key="wi_hour")
    is_weekend   = c3.checkbox("Weekend?", value=True,      key="wi_wknd")
    road_closure = c1.checkbox("Road Closure?",             key="wi_rc")
    has_vehicle  = c2.checkbox("Vehicle Involved?",         key="wi_veh")
    has_cargo    = c3.checkbox("Cargo Present?",            key="wi_cargo")

    import datetime
    bdt = datetime.datetime(2024, 1, (6 if is_weekend else 2), hour, 0, 0)
    ev = {
        'event_type': event_type, 'event_cause': 'traffic', 'zone': zone,
        'junction': junction, 'corridor': corridor, 'direction': 'both',
        'address': 'Hyderabad', 'start_datetime': bdt, 'created_date': bdt,
        'latitude': np.nan, 'longitude': np.nan, 'endlatitude': np.nan,
        'endlongitude': np.nan, 'resolved_at_latitude': np.nan,
        'resolved_at_longitude': np.nan, 'road_closure_flag': int(road_closure),
        'has_vehicle': int(has_vehicle), 'has_cargo': int(has_cargo),
        'is_authenticated': 1, 'veh_type': np.nan, 'police_station': np.nan,
        'cargo_material': None, 'reason_breakdown': None, 'age_of_truck': np.nan
    }

    predicted = predict_duration(ev, lgb_m, xgb_m, cat_m, le_dict, meta, risk_maps)
    severity  = classify_severity(predicted)

    st.markdown("---")
    st.markdown("### 📊 Live Prediction")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("⏱️ Duration",    f"{predicted:.0f} min")
    m2.metric("🚨 Severity",    severity['label'])
    m3.metric("👮 Officers",    severity['officers'])
    m4.metric("🚧 Barricades",  severity['barricades'])

    col_gauge, col_bar = st.columns([1, 1])

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min(predicted, 300),
        gauge={
            'axis': {'range': [0, 300]},
            'bar':  {'color': severity['color']},
            'steps': [
                {'range': [0,   30],  'color': '#E8F5E9'},
                {'range': [30,  90],  'color': '#FFF9C4'},
                {'range': [90,  180], 'color': '#FFE0B2'},
                {'range': [180, 300], 'color': '#FFCDD2'},
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': predicted
            }
        },
        title={'text': 'Est. Resolution (min)'}
    ))
    col_gauge.plotly_chart(fig_gauge, use_container_width=True)

    # Sensitivity bar — show impact of each boolean factor
    factors = {
        'Road Closure': road_closure,
        'Weekend':      is_weekend,
        'Vehicle':      has_vehicle,
        'Cargo':        has_cargo,
    }
    deltas = {}
    for fname, fval in factors.items():
        if not fval:
            ev_on = {**ev}
            key_map = {
                'Road Closure': 'road_closure_flag',
                'Weekend':      None,
                'Vehicle':      'has_vehicle',
                'Cargo':        'has_cargo',
            }
            k = key_map[fname]
            if k:
                ev_on[k] = 1
            else:
                alt_dt = datetime.datetime(2024, 1, 6, hour, 0, 0)
                ev_on['start_datetime'] = alt_dt
                ev_on['created_date']   = alt_dt
            pred_on = predict_duration(ev_on, lgb_m, xgb_m, cat_m, le_dict, meta, risk_maps)
            deltas[fname] = round(pred_on - predicted, 1)
        else:
            ev_off = {**ev}
            key_map = {
                'Road Closure': 'road_closure_flag',
                'Weekend':      None,
                'Vehicle':      'has_vehicle',
                'Cargo':        'has_cargo',
            }
            k = key_map[fname]
            if k:
                ev_off[k] = 0
            else:
                alt_dt = datetime.datetime(2024, 1, 2, hour, 0, 0)
                ev_off['start_datetime'] = alt_dt
                ev_off['created_date']   = alt_dt
            pred_off = predict_duration(ev_off, lgb_m, xgb_m, cat_m, le_dict, meta, risk_maps)
            deltas[fname] = round(predicted - pred_off, 1)

    fig_bar = go.Figure(go.Bar(
        x=list(deltas.values()),
        y=list(deltas.keys()),
        orientation='h',
        marker_color=['#EF5350' if v > 0 else '#66BB6A' for v in deltas.values()],
        text=[f"+{v} min" if v > 0 else f"{v} min" for v in deltas.values()],
        textposition='outside'
    ))
    fig_bar.update_layout(
        title="Factor Impact on Duration (vs current toggle state)",
        xaxis_title="Minutes added if toggled ON",
        height=280, margin=dict(l=10, r=60, t=40, b=10)
    )
    col_bar.plotly_chart(fig_bar, use_container_width=True)

    # Hour sensitivity curve
    st.markdown("### 📈 Duration by Hour of Day  *(all other params fixed)*")
    hour_preds = []
    for h in range(24):
        ev_h = {**ev, 'start_datetime': datetime.datetime(2024, 1, (6 if is_weekend else 2), h, 0, 0),
                'created_date': datetime.datetime(2024, 1, (6 if is_weekend else 2), h, 0, 0)}
        hour_preds.append(predict_duration(ev_h, lgb_m, xgb_m, cat_m, le_dict, meta, risk_maps))

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=list(range(24)), y=hour_preds, mode='lines+markers',
        line=dict(color='#1976D2', width=2),
        marker=dict(size=6),
        name='Predicted Duration'
    ))
    fig_line.add_vline(x=hour, line_dash='dash', line_color='red',
                       annotation_text=f"Current: {hour}:00")
    fig_line.update_layout(
        xaxis=dict(title='Hour of Day', tickvals=list(range(24))),
        yaxis_title='Predicted Duration (min)',
        height=300, margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_line, use_container_width=True)
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4: OFFICER BRIEFING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Officer Briefing":
    st.title("📋 Officer Deployment Briefing")
 
    if 'last_event' not in st.session_state:
        st.warning("Please forecast an event first."); st.stop()
 
    ev   = st.session_state['last_event']
    sev  = st.session_state['last_severity']
    pred = st.session_state['last_predicted']
 
    st.subheader(f"Event: {str(ev.get('event_type','')).title()}")
    st.caption(f"{ev.get('address','')} | Zone: {ev.get('zone','')} | Junction: {ev.get('junction','')}")
 
    c1,c2,c3 = st.columns(3)
    c1.metric("Severity",  sev['label'])
    c2.metric("Officers",  sev['officers'])
    c3.metric("Predicted", f"{pred:.0f} min")
 
    if st.button("📋 Generate Deployment Briefing"):
        with st.spinner("Generating briefing..."):
            briefing = generate_briefing(ev, sev, pred)
        st.markdown("---")
        st.code(briefing, language=None)
        st.download_button("📥 Download Briefing (.txt)", data=briefing,
                           file_name="deployment_briefing.txt", mime="text/plain")
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5: HEATMAP
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗺️ Heatmap":
    st.title("🗺️ Congestion Heatmap")
    from streamlit_folium import st_folium
 
    if df_full is None:
        st.error("Dataset not loaded."); st.stop()
 
    df = df_full.copy()
    if 'resolution_duration_mins' not in df.columns:
        df['resolution_duration_mins'] = 60.0
 
    m = generate_heatmap(df, intensity_col='resolution_duration_mins')
    st_folium(m, width=950, height=520)


 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6: POST-EVENT ANALYTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈 Post-Event Analytics":
    st.title("📈 Post-Event Learning Analytics")
 
    df_log = get_all_forecasts()
    if df_log.empty:
        st.info("No forecasts logged yet."); st.stop()
 
    st.subheader("All Logged Forecasts")
    # st.dataframe(df_log, use_container_width=True)
    st.dataframe(df_log, use_container_width=True)

    st.subheader("🗑️ Delete Forecast Record")
    with st.form("delete_forecast_form"):
        del_id = st.number_input("Forecast ID to delete", min_value=1, step=1)
        if st.form_submit_button("🗑️ Delete", type="primary"):
            delete_forecast(int(del_id))
            st.success(f"✅ Forecast #{del_id} deleted.")
            st.rerun()

    with st.form("actual_form"):
        fid         = st.number_input("Forecast ID to update", min_value=1, step=1)
        actual_mins = st.number_input("Actual Duration (minutes)", min_value=1.0)
        if st.form_submit_button("Update Actual"):
            update_actual(int(fid), actual_mins)
            st.success(f"Updated #{fid}")
 
    acc_df = get_accuracy_df()
    if not acc_df.empty:
        st.metric("Mean Absolute Error", f"{acc_df['abs_error'].mean():.1f} min")
        fig = px.scatter(acc_df, x='predicted_duration_mins', y='actual_duration_mins',
                         color='event_type', title='Predicted vs Actual Duration')
        mx = max(acc_df[['predicted_duration_mins','actual_duration_mins']].max())
        fig.add_shape(type='line',x0=0,y0=0,x1=mx,y1=mx,
                      line=dict(dash='dash',color='gray'))
        st.plotly_chart(fig, use_container_width=True)
 
        if st.button("📋 Generate Post-Event Report"):
            row = acc_df.iloc[-1]
            report = generate_post_event_report(
                {'event_type': row.get('event_type','unknown')},
                row['predicted_duration_mins'], row['actual_duration_mins']
            )
            st.code(report, language=None)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# PAGE 7: EVENT FEEDBACK & MEMORY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🧠 Event Feedback & Memory":
    st.title("🧠 Event Feedback & Memory")
    st.caption("Submit post-event feedback so the system learns from every event "
               "and uses that knowledge when similar events occur in the future.")
 
    # ── FEEDBACK FORM ─────────────────────────────────────────────
    st.subheader("📝 Submit Post-Event Feedback")
    st.info("Fill this form after an event ends. This data is used to improve "
            "future deployment plans for similar events.")
 
    df_log = get_all_forecasts()
 
    with st.form("feedback_form"):
 
        # Link to a logged forecast
        st.markdown("**Step 1 — Link to Forecast**")
        if not df_log.empty:
            forecast_ids = df_log['id'].tolist()
            forecast_id  = st.selectbox(
                "Select Forecast ID (from Post-Event Analytics)",
                options=forecast_ids,
                format_func=lambda x: f"#{x} — {df_log[df_log['id']==x]['event_type'].values[0]} | "
                                      f"{df_log[df_log['id']==x]['zone'].values[0]} | "
                                      f"{df_log[df_log['id']==x]['forecast_ts'].values[0][:10]}"
            )
            selected = df_log[df_log['id'] == forecast_id].iloc[0]
        else:
            st.warning("No forecasts logged yet. Run a forecast first.")
            forecast_id = None
            selected    = pd.Series()
 
        st.markdown("**Step 2 — What Actually Happened**")
        c1, c2 = st.columns(2)
        actual_duration  = c1.number_input("Actual Resolution Duration (minutes)", min_value=1.0, value=60.0)
        officers_actual  = c2.number_input("Officers Actually Deployed", min_value=0, step=1,
                                            value=int(selected.get('officers', 5)) if not selected.empty else 5)
        barricades_actual= c1.number_input("Barricades Actually Used", min_value=0, step=1,
                                            value=int(selected.get('barricades', 2)) if not selected.empty else 2)
        diversions_actual= c2.number_input("Diversions Activated", min_value=0, step=1, value=1)
 
        st.markdown("**Step 3 — Plan Assessment**")
        plan_followed = st.selectbox(
            "Was the recommended deployment plan followed?",
            ["Yes — fully", "Yes — partially", "No — modified significantly", "No — different plan used"]
        )
        outcome = st.selectbox(
            "Overall outcome of traffic management",
            ["Excellent — congestion cleared faster than predicted",
             "Good — congestion cleared as predicted",
             "Average — minor delays beyond prediction",
             "Poor — significant delays beyond prediction",
             "Critical — situation escalated unexpectedly"]
        )
 
        st.markdown("**Step 4 — Officer Feedback**")
        what_worked       = st.text_area("What worked well?",
                                          placeholder="e.g. Early barricading at junction X prevented backup, "
                                                      "diversion via Ring Road was effective...")
        what_didnt_work   = st.text_area("What didn't work?",
                                          placeholder="e.g. Not enough officers at corridor entry, "
                                                      "diversion signs were unclear...")
        unexpected_issues = st.text_area("Unexpected issues encountered?",
                                          placeholder="e.g. Secondary accident at diversion point, "
                                                      "crowd larger than expected, VIP movement...")
        officer_notes     = st.text_area("Additional notes for future events",
                                          placeholder="e.g. For similar events deploy 2 hours earlier, "
                                                      "coordinate with event organizer for crowd estimate...")
 
        submitted = st.form_submit_button("💾 Save Feedback to Event Memory")
 
    if submitted and forecast_id is not None:
        feedback = {
            'forecast_id':             forecast_id,
            'event_type':              selected.get('event_type'),
            'event_cause':             selected.get('event_cause'),
            'zone':                    selected.get('zone'),
            'junction':                selected.get('junction'),
            'corridor':                selected.get('corridor'),
            'address':                 selected.get('address'),
            'start_datetime':          selected.get('start_datetime'),
            'road_closure':            selected.get('road_closure', 0),
            'predicted_duration_mins': selected.get('predicted_duration_mins'),
            'actual_duration_mins':    actual_duration,
            'severity_label':          selected.get('severity_label'),
            'officers_recommended':    selected.get('officers'),
            'officers_actual':         officers_actual,
            'barricades_recommended':  selected.get('barricades'),
            'barricades_actual':       barricades_actual,
            'diversions_recommended':  selected.get('barricades', 1),
            'diversions_actual':       diversions_actual,
            'plan_followed':           plan_followed,
            'what_worked':             what_worked,
            'what_didnt_work':         what_didnt_work,
            'unexpected_issues':       unexpected_issues,
            'outcome':                 outcome,
            'officer_notes':           officer_notes,
        }
        log_event_memory(feedback)
        update_actual(forecast_id, actual_duration)
        st.success("✅ Feedback saved to Event Memory! Future similar events will now "
                   "show this record as a reference on the Forecast page.")
        st.balloons()
 
    # ── MEMORY BROWSER ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📚 Event Memory Browser")
    st.caption("All past events with feedback — shown as reference on the Forecast page "
               "when similar events are detected.")
 
    memory_df = get_all_memory()
 
    if memory_df.empty:
        st.info("No feedback submitted yet. Events you submit feedback for will appear here.")
    else:
        # Summary metrics
        m1,m2,m3 = st.columns(3)
        m1.metric("Events in Memory", len(memory_df))
        if 'actual_duration_mins' in memory_df.columns:
            m2.metric("Avg Actual Duration",
                      f"{memory_df['actual_duration_mins'].mean():.0f} min")
        if 'outcome' in memory_df.columns:
            good = memory_df['outcome'].str.startswith(('Excellent','Good')).sum()
            m3.metric("Good Outcomes", f"{good} / {len(memory_df)}")
 
        # Filter by event type
        if 'event_type' in memory_df.columns:
            types = ['All'] + sorted(memory_df['event_type'].dropna().unique().tolist())
            selected_type = st.selectbox("Filter by Event Type", types)
            if selected_type != 'All':
                memory_df = memory_df[memory_df['event_type'] == selected_type]
 
        # Display key columns
        display_cols = [c for c in [
            'id','event_type','zone','junction','start_datetime',
            'predicted_duration_mins','actual_duration_mins',
            'officers_recommended','officers_actual',
            'plan_followed','outcome','logged_ts'
        ] if c in memory_df.columns]
        st.dataframe(memory_df[display_cols], use_container_width=True)
 
        # Expandable detail view
        st.markdown("#### Detailed Feedback Records")
        for _, row in memory_df.head(10).iterrows():
            with st.expander(
                f"#{row.get('id')} | {str(row.get('event_type','')).title()} | "
                f"Zone: {row.get('zone','?')} | {row.get('outcome','?')}"
            ):
                d1,d2,d3 = st.columns(3)
                d1.metric("Predicted", f"{row.get('predicted_duration_mins',0):.0f} min")
                d2.metric("Actual",    f"{row.get('actual_duration_mins',0):.0f} min")
                d3.metric("Officers",  f"{row.get('officers_actual','?')} deployed")
 
                if row.get('what_worked'):
                    st.success(f"✅ {row['what_worked']}")
                if row.get('what_didnt_work'):
                    st.error(f"❌ {row['what_didnt_work']}")
                if row.get('unexpected_issues'):
                    st.warning(f"⚠️ {row['unexpected_issues']}")
                if row.get('officer_notes'):
                    st.info(f"📝 {row['officer_notes']}")
 
st.subheader("🗑️ Delete Memory Record")
with st.form("delete_memory_form"):
    del_mem_id = st.number_input("Memory ID to delete", min_value=1, step=1)
    if st.form_submit_button("🗑️ Delete", type="primary"):
        delete_memory(int(del_mem_id))
        st.success(f"✅ Memory record #{del_mem_id} deleted.")
        st.rerun()