# # """
# # database.py — SQLite logging for forecasts and post-event feedback.
# # """
# # import sqlite3, os
# # from datetime import datetime

# # DB_PATH = 'data/event_forecasts.db'
# # os.makedirs('data', exist_ok=True)

# # def init_db():
# #     conn = sqlite3.connect(DB_PATH)
# #     c = conn.cursor()
# #     c.execute('''CREATE TABLE IF NOT EXISTS forecasts (
# #         id INTEGER PRIMARY KEY AUTOINCREMENT,
# #         event_type TEXT, event_cause TEXT, zone TEXT,
# #         junction TEXT, corridor TEXT, address TEXT,
# #         start_datetime TEXT, road_closure INTEGER,
# #         predicted_duration_mins REAL, severity_label TEXT,
# #         officers INTEGER, barricades INTEGER,
# #         actual_duration_mins REAL,
# #         forecast_ts TEXT
# #     )''')
# #     conn.commit(); conn.close()

# # def log_forecast(data: dict, predicted: float, severity: dict):
# #     conn = sqlite3.connect(DB_PATH)
# #     conn.execute('''INSERT INTO forecasts
# #         (event_type,event_cause,zone,junction,corridor,address,
# #          start_datetime,road_closure,predicted_duration_mins,
# #          severity_label,officers,barricades,forecast_ts)
# #         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
# #         data.get('event_type'), data.get('event_cause'),
# #         data.get('zone'), data.get('junction'), data.get('corridor'),
# #         data.get('address'), str(data.get('start_datetime')),
# #         int(data.get('road_closure_flag',0)),
# #         round(predicted,2), severity['label'],
# #         severity['officers'], severity['barricades'],
# #         datetime.now().isoformat()
# #     ))
# #     conn.commit(); conn.close()

# # def update_actual(forecast_id: int, actual_mins: float):
# #     conn = sqlite3.connect(DB_PATH)
# #     conn.execute('UPDATE forecasts SET actual_duration_mins=? WHERE id=?',
# #                  (actual_mins, forecast_id))
# #     conn.commit(); conn.close()

# # def get_all_forecasts():
# #     import pandas as pd
# #     conn = sqlite3.connect(DB_PATH)
# #     df = pd.read_sql('SELECT * FROM forecasts ORDER BY id DESC', conn)
# #     conn.close()
# #     return df

# # def get_accuracy_df():
# #     import pandas as pd
# #     conn = sqlite3.connect(DB_PATH)
# #     df = pd.read_sql('''SELECT event_type, predicted_duration_mins,
# #         actual_duration_mins,
# #         ABS(predicted_duration_mins - actual_duration_mins) AS abs_error
# #         FROM forecasts WHERE actual_duration_mins IS NOT NULL''', conn)
# #     conn.close()
# #     return df



# #-------------------------------New-----------------------------------------

# """
# database.py — SQLite logging for forecasts, post-event feedback, and institutional memory.
# """
# import sqlite3, os
# from datetime import datetime
 
# DB_PATH = 'data/event_forecasts.db'
# os.makedirs('data', exist_ok=True)
 
 
# def init_db():
#     conn = sqlite3.connect(DB_PATH)
#     c = conn.cursor()
 
#     # ── Original forecasts table ─────────────────────────────────────────────
#     c.execute('''CREATE TABLE IF NOT EXISTS forecasts (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         event_type TEXT, event_cause TEXT, zone TEXT,
#         junction TEXT, corridor TEXT, address TEXT,
#         start_datetime TEXT, road_closure INTEGER,
#         predicted_duration_mins REAL, severity_label TEXT,
#         officers INTEGER, barricades INTEGER,
#         actual_duration_mins REAL,
#         forecast_ts TEXT
#     )''')
 
#     # ── NEW: Event Memory table ──────────────────────────────────────────────
#     # One row per closed event. Links back to forecasts.id.
#     # Stores structured learnings captured via the Post-Event Feedback form.
#     c.execute('''CREATE TABLE IF NOT EXISTS event_memory (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         forecast_id INTEGER,                    -- FK → forecasts.id
#         event_type TEXT,
#         event_cause TEXT,
#         zone TEXT,
#         junction TEXT,
#         corridor TEXT,
#         address TEXT,
#         event_date TEXT,                        -- YYYY-MM-DD of the event
#         predicted_duration_mins REAL,
#         actual_duration_mins REAL,
#         officers_deployed INTEGER,
#         officers_were_sufficient INTEGER,       -- 1=yes, 0=no
#         barricades_deployed INTEGER,
#         barricades_were_sufficient INTEGER,
#         diversion_routes_used TEXT,             -- free text
#         diversion_effective INTEGER,            -- 1=yes, 0=no, -1=NA
#         road_closure_flag INTEGER,
#         what_worked_well TEXT,                  -- free text learnings
#         what_didnt_work TEXT,                   -- free text
#         crowd_density TEXT,                     -- low/medium/high/very_high
#         weather_impact TEXT,                    -- none/mild/significant
#         unexpected_factors TEXT,               -- free text
#         overall_rating INTEGER,                 -- 1-5 by officer
#         submitted_by TEXT,
#         created_ts TEXT
#     )''')
 
#     conn.commit()
#     conn.close()
 
 
# # ── ORIGINAL FUNCTIONS (unchanged) ───────────────────────────────────────────
 
# def log_forecast(data: dict, predicted: float, severity: dict):
#     conn = sqlite3.connect(DB_PATH)
#     conn.execute('''INSERT INTO forecasts
#         (event_type,event_cause,zone,junction,corridor,address,
#          start_datetime,road_closure,predicted_duration_mins,
#          severity_label,officers,barricades,forecast_ts)
#         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
#         data.get('event_type'), data.get('event_cause'),
#         data.get('zone'), data.get('junction'), data.get('corridor'),
#         data.get('address'), str(data.get('start_datetime')),
#         int(data.get('road_closure_flag', 0)),
#         round(predicted, 2), severity['label'],
#         severity['officers'], severity['barricades'],
#         datetime.now().isoformat()
#     ))
#     conn.commit()
#     conn.close()
 
 
# def update_actual(forecast_id: int, actual_mins: float):
#     conn = sqlite3.connect(DB_PATH)
#     conn.execute('UPDATE forecasts SET actual_duration_mins=? WHERE id=?',
#                  (actual_mins, forecast_id))
#     conn.commit()
#     conn.close()
 
 
# def get_all_forecasts():
#     import pandas as pd
#     conn = sqlite3.connect(DB_PATH)
#     df = pd.read_sql('SELECT * FROM forecasts ORDER BY id DESC', conn)
#     conn.close()
#     return df
 
 
# def get_accuracy_df():
#     import pandas as pd
#     conn = sqlite3.connect(DB_PATH)
#     df = pd.read_sql('''SELECT event_type, predicted_duration_mins,
#         actual_duration_mins,
#         ABS(predicted_duration_mins - actual_duration_mins) AS abs_error
#         FROM forecasts WHERE actual_duration_mins IS NOT NULL''', conn)
#     conn.close()
#     return df
 
 
# # ── NEW: EVENT MEMORY FUNCTIONS ───────────────────────────────────────────────
 
# def log_event_memory(memory: dict) -> int:
#     """
#     Save post-event feedback into event_memory table.
#     Returns the new row id.
#     """
#     conn = sqlite3.connect(DB_PATH)
#     cursor = conn.execute('''INSERT INTO event_memory (
#         forecast_id, event_type, event_cause, zone, junction, corridor,
#         address, event_date, predicted_duration_mins, actual_duration_mins,
#         officers_deployed, officers_were_sufficient,
#         barricades_deployed, barricades_were_sufficient,
#         diversion_routes_used, diversion_effective, road_closure_flag,
#         what_worked_well, what_didnt_work, crowd_density, weather_impact,
#         unexpected_factors, overall_rating, submitted_by, created_ts
#     ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
#         memory.get('forecast_id'),
#         memory.get('event_type'), memory.get('event_cause'),
#         memory.get('zone'), memory.get('junction'), memory.get('corridor'),
#         memory.get('address'), memory.get('event_date'),
#         memory.get('predicted_duration_mins'), memory.get('actual_duration_mins'),
#         memory.get('officers_deployed'), int(memory.get('officers_were_sufficient', 0)),
#         memory.get('barricades_deployed'), int(memory.get('barricades_were_sufficient', 0)),
#         memory.get('diversion_routes_used'), memory.get('diversion_effective', -1),
#         int(memory.get('road_closure_flag', 0)),
#         memory.get('what_worked_well'), memory.get('what_didnt_work'),
#         memory.get('crowd_density'), memory.get('weather_impact'),
#         memory.get('unexpected_factors'), memory.get('overall_rating'),
#         memory.get('submitted_by'), datetime.now().isoformat()
#     ))
#     new_id = cursor.lastrowid
#     conn.commit()
#     conn.close()
#     return new_id
 
 
# # def get_similar_past_events(event_type: str, zone: str = None,
# #                              junction: str = None, limit: int = 5) -> list[dict]:
# #     """
# #     Retrieve past event memories similar to the incoming event.
# #     Similarity priority: event_type + zone + junction → event_type + zone → event_type only.
# #     Returns a list of dicts ready for display.
# #     """
# #     import pandas as pd
# #     conn = sqlite3.connect(DB_PATH)
 
# #     rows = []
 
# #     if zone and junction:
# #         df = pd.read_sql('''
# #             SELECT * FROM event_memory
# #             WHERE event_type=? AND zone=? AND junction=?
# #             ORDER BY id DESC LIMIT ?''',
# #             conn, params=(event_type, zone, junction, limit))
# #         rows = df.to_dict('records')
 
# #     if len(rows) < limit and zone:
# #         existing_ids = [r['id'] for r in rows]
# #         placeholder = ','.join('?' * len(existing_ids)) if existing_ids else 'NULL'
# #         query = f'''SELECT * FROM event_memory
# #             WHERE event_type=? AND zone=?
# #             {'AND id NOT IN (' + placeholder + ')' if existing_ids else ''}
# #             ORDER BY id DESC LIMIT ?'''
# #         params = [event_type, zone] + existing_ids + [limit - len(rows)]
# #         df2 = pd.read_sql(query, conn, params=params)
# #         rows += df2.to_dict('records')
 
# #     if len(rows) < limit:
# #         existing_ids = [r['id'] for r in rows]
# #         placeholder = ','.join('?' * len(existing_ids)) if existing_ids else 'NULL'
# #         query = f'''SELECT * FROM event_memory
# #             WHERE event_type=?
# #             {'AND id NOT IN (' + placeholder + ')' if existing_ids else ''}
# #             ORDER BY id DESC LIMIT ?'''
# #         params = [event_type] + existing_ids + [limit - len(rows)]
# #         df3 = pd.read_sql(query, conn, params=params)
# #         rows += df3.to_dict('records')
 
# #     conn.close()
# #     return rows[:limit]

# # def get_similar_past_events(forecast_id: int = None, limit: int = 5) -> list:
# #     import pandas as pd
# #     if not forecast_id:
# #         return []
# #     conn = sqlite3.connect(DB_PATH)
# #     df = pd.read_sql('''
# #         SELECT * FROM event_memory
# #         WHERE forecast_id = ?
# #         ORDER BY id DESC LIMIT ?''',
# #         conn, params=(forecast_id, limit))
# #     conn.close()
# #     return df.to_dict('records')

# def get_similar_past_events(event_type: str = None, zone: str = None,
#                              junction: str = None, corridor: str = None,
#                              road_closure_flag: int = None, hour: int = None,
#                              is_weekend: int = None, limit: int = 5) -> list:
#     """
#     Retrieve past event memories scored by weighted similarity.
    
#     Scoring weights:
#     - event_type match      : 40 pts  (most important)
#     - zone match            : 20 pts  (same area = same road geometry)
#     - junction match        : 15 pts  (exact bottleneck)
#     - corridor match        : 10 pts  (same road stretch)
#     - road_closure match    : 8  pts  (changes resource needs drastically)
#     - time_of_day bucket    : 4  pts  (morning/evening/night peak)
#     - weekend match         : 3  pts  (crowd behavior differs)
    
#     Returns top-N records sorted by score descending. Only returns records
#     with score >= 40 (must at least match event_type).
#     """
#     import pandas as pd

#     conn = sqlite3.connect(DB_PATH)
#     df = pd.read_sql('SELECT * FROM event_memory', conn)
#     conn.close()

#     if df.empty:
#         return []

#     # Time bucket helper
#     def time_bucket(h):
#         if h is None:
#             return None
#         if h in [8, 9, 10]:
#             return 'morning_peak'
#         elif h in [17, 18, 19, 20]:
#             return 'evening_peak'
#         elif h in [22, 23, 0, 1, 2, 3, 4, 5]:
#             return 'night'
#         else:
#             return 'midday'

#     input_bucket  = time_bucket(hour)
#     input_weekend = is_weekend

#     scores = []
#     for _, row in df.iterrows():
#         score = 0

#         # event_type — 40 pts
#         if event_type and str(row.get('event_type', '')).lower() == str(event_type).lower():
#             score += 40

#         # zone — 20 pts
#         if zone and str(row.get('zone', '')).lower() == str(zone).lower():
#             score += 20

#         # junction — 15 pts
#         if junction and str(row.get('junction', '')).lower() == str(junction).lower():
#             score += 15

#         # corridor — 10 pts
#         if corridor and str(row.get('corridor', '')).lower() == str(corridor).lower():
#             score += 10

#         # road_closure — 8 pts
#         if road_closure_flag is not None and int(row.get('road_closure_flag', 0)) == int(road_closure_flag):
#             score += 8

#         # time of day bucket — 4 pts
#         if input_bucket and row.get('event_date'):
#             try:
#                 row_hour = pd.to_datetime(row['event_date']).hour
#                 if time_bucket(row_hour) == input_bucket:
#                     score += 4
#             except:
#                 pass

#         # weekend — 3 pts
#         if input_weekend is not None and row.get('event_date'):
#             try:
#                 row_dow = pd.to_datetime(row['event_date']).weekday()
#                 row_weekend = 1 if row_dow >= 5 else 0
#                 if row_weekend == input_weekend:
#                     score += 3
#             except:
#                 pass

#         scores.append((score, row.to_dict()))

#     # Filter: must match event_type at minimum (score >= 40)
#     scores = [(s, r) for s, r in scores if s >= 40]

#     # Sort by score descending
#     scores.sort(key=lambda x: x[0], reverse=True)

#     # Attach score to each record for display
#     results = []
#     for s, r in scores[:limit]:
#         r['similarity_score'] = s
#         r['similarity_pct']   = round((s / 100) * 100, 1)
#         results.append(r)

#     return results
 
 
# def get_all_event_memories():
#     """Return all event_memory rows as a DataFrame."""
#     import pandas as pd
#     conn = sqlite3.connect(DB_PATH)
#     df = pd.read_sql('SELECT * FROM event_memory ORDER BY id DESC', conn)
#     conn.close()
#     return df
 
 
# def get_memory_stats():
#     """Aggregate stats from event_memory for analytics."""
#     import pandas as pd
#     conn = sqlite3.connect(DB_PATH)
#     df = pd.read_sql('SELECT * FROM event_memory', conn)
#     conn.close()
#     return df




#--------------------------------------------New----------------------------------------

"""
database.py — SQLite logging for forecasts, post-event feedback, and event memory.
"""
import sqlite3, os
import pandas as pd
from datetime import datetime
 
DB_PATH = 'data/event_forecasts.db'
os.makedirs('data', exist_ok=True)
 
 
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
 
    # Original forecasts table
    c.execute('''CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT, event_cause TEXT, zone TEXT,
        junction TEXT, corridor TEXT, address TEXT,
        start_datetime TEXT, road_closure INTEGER,
        predicted_duration_mins REAL, severity_label TEXT,
        officers INTEGER, barricades INTEGER,
        actual_duration_mins REAL,
        forecast_ts TEXT
    )''')
 
    # Event memory / feedback table
    c.execute('''CREATE TABLE IF NOT EXISTS event_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        forecast_id INTEGER,
        event_type TEXT,
        event_cause TEXT,
        zone TEXT,
        junction TEXT,
        corridor TEXT,
        address TEXT,
        start_datetime TEXT,
        road_closure INTEGER,
        predicted_duration_mins REAL,
        actual_duration_mins REAL,
        severity_label TEXT,
        officers_recommended INTEGER,
        officers_actual INTEGER,
        barricades_recommended INTEGER,
        barricades_actual INTEGER,
        diversions_recommended INTEGER,
        diversions_actual INTEGER,
        plan_followed TEXT,
        what_worked TEXT,
        what_didnt_work TEXT,
        unexpected_issues TEXT,
        outcome TEXT,
        officer_notes TEXT,
        logged_ts TEXT
    )''')
 
    conn.commit()
    conn.close()
 
 
# ── FORECAST LOGGING ──────────────────────────────────────────────────────────
def log_forecast(data: dict, predicted: float, severity: dict) -> int:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute('''INSERT INTO forecasts
        (event_type, event_cause, zone, junction, corridor, address,
         start_datetime, road_closure, predicted_duration_mins,
         severity_label, officers, barricades, forecast_ts)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
        data.get('event_type'), data.get('event_cause'),
        data.get('zone'), data.get('junction'), data.get('corridor'),
        data.get('address'), str(data.get('start_datetime')),
        int(data.get('road_closure_flag', 0)),
        round(predicted, 2), severity['label'],
        severity['officers'], severity['barricades'],
        datetime.now().isoformat()
    ))
    forecast_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return forecast_id
 
 
def update_actual(forecast_id: int, actual_mins: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('UPDATE forecasts SET actual_duration_mins=? WHERE id=?',
                 (actual_mins, forecast_id))
    conn.commit()
    conn.close()
 
 
def get_all_forecasts() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql('SELECT * FROM forecasts ORDER BY id DESC', conn)
    conn.close()
    return df
 
 
def get_accuracy_df() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql('''SELECT event_type, predicted_duration_mins,
        actual_duration_mins,
        ABS(predicted_duration_mins - actual_duration_mins) AS abs_error
        FROM forecasts WHERE actual_duration_mins IS NOT NULL''', conn)
    conn.close()
    return df
 
 
# ── EVENT MEMORY ──────────────────────────────────────────────────────────────
def log_event_memory(feedback: dict):
    """
    Store a complete post-event feedback record in event_memory table.
    feedback dict keys match the event_memory columns.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''INSERT INTO event_memory (
        forecast_id, event_type, event_cause, zone, junction, corridor,
        address, start_datetime, road_closure,
        predicted_duration_mins, actual_duration_mins,
        severity_label, officers_recommended, officers_actual,
        barricades_recommended, barricades_actual,
        diversions_recommended, diversions_actual,
        plan_followed, what_worked, what_didnt_work,
        unexpected_issues, outcome, officer_notes, logged_ts
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
        feedback.get('forecast_id'),
        feedback.get('event_type'),
        feedback.get('event_cause'),
        feedback.get('zone'),
        feedback.get('junction'),
        feedback.get('corridor'),
        feedback.get('address'),
        feedback.get('start_datetime'),
        int(feedback.get('road_closure', 0)),
        feedback.get('predicted_duration_mins'),
        feedback.get('actual_duration_mins'),
        feedback.get('severity_label'),
        feedback.get('officers_recommended'),
        feedback.get('officers_actual'),
        feedback.get('barricades_recommended'),
        feedback.get('barricades_actual'),
        feedback.get('diversions_recommended'),
        feedback.get('diversions_actual'),
        feedback.get('plan_followed'),
        feedback.get('what_worked'),
        feedback.get('what_didnt_work'),
        feedback.get('unexpected_issues'),
        feedback.get('outcome'),
        feedback.get('officer_notes'),
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()
 
 
def get_all_memory() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql('SELECT * FROM event_memory ORDER BY id DESC', conn)
    conn.close()
    return df
 
 
# def find_similar_events(event_type: str, zone: str,
#                          hour: int, day_of_week: int,
#                          top_n: int = 3) -> pd.DataFrame:
#     """
#     Find past events from event_memory that are similar to the current input.
 
#     Similarity logic (in priority order):
#     1. Exact match: same event_type AND same zone
#     2. Partial match: same event_type OR same zone
#     3. Same time regime: peak/off-peak alignment
#     4. Same day type: weekday/weekend alignment
 
#     Returns top_n most relevant past events with their feedback.
#     """
#     conn = sqlite3.connect(DB_PATH)
 
#     # Determine time regime of new event
#     is_peak = 1 if hour in [8, 9, 10, 17, 18, 19, 20] else 0
#     is_weekend = 1 if day_of_week in [5, 6] else 0
 
#     query = '''
#         SELECT *,
#             -- Similarity score: higher = more similar
#             (
#                 CASE WHEN LOWER(event_type) = LOWER(?) THEN 40 ELSE 0 END +
#                 CASE WHEN LOWER(zone)       = LOWER(?) THEN 30 ELSE 0 END +
#                 CASE WHEN CAST(
#                     CASE WHEN CAST(strftime('%H', start_datetime) AS INTEGER)
#                          IN (8,9,10,17,18,19,20) THEN 1 ELSE 0 END
#                 AS INTEGER) = ? THEN 15 ELSE 0 END +
#                 CASE WHEN CAST(
#                     CASE WHEN CAST(strftime('%w', start_datetime) AS INTEGER)
#                          IN (0,6) THEN 1 ELSE 0 END
#                 AS INTEGER) = ? THEN 15 ELSE 0 END
#             ) AS similarity_score
#         FROM event_memory
#         WHERE actual_duration_mins IS NOT NULL
#         ORDER BY similarity_score DESC, id DESC
#         LIMIT ?
#     '''

def find_similar_events(event_type: str, zone: str,
                         hour: int, day_of_week: int,
                         top_n: int = 3) -> pd.DataFrame:
    """
    Find past events from event_memory that are strictly similar.
    ALL of the following must match exactly:
      - event_type
      - zone
      - same time regime (peak / off-peak)
      - same day type (weekend / weekday)
    If any one of these doesn't match, the event is excluded.
    """
    conn = sqlite3.connect(DB_PATH)

    is_peak    = 1 if hour in [8, 9, 10, 17, 18, 19, 20] else 0
    is_weekend = 1 if day_of_week in [5, 6] else 0

    query = '''
        SELECT *
        FROM event_memory
        WHERE actual_duration_mins IS NOT NULL
          AND LOWER(TRIM(event_type)) = LOWER(TRIM(?))
          AND LOWER(TRIM(zone))       = LOWER(TRIM(?))
          AND CAST(
                CASE WHEN CAST(strftime('%H', start_datetime) AS INTEGER)
                     IN (8,9,10,17,18,19,20) THEN 1 ELSE 0 END
              AS INTEGER) = ?
          AND CAST(
                CASE WHEN CAST(strftime('%w', start_datetime) AS INTEGER)
                     IN (0,6) THEN 1 ELSE 0 END
              AS INTEGER) = ?
        ORDER BY id DESC
        LIMIT ?
    '''

    df = pd.read_sql(query, conn,
                     params=(event_type, zone, is_peak, is_weekend, top_n))
    conn.close()
    return df
 
    df = pd.read_sql(query, conn,
                     params=(event_type, zone, is_peak, is_weekend, top_n))
    conn.close()
 
    # Only return events with at least some similarity
    df = df[df['similarity_score'] > 0]
    return df


def delete_forecast(forecast_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM forecasts WHERE id = ?', (forecast_id,))
    conn.commit()
    conn.close()

def delete_memory(memory_id: int):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM event_memory WHERE id = ?', (memory_id,))
    conn.commit()
    conn.close()