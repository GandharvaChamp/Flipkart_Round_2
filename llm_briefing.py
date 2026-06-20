# """
# llm_briefing.py — Claude API integration for deployment briefings.
# """
# import os
# import anthropic

# def generate_briefing(event_data: dict, severity: dict,
#                       predicted_duration: float, conflicts: list = []) -> str:
#     client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))
#     conflict_text = ""
#     if conflicts:
#         conflict_text = f"\nCONCURRENT EVENTS ON SAME DATE: {', '.join(conflicts)}. COMPOUND RISK SCENARIO."

#     prompt = f"""You are a senior traffic management officer. Write a concise deployment briefing.

# EVENT:
# - Type: {event_data.get('event_type','Unknown')}
# - Cause: {event_data.get('event_cause','Unknown')}
# - Location: {event_data.get('address','Unknown')} | Zone: {event_data.get('zone','Unknown')}
# - Junction: {event_data.get('junction','Unknown')} | Corridor: {event_data.get('corridor','Unknown')}
# - Start: {event_data.get('start_datetime','Unknown')}
# - Road Closure Required: {event_data.get('road_closure_flag','Unknown')}

# FORECAST:
# - Estimated Resolution Time: {predicted_duration:.0f} minutes
# - Severity: {severity['label']}
# - Officers Recommended: {severity['officers']}
# - Barricades: {severity['barricades']}
# - Diversions: {severity['diversions']}
# {conflict_text}

# Write the briefing with exactly these 5 sections (keep under 220 words total):
# 1. SITUATION SUMMARY (2 sentences)
# 2. DEPLOYMENT ORDER (officer count, positions, timing)
# 3. BARRICADING PLAN (specific junctions)
# 4. DIVERSION ROUTES (alternate roads)
# 5. SPECIAL INSTRUCTIONS (any compound risk or escalation triggers)
# """
#     message = client.messages.create(
#         model="claude-sonnet-4-6",
#         max_tokens=600,
#         messages=[{"role":"user","content":prompt}]
#     )
#     return message.content[0].text


# def generate_post_event_report(event_info: dict,
#                                  predicted: float, actual: float) -> str:
#     client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY",""))
#     err  = abs(predicted - actual)
#     direction = "overestimated" if predicted > actual else "underestimated"
#     prompt = f"""Write a short post-event traffic analysis report (under 150 words):
# - Event Type: {event_info.get('event_type','Unknown')}
# - Location: {event_info.get('address','Unknown')}
# - Predicted Duration: {predicted:.0f} min
# - Actual Duration: {actual:.0f} min
# - Forecast Error: {err:.0f} min ({direction})

# Include: what the model got right, likely cause of error, one improvement recommendation.
# """
#     message = client.messages.create(
#         model="claude-sonnet-4-6",
#         max_tokens=350,
#         messages=[{"role":"user","content":prompt}]
#     )
#     return message.content[0].text


# src/llm_briefing.py — Gemini version
import os
from mistralai.client import Mistral
from dotenv import load_dotenv
load_dotenv()
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
MODEL = "mistral-small-2503"

def generate_briefing(event_data: dict, severity: dict,
                      predicted_duration: float, conflicts: list = []) -> str:

    conflict_text = ""
    if conflicts:
        conflict_text = f"\nCONCURRENT EVENTS: {', '.join(conflicts)} — compound risk scenario."

    prompt = f"""You are a senior traffic management officer. Write a deployment briefing.

EVENT:
- Type: {event_data.get('event_type', 'Unknown')}
- Cause: {event_data.get('event_cause', 'Unknown')}
- Location: {event_data.get('address', 'Unknown')} | Zone: {event_data.get('zone', 'Unknown')}
- Junction: {event_data.get('junction', 'Unknown')} | Corridor: {event_data.get('corridor', 'Unknown')}
- Start: {event_data.get('start_datetime', 'Unknown')}
- Road Closure: {event_data.get('road_closure_flag', 0)}

FORECAST:
- Estimated Duration: {predicted_duration:.0f} minutes
- Severity: {severity['label']}
- Officers: {severity['officers']}
- Barricades: {severity['barricades']}
- Diversions: {severity['diversions']}
{conflict_text}

Write with exactly these 5 sections (under 220 words total):
1. SITUATION SUMMARY
2. DEPLOYMENT ORDER
3. BARRICADING PLAN
4. DIVERSION ROUTES
5. SPECIAL INSTRUCTIONS
"""

    response = client.chat.complete(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def generate_post_event_report(event_info: dict,
                                predicted: float, actual: float) -> str:
    error     = abs(predicted - actual)
    direction = "overestimated" if predicted > actual else "underestimated"

    prompt = f"""Write a short post-event traffic analysis report (under 150 words):
- Event Type: {event_info.get('event_type', 'Unknown')}
- Predicted Duration: {predicted:.0f} min
- Actual Duration: {actual:.0f} min
- Forecast Error: {error:.0f} min ({direction})

Include: what went well, likely cause of error, one improvement recommendation.
"""

    response = client.chat.complete(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# from datetime import datetime

# def generate_briefing(event_data: dict, severity: dict,
#                       predicted_duration: float, conflicts: list = []) -> str:

#     event_type   = str(event_data.get('event_type', 'Unknown')).replace('_', ' ').title()
#     event_cause  = str(event_data.get('event_cause', 'Unknown')).replace('_', ' ').title()
#     zone         = str(event_data.get('zone', 'Unknown')).upper()
#     junction     = str(event_data.get('junction', 'Unknown')).title()
#     corridor     = str(event_data.get('corridor', 'Unknown')).title()
#     address      = str(event_data.get('address', 'Unknown'))
#     start_dt     = event_data.get('start_datetime', 'Unknown')
#     road_closure = bool(event_data.get('road_closure_flag', 0))
#     officers     = severity['officers']
#     barricades   = severity['barricades']
#     diversions   = severity['diversions']
#     severity_label = severity['label']

#     try:
#         start_str = start_dt.strftime('%d %b %Y, %H:%M') if hasattr(start_dt, 'strftime') else str(start_dt)
#     except:
#         start_str = str(start_dt)

#     hrs  = int(predicted_duration // 60)
#     mins = int(predicted_duration % 60)
#     duration_str = f"{hrs}h {mins}min" if hrs > 0 else f"{mins} min"

#     sev_code = severity_label.split()[1] if len(severity_label.split()) > 1 else 'LOW'

#     situation_map = {
#         'LOW':      f"A {event_type} event is scheduled at {address} (Zone {zone}). "
#                     f"Estimated impact duration is {duration_str}. Standard monitoring protocols apply.",
#         'MODERATE': f"A {event_type} event at {address} (Zone {zone}) is forecast to cause moderate "
#                     f"traffic disruption lasting approximately {duration_str}. "
#                     f"Pre-emptive officer deployment is recommended.",
#         'HIGH':     f"A HIGH-impact {event_type} event at {address} (Zone {zone}) is expected to "
#                     f"cause significant congestion for approximately {duration_str}. "
#                     f"Immediate resource pre-positioning is required.",
#         'CRITICAL': f"CRITICAL ALERT: A {event_type} event at {address} (Zone {zone}) is forecast to "
#                     f"cause severe traffic breakdown for approximately {duration_str}. "
#                     f"Full emergency deployment protocol must be activated immediately."
#     }
#     situation = situation_map.get(sev_code, situation_map['MODERATE'])

#     deploy_map = {
#         'LOW':      f"- Deploy {officers} officers to {junction} junction by event start.\n"
#                     f"- One at corridor entry, one at exit for flow monitoring.\n"
#                     f"- Radio check every 60 minutes.",
#         'MODERATE': f"- Deploy {officers} officers: 2 at {junction}, 2 along {corridor}, 1 standby.\n"
#                     f"- Officers in position 30 minutes before event start.\n"
#                     f"- Radio check every 30 minutes.",
#         'HIGH':     f"- Deploy {officers} officers: 4 at {junction}, 3 along {corridor}, "
#                     f"2 at adjacent intersections, 2 diversion duty, 1 command officer.\n"
#                     f"- All in position 2 hours before event start.\n"
#                     f"- Mobile command unit at Zone {zone} entry. Radio check every 15 minutes.",
#         'CRITICAL': f"- Deploy {officers} officers immediately: 6 at {junction}, 4 along {corridor}, "
#                     f"4 at adjacent intersections, 3 diversion duty, 2 rapid response, 1 commander.\n"
#                     f"- All units deployed 3 hours before event start.\n"
#                     f"- Forward command post at Zone {zone}. Continuous radio communication.\n"
#                     f"- Alert nearest police station for backup."
#     }
#     deploy_order = deploy_map.get(sev_code, deploy_map['MODERATE'])

#     if barricades == 0:
#         barricade_plan = "No barricades required. Use traffic cones for soft lane guidance only."
#     else:
#         points = [f"  Point {i+1}: Approach to {junction}" if i == 0
#                   else f"  Point {i+1}: {corridor} checkpoint {i}"
#                   for i in range(barricades)]
#         barricade_plan = f"- Place {barricades} barricades:\n" + "\n".join(points)
#         if road_closure:
#             barricade_plan += "\n- Coordinate with GHMC for road closure signage."

#     if diversions == 0:
#         diversion_plan = "No active diversions required. Advise motorists verbally."
#     else:
#         lines = []
#         if diversions >= 1:
#             lines.append(f"- Primary: Reroute traffic away from {corridor} via parallel arterial road.")
#         if diversions >= 2:
#             lines.append(f"- Secondary: Redirect Zone {zone} entry via ring road bypass.")
#         if diversions >= 3:
#             lines.append(f"- Tertiary: Full corridor closure — activate alternate route plan for Zone {zone}.")
#         diversion_plan = "\n".join(lines)

#     special = []
#     if conflicts:
#         special.append(f"CONCURRENT EVENTS: {', '.join(conflicts)} — request additional reserve force.")
#     if road_closure:
#         special.append("Road closure in effect. Notify public via VMS boards.")
#     if event_type.lower() in ['political rally', 'rally', 'protest', 'march']:
#         special.append("Political event — maintain neutrality. Senior officer must be present.")
#     if sev_code == 'CRITICAL':
#         special.append("Escalate to DCP if queue exceeds 2 km or duration exceeds forecast by 30 min.")
#     if not special:
#         special.append("No special risk flags. Follow standard operating procedure.")

#     now_str = datetime.now().strftime('%d %b %Y, %H:%M')

#     briefing = f"""
# ╔══════════════════════════════════════════════════════════╗
#   TRAFFIC DEPLOYMENT BRIEFING — {severity_label}
#   Generated: {now_str}
# ╚══════════════════════════════════════════════════════════╝

# EVENT REFERENCE
#   Type     : {event_type} ({event_cause})
#   Location : {address}
#   Zone     : {zone}  |  Junction : {junction}  |  Corridor : {corridor}
#   Start    : {start_str}
#   Forecast : {duration_str}  |  Road Closure: {'YES' if road_closure else 'NO'}

# ────────────────────────────────────────────────────────────
# 1. SITUATION SUMMARY
# ────────────────────────────────────────────────────────────
# {situation}

# ────────────────────────────────────────────────────────────
# 2. DEPLOYMENT ORDER  ({officers} officers required)
# ────────────────────────────────────────────────────────────
# {deploy_order}

# ────────────────────────────────────────────────────────────
# 3. BARRICADING PLAN  ({barricades} barricades)
# ────────────────────────────────────────────────────────────
# {barricade_plan}

# ────────────────────────────────────────────────────────────
# 4. DIVERSION ROUTES  ({diversions} active diversions)
# ────────────────────────────────────────────────────────────
# {diversion_plan}

# ────────────────────────────────────────────────────────────
# 5. SPECIAL INSTRUCTIONS
# ────────────────────────────────────────────────────────────
# {chr(10).join(special)}

# ══════════════════════════════════════════════════════════════
#   END OF BRIEFING — File this document after event completion.
# ══════════════════════════════════════════════════════════════
# """.strip()

#     return briefing


# def generate_post_event_report(event_info: dict,
#                                 predicted: float, actual: float) -> str:
#     error     = abs(predicted - actual)
#     direction = "overestimated" if predicted > actual else "underestimated"
#     pct_error = (error / actual * 100) if actual > 0 else 0

#     if pct_error < 15:
#         accuracy_label = "GOOD (within 15%)"
#         recommendation = "Model performed well. Continue using current feature set."
#     elif pct_error < 35:
#         accuracy_label = "ACCEPTABLE (15-35% error)"
#         recommendation = "Consider adding real-time crowd density or weather features."
#     else:
#         accuracy_label = "POOR (>35% error)"
#         recommendation = (f"Model significantly {direction} duration. Add this event to "
#                           f"the retraining dataset and review features for "
#                           f"{event_info.get('event_type','this')} events.")

#     now_str = datetime.now().strftime('%d %b %Y, %H:%M')

#     return f"""
# POST-EVENT ACCURACY REPORT — {now_str}

# Event Type        : {str(event_info.get('event_type','Unknown')).title()}
# Predicted Duration: {predicted:.0f} min
# Actual Duration   : {actual:.0f} min
# Absolute Error    : {error:.0f} min ({pct_error:.1f}%)
# Direction         : Model {direction}
# Accuracy Rating   : {accuracy_label}

# Recommendation: {recommendation}
# Action: {'No retraining needed.' if pct_error < 15 else 'Add to feedback log for next retraining cycle.'}
# """.strip()