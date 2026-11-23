import numpy as np
import pandas as pd
import re

def parse_target_time(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        if val < 1.0:
            hours = val * 24
            return round(hours * 4) / 4
        else:
            val_str = str(int(val))
    else:
        val_str = str(val).strip()
    if '-' in val_str:
        val_str = val_str.split('-')[0].strip()
    if ',' in val_str:
        val_str = val_str.split(',')[0].strip()
    if ';' in val_str:
        val_str = val_str.split(';')[0].strip()
    val_str = val_str.replace(':', '')
    digits = re.sub(r'\D', '', val_str)
    if not digits:
        return np.nan
    if len(digits) <= 2:
        hr = int(digits)
        mn = 0
    else:
        hr = int(digits[:-2])
        mn = int(digits[-2:])
    return hr + mn / 60.0

def build_candidates(response: dict, meeting: dict) -> pd.DataFrame:
    days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat']
    L = int(meeting['meeting_length_min']) // 60

    query_candidates = []
    for day in days:
        for s in range(7, 22 - L + 1):
            occupied = False
            for h in range(s, s + L):
                col_name = f"sched_{day}_{h:02d}00"
                if response.get(col_name) in ['ON', 'OL']:
                    occupied = True
                    break
            if not occupied:
                query_candidates.append({
                    'day': day,
                    'start_hour': s,
                    'end_hour': s + L
                })
    return pd.DataFrame(query_candidates)

def build_features(candidates: pd.DataFrame, response: dict, meeting: dict) -> pd.DataFrame:
    days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat']
    L = int(meeting['meeting_length_min']) // 60
    features = []

    # Precompute busy counts
    weekly_class_hours = []
    for d in days:
        cnt = sum([1 for h in range(7, 22) if response.get(f"sched_{d}_{h:02d}00") in ['ON', 'OL']])
        weekly_class_hours.append(cnt)
    max_weekly_hours = max(weekly_class_hours)
    total_class_hours = float(sum(weekly_class_hours))

    # Override with dataset total class hours if available
    if 'total_class_hours' in response:
        total_class_hours = float(response['total_class_hours'])

    # Precompute overall feasible candidate count
    if 'n_feasible_slots' in response:
        n_feasible = float(response['n_feasible_slots'])
    elif 'n_feasible' in response:
        n_feasible = float(response['n_feasible'])
    else:
        n_feasible = 0
        for d in days:
            for s in range(7, 22 - L + 1):
                occ = False
                for h in range(s, s + L):
                    if response.get(f"sched_{d}_{h:02d}00") in ['ON', 'OL']:
                        occ = True
                        break
                if not occ:
                    n_feasible += 1

    # Check if density_tier is explicitly provided in response, otherwise fallback to class hours calculation
    if 'density_tier' in response:
        density_tier = response['density_tier']
    else:
        density_tier = 'Light'
        if total_class_hours > 15:
            density_tier = 'Heavy'
        elif total_class_hours > 5:
            density_tier = 'Medium'

    for _, cand in candidates.iterrows():
        day = cand['day']
        s = cand['start_hour']
        e = cand['end_hour']

        day_idx = days.index(day)
        is_saturday = 1.0 if day == 'sat' else 0.0
        overlaps_lunch = 1.0 if (s < 13 and e > 12) else 0.0
        is_morning = 1.0 if s < 12 else 0.0
        is_late = 1.0 if e > 16 else 0.0

        class_hours = []
        class_types = []
        for h in range(7, 22):
            val = response.get(f"sched_{day}_{h:02d}00")
            if val in ['ON', 'OL']:
                class_hours.append(h)
                class_types.append(val)

        gap_before = 6.0
        gap_after = 6.0
        classes_before = [ch for ch in class_hours if ch < s]
        classes_after = [ch for ch in class_hours if ch >= e]

        if classes_before:
            gap_before = min(6.0, s - max(classes_before) - 1)
        if classes_after:
            gap_after = min(6.0, min(classes_after) - e)

        min_gap = min(gap_before, gap_after)
        back_to_back = 1.0 if (gap_before == 0 or gap_after == 0) else 0.0

        day_class_hours = float(len(class_hours))
        day_is_free = 1.0 if day_class_hours == 0 else 0.0
        day_is_busiest = 1.0 if (day_class_hours == max_weekly_hours and day_class_hours > 0) else 0.0
        day_onsite_hours = float(sum([1 for ct in class_types if ct == 'ON']))

        pref_day_col = 'Pref' + day.capitalize() if day != 'thu' else 'PrefThurs'
        avoid_day_col = 'Avoid' + day.capitalize() if day != 'thu' else 'AvoidThurs'
        pref_day = float(response.get(pref_day_col, 0))
        avoid_day = float(response.get(avoid_day_col, 0))

        bins = [
            (7, 8, '0708'), (8, 10, '0810'), (10, 12, '1012'), (12, 14, '1214'),
            (14, 16, '1416'), (16, 18, '1618'), (18, 20, '1820'), (20, 22, '2022'), (22, 24, '2224')
        ]

        pref_time_share = 0.0
        avoid_time_share = 0.0
        for b_start, b_end, b_lbl in bins:
            overlap = max(0.0, min(e, b_end) - max(s, b_start))
            if overlap > 0:
                p_flag = float(response.get(f"pref_time_{b_lbl}", 0))
                a_flag = float(response.get(f"avoid_time_{b_lbl}", 0))
                pref_time_share += (overlap / L) * p_flag
                avoid_time_share += (overlap / L) * a_flag

        pref_break = float(response.get('Preferred Break', 0))
        break_shortfall = max(0.0, (pref_break / 60.0) - min_gap) if day_class_hours > 0 else 0.0

        b2b_tolerance = float(response.get('Back-To-Back Tolerance', 0))
        b2b_x_tolerance = back_to_back * (6.0 - b2b_tolerance)

        meeting_mod = meeting['meeting_modality']
        pref_mod = response['Preferred Modality']
        modality_match = 1.0 if pref_mod == meeting_mod else 0.0
        pref_modality_none = 1.0 if pref_mod == 'No Preference' else 0.0

        commute_val = response['Commute']
        commute_min = float(commute_val * 60 if commute_val <= 4 else commute_val)
        extra_trip_cost = commute_min if (meeting_mod == 'Onsite' and day_onsite_hours == 0) else 0.0

        online_next_to_class = 0.0
        if gap_before == 0 and len(classes_before) > 0:
            idx_before = class_hours.index(max(classes_before))
            if class_types[idx_before] == 'OL':
                online_next_to_class = 1.0
        if gap_after == 0 and len(classes_after) > 0:
            idx_after = class_hours.index(min(classes_after))
            if class_types[idx_after] == 'OL':
                online_next_to_class = 1.0

        features.append({
            'day_idx': day_idx,
            'is_saturday': is_saturday,
            'overlaps_lunch': overlaps_lunch,
            'is_morning': is_morning,
            'is_late': is_late,
            'gap_before': gap_before,
            'gap_after': gap_after,
            'min_gap': min_gap,
            'back_to_back': back_to_back,
            'day_class_hours': day_class_hours,
            'day_is_free': day_is_free,
            'day_is_busiest': day_is_busiest,
            'day_onsite_hours': day_onsite_hours,
            'meeting_len_h': float(L),
            'meeting_onsite': 1.0 if meeting_mod == 'Onsite' else 0.0,
            'type_project': 1.0 if meeting['meeting_type'] == 'Project' else 0.0,
            'type_event': 1.0 if meeting['meeting_type'] == 'Event' else 0.0,
            'tier_heavy': 1.0 if density_tier == 'Heavy' else 0.0,
            'tier_medium': 1.0 if density_tier == 'Medium' else 0.0,
            'n_feasible': float(n_feasible),
            'total_class_hours': float(total_class_hours),
            'pref_day': pref_day,
            'avoid_day': avoid_day,
            'pref_time_share': pref_time_share,
            'avoid_time_share': avoid_time_share,
            'break_shortfall': break_shortfall,
            'b2b_tolerance': b2b_tolerance,
            'b2b_x_tolerance': b2b_x_tolerance,
            'modality_match': modality_match,
            'pref_modality_none': pref_modality_none,
            'commute_min': commute_min,
            'extra_trip_cost': extra_trip_cost,
            'online_next_to_class': online_next_to_class
        })
    return pd.DataFrame(features)
