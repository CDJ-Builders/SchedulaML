"""
plot_schedules.py - render one PNG per Schedula scenario (Monday-Saturday week).

Library: matplotlib, pandas  (pip install matplotlib pandas)

Usage:
    python plot_schedules.py scenarios.csv
    python plot_schedules.py scenarios.csv --out schedule_pngs --start-hour 8 --end-hour 18

Supported CSV layouts (auto-detected):

1) WIDE  - one row per scenario, one column per hourly block, e.g.
       scenario_id, sched_mon_0800, sched_mon_0900, ..., sched_sat_1700
   Cell values: F / free / 0 / blank = free
                ON / onsite / 1       = onsite class
                OL / online / 2       = online class
   Optional columns used if present: density_tier, meeting_length_min,
   meeting_modality, meeting_type, target_day, target_start_time.

2) LONG  - one row per class, e.g.
       scenario_id, day, start_time, end_time[, modality]
   Times can be "08:00", "8:00", "8", or "13:30".
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat"]
DAY_LABELS = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri", "sat": "Sat"}

COLORS = {"ON": "#1A6B3C", "OL": "#2F6DB5"}     # onsite = DLSU-D green, online = blue
LABELS = {"ON": "Class (onsite)", "OL": "Class (online)"}
FREE_COLOR = "#F7F7F7"
GRID_COLOR = "#D0D0D0"
TARGET_COLOR = "#E8A200"

WIDE_RE = re.compile(r"^sched_([a-z]{3})_(\d{2})(\d{2})$")


# ------------------------------------------------------------------ parsing
def norm_day(value):
    return str(value).strip().lower()[:3]


def norm_state(value):
    if pd.isna(value):
        return "F"
    v = str(value).strip().lower()
    if v in ("", "f", "free", "0", "0.0", "nan", "none"):
        return "F"
    if v in ("ol", "online", "2", "2.0"):
        return "OL"
    return "ON"  # on, onsite, 1, class, anything else non-empty


def parse_hour(value):
    """'08:00' -> 8.0, '13:30' -> 13.5, '8' -> 8.0, datetime.time -> hours."""
    if hasattr(value, "hour"):
        return value.hour + value.minute / 60
    s = str(value).strip()
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?", s)
    if not m:
        raise ValueError(f"Cannot parse time: {value!r}")
    return int(m.group(1)) + int(m.group(2) or 0) / 60


def wide_to_blocks(row, wide_cols):
    """Return list of (day, start_hour, end_hour, state) merged into runs."""
    cells = {}
    for col, (day, hour) in wide_cols.items():
        state = norm_state(row[col])
        if state != "F":
            cells[(day, hour)] = state
    blocks = []
    for day in DAYS:
        hours = sorted(h for (d, h) in cells if d == day)
        i = 0
        while i < len(hours):
            start, state = hours[i], cells[(day, hours[i])]
            j = i + 1
            while j < len(hours) and hours[j] == hours[j - 1] + 1 and cells[(day, hours[j])] == state:
                j += 1
            blocks.append((day, start, hours[j - 1] + 1, state))
            i = j
    return blocks


def load_scenarios(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    wide_cols = {}
    for c in df.columns:
        m = WIDE_RE.match(c.lower())
        if m and m.group(1) in DAYS:
            wide_cols[c] = (m.group(1), int(m.group(2)) + int(m.group(3)) / 60)

    id_col = lower.get("scenario_id") or lower.get("scenario") or lower.get("id")

    scenarios = []  # list of dicts: id, meta, blocks
    if wide_cols:
        for idx, row in df.iterrows():
            sid = str(row[id_col]) if id_col else f"SC{idx + 1:03d}"
            scenarios.append({"id": sid, "meta": row, "blocks": wide_to_blocks(row, wide_cols)})
        return scenarios

    needed = {"day", "start_time", "end_time"}
    if needed.issubset(lower):
        if not id_col:
            raise ValueError("Long format needs a scenario_id column.")
        mod_col = lower.get("modality") or lower.get("class_modality")
        for sid, grp in df.groupby(id_col, sort=False):
            blocks = []
            for _, r in grp.iterrows():
                state = norm_state(r[mod_col]) if mod_col else "ON"
                state = "ON" if state == "F" else state
                blocks.append((norm_day(r[lower["day"]]), parse_hour(r[lower["start_time"]]),
                               parse_hour(r[lower["end_time"]]), state))
            scenarios.append({"id": str(sid), "meta": grp.iloc[0], "blocks": blocks})
        return scenarios

    raise ValueError(
        "Could not detect the layout. Expected wide columns like 'sched_mon_0800' "
        "or long columns 'scenario_id, day, start_time, end_time'.\n"
        f"Columns found: {list(df.columns)[:15]}..."
    )


# ------------------------------------------------------------------ drawing
def meta_title(meta):
    parts = []
    for key, fmt in (("density_tier", "{}"), ("meeting_length_min", "{} min meeting"),
                     ("meeting_modality", "{}"), ("meeting_type", "{}")):
        if key in meta.index and not pd.isna(meta[key]):
            parts.append(fmt.format(meta[key]))
    return " | ".join(parts)


def target_block(meta):
    """Return (day, start, end) of the chosen meeting if the CSV has it."""
    if "target_day" in meta.index and "target_start_time" in meta.index:
        if pd.isna(meta["target_day"]) or pd.isna(meta["target_start_time"]):
            return None
        length = meta["meeting_length_min"] / 60 if "meeting_length_min" in meta.index \
            and not pd.isna(meta["meeting_length_min"]) else 1
        start = parse_hour(meta["target_start_time"])
        return norm_day(meta["target_day"]), start, start + length
    return None


def draw_scenario(sc, out_path, start_hour, end_hour, dpi):
    n_days = len(DAYS)
    fig, ax = plt.subplots(figsize=(9, 0.55 * (end_hour - start_hour) + 1.8))

    # background grid
    for d in range(n_days):
        for h in range(start_hour, end_hour):
            ax.add_patch(Rectangle((d, h), 1, 1, facecolor=FREE_COLOR,
                                   edgecolor=GRID_COLOR, linewidth=0.6))

    # class blocks
    used = set()
    for day, s, e, state in sc["blocks"]:
        if day not in DAYS:
            continue
        d = DAYS.index(day)
        s, e = max(s, start_hour), min(e, end_hour)
        if e <= s:
            continue
        used.add(state)
        ax.add_patch(Rectangle((d + 0.04, s + 0.04), 0.92, (e - s) - 0.08,
                               facecolor=COLORS[state], edgecolor="white", linewidth=1))
        ax.text(d + 0.5, (s + e) / 2, LABELS[state].replace(" (", "\n("), ha="center",
                va="center", color="white", fontsize=7.5)

    # chosen meeting (only if the CSV includes target columns)
    tb = target_block(sc["meta"])
    if tb and tb[0] in DAYS:
        d = DAYS.index(tb[0])
        ax.add_patch(Rectangle((d + 0.04, tb[1] + 0.04), 0.92, (tb[2] - tb[1]) - 0.08,
                               facecolor="none", edgecolor=TARGET_COLOR, linewidth=2.5, linestyle="--"))
        ax.text(d + 0.5, (tb[1] + tb[2]) / 2, "Meeting", ha="center", va="center",
                color=TARGET_COLOR, fontsize=8, fontweight="bold")
        used.add("TARGET")

    # axes
    ax.set_xlim(0, n_days)
    ax.set_ylim(end_hour, start_hour)  # time runs top to bottom
    ax.set_xticks([d + 0.5 for d in range(n_days)])
    ax.set_xticklabels([DAY_LABELS[d] for d in DAYS], fontsize=10, fontweight="bold")
    ax.xaxis.tick_top()
    ax.set_yticks(range(start_hour, end_hour + 1))
    ax.set_yticklabels([f"{h:02d}:00" for h in range(start_hour, end_hour + 1)], fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    title = f"Scenario {sc['id']}"
    sub = meta_title(sc["meta"])
    fig.suptitle(title, fontsize=13, fontweight="bold", x=0.05, ha="left", y=0.99)
    if sub:
        fig.text(0.05, 0.93, sub, fontsize=9, color="#555555", ha="left")

    # legend
    handles = [Rectangle((0, 0), 1, 1, facecolor=COLORS[k]) for k in ("ON", "OL")]
    names = ["Onsite class", "Online class"]
    handles.append(Rectangle((0, 0), 1, 1, facecolor=FREE_COLOR, edgecolor=GRID_COLOR))
    names.append("Free")
    if "TARGET" in used:
        handles.append(Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=TARGET_COLOR, linestyle="--"))
        names.append("Chosen meeting")
    fig.legend(handles, names, loc="lower center", ncol=len(names), frameon=False, fontsize=8)

    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Render schedule scenarios as PNG files.")
    ap.add_argument("csv", help="Path to the scenarios CSV")
    ap.add_argument("--out", default="schedule_pngs", help="Output folder (default: schedule_pngs)")
    ap.add_argument("--start-hour", type=int, default=8, help="First hour shown (default 8)")
    ap.add_argument("--end-hour", type=int, default=18, help="Last hour shown, exclusive (default 18)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--limit", type=int, default=None, help="Only render the first N scenarios")
    args = ap.parse_args()

    scenarios = load_scenarios(args.csv)
    if args.limit:
        scenarios = scenarios[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for sc in scenarios:
        safe_id = re.sub(r"[^\w\-]", "_", sc["id"])
        draw_scenario(sc, out_dir / f"{safe_id}.png", args.start_hour, args.end_hour, args.dpi)
    print(f"Saved {len(scenarios)} PNGs to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
