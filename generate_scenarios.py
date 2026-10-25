"""
generate_scenarios.py - build the pre-generated part of the SchedulaML Primary Dataset.

Creates N weekly schedule scenarios (Mon-Sat, hourly blocks 08:00-17:00 starts) with
meeting-request conditions. Survey columns (preferences, targets) are NOT included;
they are collected from respondents.

Usage:
    python generate_scenarios.py                       # 50 scenarios -> sample_scenarios.csv
    python generate_scenarios.py --n 50 --seed 7 --secondary SML_SecondaryDataset.xlsx

If the secondary dataset is available, each scenario is seeded from a real student's
class days/start hours (rdb_schedule + valid rdb_timeslots), then filled up to the
density tier. Without it, scenarios are generated from scratch.
"""
import argparse
import hashlib
import random
from pathlib import Path

import pandas as pd

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat"]
HOURS = list(range(7, 18))            # block start hours 08..17 (grid ends 18:00)
DAY_WEIGHT = {"mon": 1, "tue": 1, "wed": 1, "thu": 1, "fri": 1, "sat": 0.35}
FULL_DAY = {"monday": "mon", "tuesday": "tue", "wednesday": "wed", "thursday": "thu", "friday": "fri", "saturday": "sat"}

# target total class hours per week for each density tier (inclusive range)
TIERS = {"Light": (4, 7), "Medium": (10, 15), "Heavy": (18, 26)}
BLOCK_LEN = [1, 2, 3]
BLOCK_LEN_W = [0.25, 0.45, 0.30]
P_ONLINE = 0.15
MEETING_TYPES = ["Project", "Meeting", "Consultation"]
MIN_FEASIBLE = 4


def load_anchors(path):
    """Return {student_id: [(day, start_hour), ...]} from valid timeslots, or {}."""
    if not path or not Path(path).exists():
        return {}
    x = pd.read_excel(path, sheet_name=["rdb_schedule", "rdb_timeslots"])
    sched, ts = x["rdb_schedule"], x["rdb_timeslots"].copy()
    ts["day"] = ts["day"].astype(str).str.lower().map(FULL_DAY)
    ts["sh"] = ts["start_time"].map(lambda t: t.hour)
    ts["eh"] = ts["end_time"].map(lambda t: t.hour)
    ts = ts[(ts["eh"] > ts["sh"]) & ts["sh"].between(HOURS[0], HOURS[-1]) & ts["day"].notna()]
    m = sched.merge(ts[["timeslot_id", "day", "sh"]], on="timeslot_id")
    return {sid: list(zip(g["day"], g["sh"])) for sid, g in m.groupby("student_id")}


def fits(grid, day, start, length):
    if start + length > HOURS[-1] + 1:
        return False
    return all(grid[day][h] == "F" for h in range(start, start + length))


def place(grid, day, start, length):
    state = "OL" if random.random() < P_ONLINE else "ON"
    for h in range(start, start + length):
        grid[day][h] = state


def n_feasible(grid, length):
    return sum(fits(grid, d, s, length) for d in DAYS for s in HOURS)


def build_grid(tier, anchors):
    lo, hi = TIERS[tier]
    target = random.randint(lo, hi)
    grid = {d: {h: "F" for h in HOURS} for d in DAYS}
    hours_used = 0

    for day, start in random.sample(anchors, len(anchors)) if anchors else []:
        if hours_used >= target:
            break
        length = random.choices(BLOCK_LEN, BLOCK_LEN_W)[0]
        if fits(grid, day, start, length):
            place(grid, day, start, length)
            hours_used += length

    tries = 0
    while hours_used < target and tries < 500:
        tries += 1
        day = random.choices(DAYS, [DAY_WEIGHT[d] for d in DAYS])[0]
        length = random.choices(BLOCK_LEN, BLOCK_LEN_W)[0]
        start = random.choice(HOURS)
        if fits(grid, day, start, length):
            place(grid, day, start, length)
            hours_used += length
    return grid


def derived(grid, length):
    occ = {d: [h for h in HOURS if grid[d][h] != "F"] for d in DAYS}
    all_occ = [h for d in DAYS for h in occ[d]]
    longest, gaps = 0, 0
    for d in DAYS:
        run = 0
        for h in HOURS:
            run = run + 1 if grid[d][h] == "F" else 0
            longest = max(longest, run)
        hs = occ[d]
        gaps += sum(1 for a, b in zip(hs, hs[1:]) if b > a + 1)
    out = {"total_class_hours": len(all_occ)}
    out.update({f"class_hours_{d}": len(occ[d]) for d in DAYS})
    out.update({
        "n_class_days": sum(1 for d in DAYS if occ[d]),
        "earliest_class_start": min(all_occ) if all_occ else "",
        "latest_class_end": (max(all_occ) + 1) if all_occ else "",
        "longest_free_block_hrs": longest,
        "n_inter_class_gaps": gaps,
        "n_feasible_slots": n_feasible(grid, length),
    })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--secondary", default="/mnt/user-data/uploads/SML_SecondaryDataset.xlsx")
    ap.add_argument("--out", default="sample_scenarios.csv")
    args = ap.parse_args()
    random.seed(args.seed)

    anchors_by_student = load_anchors(args.secondary)
    student_ids = list(anchors_by_student)
    print(f"Anchor students available: {len(student_ids)}")

    # balanced conditions: tier x length x modality cycles evenly
    tiers = list(TIERS)
    conditions = [(tiers[i % 3], [60, 120][(i // 3) % 2], ["Online", "Onsite"][(i // 6) % 2]) for i in range(args.n)]
    random.shuffle(conditions)

    rows = []
    for i, (tier, length_min, modality) in enumerate(conditions, 1):
        length = length_min // 60
        for attempt in range(200):
            sid = random.choice(student_ids) if student_ids else f"synthetic-{i}-{attempt}"
            grid = build_grid(tier, anchors_by_student.get(sid, []))
            if n_feasible(grid, length) >= MIN_FEASIBLE:
                break
        else:
            raise RuntimeError(f"Could not build scenario {i} with >= {MIN_FEASIBLE} feasible slots")

        row = {
            "scenario_id": f"SC{i:03d}",
            "source_student_hash": hashlib.sha1(str(sid).encode()).hexdigest()[:10],
            "density_tier": tier,
            "meeting_length_min": length_min,
            "meeting_modality": modality,
            "meeting_type": random.choice(MEETING_TYPES),
        }
        for d in DAYS:
            for h in HOURS:
                row[f"sched_{d}_{h:02d}00"] = grid[d][h]
        row.update(derived(grid, length))
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} scenarios x {df.shape[1]} columns to {args.out}")


if __name__ == "__main__":
    main()
