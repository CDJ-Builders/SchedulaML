# Schedula MindLab (SML)

**Schedula MindLab** explores preference-aware machine learning for recommending feasible, personalized meeting slots around university students' weekly class schedules. The system transforms class schedule matrices and survey preference responses into viable candidate meeting windows, extracts rich calendar and behavioral features, and ranks candidates using multi-layer ranking architectures and machine learning estimators.
---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Dataset & Data Gathering](#dataset--data-gathering)
- [Feature Engineering (33 Features)](#feature-engineering-33-features)
- [Model Evaluation & Benchmarks](#model-evaluation--benchmarks)
- [System Architecture & Layered Inference](#system-architecture--layered-inference)
- [Notebooks](#notebooks)
- [CLI Scripts & Utilities](#cli-scripts--utilities)
- [Getting Started & Usage](#getting-started--usage)

---

## Project Overview

For each respondent and meeting request, the Schedula pipeline executes the following workflow:

1. **Input Ingestion:** Loads preprocessed primary survey records containing student schedule matrices (Monday–Saturday, 07:00–22:00), meeting requirements (length, modality, type), commute duration, tolerance for back-to-back classes, preferred break times, day/time preferences, and up to three ranked target choices.
2. **Candidate Slot Generation:** Generates all valid candidate meeting slots matching the requested meeting length within the Monday–Saturday 07:00–22:00 grid, strictly filtering out conflicting class hours.
3. **Graded Relevance Assignment:** Assigns ground-truth graded relevance scores to respondent choices: Rank 1 = 3, Rank 2 = 2, Rank 3 = 1. Non-selected feasible slots receive 0.
4. **Feature Extraction:** Extracts **33 dense features** for each candidate slot, capturing temporal position, calendar density, schedule gaps, modality compatibility, transit overhead, lunch overlaps, and positive/negative preference alignments.
5. **Ranker Evaluation & Stacking:** Evaluates candidate ranking performance across out-of-fold `GroupKFold` splits (grouped by respondent schedule clusters to prevent data leakage) across multiple L1 classifiers/regressors, L2 meta-rankers, and an L3 soft-penalty hybrid blend.

### Pipeline Statistics (Current Dataset v1.1.0)

| Metric | Value |
| :--- | :--- |
| **Total Survey Responses** | 205 |
| **Unique Schedule Scenarios** | 75 |
| **Inferred Respondent Preference Profiles** | 177 |
| **Cross-Validation Groups (Connected Components)** | 50 |
| **Operational Grid** | Monday–Saturday, 07:00–22:00 |
| **Queries with Usable Target Labels** | 197 |
| **Feasible Candidate Slots Generated** | 14,243 |
| **Retained Target Labels** | 581 |
| **Labels Dropped (Outside Grid / Occupied by Class)** | 5 outside grid / 29 occupied |
| **Extracted Features** | 33 |

---

## Repository Structure

```text
.
├── CDJ_SchedulaML_ACM.pdf              # Academic research paper (ACM format)
├── README.md                           # Project documentation and benchmarks
├── SML_SourceCode_Final.ipynb          # Core research notebook: EDA, CV, models & importances
├── SML_SchedulaImplementation.ipynb    # Engineering notebook: Nested CV, L1-L3 stacking & artifacts
│
├── artifacts/                          # Production deployment package (v1.1.0)
│   ├── config.json                     # System configuration and dataset metadata
│   ├── features.py                     # Zero-dependency candidate generation & feature engineering
│   ├── feature_schema.json             # Schema of 33 ordered features and expected input keys
│   ├── golden_fixtures.json            # Regression test fixtures for parity checking
│   ├── gradient_boosting_model.joblib  # Serialized Gradient Boosting L1 model
│   ├── random_forest_model.joblib      # Serialized Random Forest L1 model
│   ├── l1_models.joblib                # Serialized dictionary of all 8 Level-1 estimators
│   ├── l2_models.joblib                # Serialized Level-2 LightGBMRanker model
│   ├── metrics.json                    # Benchmark metric scores across all evaluated models
│   ├── predict_layered.py              # Pure standalone inference entry point (L1 -> L2 -> L3)
│   └── survey_rows.json                # Standardized survey rows for verification
│
├── datasets/                           # Data storage
│   ├── SML_PrimaryDataset.xlsx         # Raw survey response workbook (205 responses)
│   ├── SML_PrimaryPreprocessed.xlsx    # Standardized and cleaned primary survey dataset
│   ├── SML_SecondaryDataset.xlsx       # Secondary dataset of university class schedules (DLSU-D)
│   ├── SML_SecondaryDesign.xlsx        # Design specification for secondary schedule data
│   └── sample_scenarios.csv            # Pre-generated schedule scenarios (Light/Medium/Heavy)
│
├── gathering/                          # Schedule cards and survey instruments
│   ├── SML_SurveyPrinted.pdf           # Printed survey instrument used for onsite data collection
│   ├── online_sends/                   # PNG schedule cards sent to online respondents (21 cards)
│   ├── onsite_print/                   # Printed scenario cards for onsite collection (26 cards)
│   ├── primary/                        # Primary scenario schedule cards (49 cards)
│   └── survey_pngs/                    # Additional survey schedule scenario cards (51 cards)
│
└── scripts/                            # Utility scripts
    ├── generate_scenarios.py           # Scenario generation CLI (seeded or synthetic)
    └── plot_schedules.py               # Visualizer rendering schedule CSVs into PNG cards
```

---

## Dataset & Data Gathering

The dataset is partitioned into primary survey data and secondary institutional schedules:

1. **Primary Dataset ([`datasets/SML_PrimaryDataset.xlsx`](datasets/SML_PrimaryDataset.xlsx), [`datasets/SML_PrimaryPreprocessed.xlsx`](datasets/SML_PrimaryPreprocessed.xlsx)):**
   - 205 student responses evaluating realistic weekly class schedules.
   - Includes schedule occupancy matrices across 6 days (Monday through Saturday) from 07:00 to 22:00, with class modalities designated as Onsite (`ON`), Online (`OL`), or Free (`F`).
   - Captures meeting constraints: `meeting_length_min` (60 or 120 mins), `meeting_modality` (Onsite, Online, Any), `meeting_type` (Meeting, Project, Event), transit time (`Commute`), tolerance for consecutive classes (`Back-To-Back Tolerance`), `Preferred Break`, preferred/avoided days, and preferred/avoided time windows.
   - Contains up to 3 ranked target slot selections chosen by human respondents.

2. **Secondary Dataset ([`datasets/SML_SecondaryDataset.xlsx`](datasets/SML_SecondaryDataset.xlsx), [`datasets/SML_SecondaryDesign.xlsx`](datasets/SML_SecondaryDesign.xlsx)):**
   - Real class schedules from De La Salle University - Dasmariñas (DLSU-D) used as realistic seeds (student class days and start hours) to build balanced scenario tiers.
   - Pre-generated scenarios are saved in [`datasets/sample_scenarios.csv`](datasets/sample_scenarios.csv).

3. **Schedule Gathering Artifacts (`gathering/`):**
   - Contains 147 schedule visual cards used during physical and online survey administration (`online_sends/`, `onsite_print/`, `primary/`, `survey_pngs/`), as well as the printable survey form [`SML_SurveyPrinted.pdf`](gathering/SML_SurveyPrinted.pdf).

---

## Feature Engineering (33 Features)

The pipeline constructs exactly **33 tabular features** per candidate slot to represent temporal context, schedule density, user convenience, and preference alignment. The exact ordered features defined in [`artifacts/feature_schema.json`](artifacts/feature_schema.json) and [`artifacts/features.py`](artifacts/features.py) are:

| # | Feature Name | Description |
| :-: | :--- | :--- |
| 1 | `day_idx` | Day index (0 = Monday, ..., 5 = Saturday) |
| 2 | `is_saturday` | Binary indicator if the candidate slot is on Saturday |
| 3 | `overlaps_lunch` | Binary flag indicating overlap with the common lunch window (12:00–13:00) |
| 4 | `is_morning` | Binary flag for morning slots (starting before 12:00) |
| 5 | `is_late` | Binary flag for evening slots (starting at or after 17:00) |
| 6 | `gap_before` | Hours of free time between preceding class and slot start (or from day start) |
| 7 | `gap_after` | Hours of free time between slot end and subsequent class (or until day end) |
| 8 | `min_gap` | Minimum of `gap_before` and `gap_after` |
| 9 | `back_to_back` | Flag indicating whether the slot directly borders an existing class |
| 10 | `day_class_hours` | Total scheduled class hours on the candidate slot's day |
| 11 | `day_is_free` | Binary flag indicating if the student has zero classes on this day |
| 12 | `day_is_busiest` | Binary flag indicating if this day has the student's highest class load |
| 13 | `day_onsite_hours` | Number of onsite class hours scheduled on this day |
| 14 | `meeting_len_h` | Meeting duration in hours (1.0 or 2.0) |
| 15 | `meeting_onsite` | Binary indicator if the requested meeting is onsite |
| 16 | `type_project` | One-hot indicator for `meeting_type == 'Project'` |
| 17 | `type_event` | One-hot indicator for `meeting_type == 'Event'` |
| 18 | `tier_heavy` | One-hot indicator for Heavy schedule density tier |
| 19 | `tier_medium` | One-hot indicator for Medium schedule density tier |
| 20 | `n_feasible` | Total number of valid feasible candidate slots for this query |
| 21 | `total_class_hours`| Total weekly scheduled class hours across all 6 days |
| 22 | `pref_day` | Binary indicator if the candidate day is in the respondent's preferred days |
| 23 | `avoid_day` | Binary indicator if the candidate day is in the respondent's avoided days |
| 24 | `pref_time_share` | Fraction of candidate slot duration falling inside preferred time windows |
| 25 | `avoid_time_share` | Fraction of candidate slot duration falling inside avoided time windows |
| 26 | `break_shortfall` | Shortfall (in hours) relative to respondent's requested minimum break |
| 27 | `b2b_tolerance` | Respondent's numeric back-to-back tolerance score |
| 28 | `b2b_x_tolerance` | Interaction term: `back_to_back` multiplied by `b2b_tolerance` |
| 29 | `modality_match` | Match score between requested meeting modality and day's class modality |
| 30 | `pref_modality_none`| Flag indicating no specific modality preference was expressed |
| 31 | `commute_min` | Respondent's reported one-way commute time in minutes |
| 32 | `extra_trip_cost` | Estimated additional commute overhead (in minutes) for isolated onsite meetings |
| 33 | `online_next_to_class` | Flag for online meetings immediately adjacent to an onsite class (commute conflict) |

### Top Permutation & Gini Importances

Evaluated in `SML_SourceCode_Final.ipynb` and `SML_SchedulaImplementation.ipynb`:
- **Gradient Boosting:** `pref_time_share`, `pref_day`, `is_late`, `gap_before`, `min_gap`, `day_idx`.
- **Random Forest:** `is_late`, `pref_time_share`, `min_gap`, `day_idx`, `gap_before`, `is_saturday`.
- **Permutation Importance:** `gap_before` ($0.0590 \pm 0.0038$), `pref_time_share` ($0.0519 \pm 0.0029$), `pref_day` ($0.0458 \pm 0.0048$), `is_late` ($0.0447 \pm 0.0040$).

---

## Model Evaluation & Benchmarks

All models were evaluated using out-of-fold cross-validation on 50 connected respondent clusters to strictly eliminate leakage. Metrics are pooled across all 197 test queries (higher is better for all metrics):

| Model Level | Model / Scoring Method | NDCG@1 | NDCG@3 | NDCG@5 | Hit@1 | Hit@3 | MRR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | Random Guessing | 0.0167 | 0.0310 | 0.0447 | 0.0457 | 0.1269 | 0.1492 |
| **Baseline** | Popularity Prior | 0.1547 | 0.1807 | 0.2401 | 0.2132 | 0.4365 | 0.3755 |
| **Baseline** | Hand-crafted Rule Score | 0.1922 | 0.2149 | 0.2429 | 0.2183 | 0.4112 | 0.3619 |
| **Level 1** | Gaussian Naive Bayes (`pred_gnb`) | 0.0442 | 0.0798 | 0.1154 | 0.0761 | 0.2386 | 0.2355 |
| **Level 1** | K-Nearest Neighbors (`pred_knn`) | 0.1363 | 0.1719 | 0.2075 | 0.2234 | 0.4619 | 0.3840 |
| **Level 1** | Decision Tree (`pred_dt`) | 0.1769 | 0.2285 | 0.2655 | 0.2640 | 0.5228 | 0.4341 |
| **Level 1** | Random Forest (`pred_rf`) | 0.1907 | 0.2452 | 0.2994 | 0.3299 | 0.5533 | 0.4920 |
| **Level 1** | Extra Trees (`pred_et`) | 0.2103 | 0.2864 | 0.3408 | 0.3249 | 0.6142 | 0.5098 |
| **Level 1** | Logistic Regression (`pred_lr`) | 0.2263 | 0.2611 | 0.3268 | 0.3249 | 0.5787 | 0.4969 |
| **Level 1** | **Gradient Boosting (`pred_gb`)** | 0.2429 | **0.3315** | **0.3720** | 0.3198 | **0.6497** | 0.5139 |
| **Level 1** | **LightGBM (`pred_lgb`)** | **0.2451** | 0.3060 | 0.3676 | **0.3350** | 0.6447 | **0.5209** |
| **Level 2** | L2 Logistic Regression Meta-Stack | 0.2357 | 0.3199 | 0.3708 | 0.3299 | 0.6345 | 0.5185 |
| **Level 2** | L2 LightGBM Meta-Ranker (`pred_l2_lgb`)| 0.1893 | 0.2998 | 0.3514 | 0.2589 | 0.6294 | 0.4778 |
| **Level 3** | L3 Hybrid + Soft Penalty (`pred_l3`) | 0.1603 | 0.2170 | 0.2664 | 0.2284 | 0.4569 | 0.3912 |

> **Key Findings:**
> - Gradient Boosting (`pred_gb`) delivers the highest overall ranking accuracy (**NDCG@3 = 0.3315**, **NDCG@5 = 0.3720**, **Hit@3 = 64.97%**), followed closely by LightGBM (**NDCG@1 = 0.2451**, **MRR = 0.5209**).
> - All tuned ML models dramatically outperform standard heuristic rules and popularity baselines (NDCG@3 improvement of $>+54\%$).
> - Paired bootstrap hypothesis testing demonstrates statistically significant improvements over rule-based baselines ($p < 0.001$).

---

## System Architecture & Layered Inference

The production-ready pipeline in `artifacts/predict_layered.py` implements a 3-layer architecture:

```
[User Schedule + Meeting Spec]
               │
               ▼
   [Candidate Slot Generator]  ──────> 07:00–22:00 Mon–Sat Feasible Grid
               │
               ▼
    [33 Feature Extraction]   ──────> Temporal, Gaps, Density, Modality, Preferences
               │
               ▼
   ┌───────────────────────┐
   │ Layer 1: Base Models  │ ──────> GB, LGB, RF, LR, ET, DT, KNN, GNB
   └───────────────────────┘
               │
               ▼
   ┌───────────────────────┐
   │ Layer 2: Meta-Ranker  │ ──────> Z-score & Percentile Ranks of (GB, LGB, RF) + Rule
   └───────────────────────┘
               │
               ▼
   ┌───────────────────────┐
   │ Layer 3: Hybrid Blend │ ──────> 0.6 * P_meta + 0.4 * P_rule
   │   + Soft Penalty      │ ──────> -0.5 for avoid_day==1 or avoid_time_share>0.5
   └───────────────────────┘
               │
               ▼
  Top-K Ranked Recommendations [Day, Start, End, Score]
```

1. **Layer 1 (L1 Base Models):** Eight diverse estimators predict candidate scores based on the 33 extracted features.
2. **Layer 2 (L2 Meta-Ranker):** Normalizes the top 3 base predictors (`gb`, `lgb`, `rf`) into z-scores and percentile ranks, concatenates them with the rule heuristic, and ranks via a trained `LightGBMRanker`.
3. **Layer 3 (L3 Hybrid & Soft Avoidance Penalty):** Blends the meta-ranker score with the domain heuristic ($0.6 \times p_{\text{meta}} + 0.4 \times p_{\text{rule}}$). To prevent user starvation in heavily constrained schedules, candidate slots violating negative preferences (`avoid_day == 1.0` or `avoid_time_share > 0.5`) receive a soft deduction of `-0.5` rather than hard pruning.

---

## Notebooks

The repository contains two Jupyter notebooks documenting the research and production development:

1. **[`SML_SourceCode_Final.ipynb`](SML_SourceCode_Final.ipynb):**
   - Step 1: Cleaning & schedule matrix preprocessing.
   - Step 2: Feasible candidate slot generation and graded relevance assignment.
   - Step 3: 33-feature engineering pipeline.
   - Step 4: Model training, evaluation setups, and 5-fold `GroupKFold` cross-validation.
   - Step 5: Metric evaluation tables (NDCG@1..5, Hit@1..3, MRR) and feature importance analysis.
   - Step 6: Inference demonstration for sample survey queries.

2. **[`SML_SchedulaImplementation.ipynb`](SML_SchedulaImplementation.ipynb):**
   - Extends the core pipeline with production engineering steps:
   - Step 7: Leakage-free nested cross-validation and meta-stacking (L1, L2, L3).
   - Step 8: Comprehensive evaluation and paired bootstrap hypothesis testing (95% CI).
   - Step 9: Ablation studies, permutation feature importance, and error analysis.
   - Step 10: Standalone pure Python refactored inference and strict parity validation against golden fixtures.
   - Step 11: Build and packaging of deployment artifacts (`artifacts/`).

---

## CLI Scripts & Utilities

### 1. Scenario Generator (`scripts/generate_scenarios.py`)

Generates synthetic or secondary-seeded weekly schedule scenarios across Light (4–7 hrs), Medium (10–15 hrs), and Heavy (18–26 hrs) density tiers.

```bash
# Generate 50 scenarios from scratch
python scripts/generate_scenarios.py

# Seed scenarios using real university student schedules
python scripts/generate_scenarios.py --n 75 --seed 42 --secondary datasets/SML_SecondaryDataset.xlsx --out datasets/sample_scenarios.csv
```

### 2. Schedule Card Visualizer (`scripts/plot_schedules.py`)

Renders schedule CSVs (wide or long formats) into color-coded schedule PNGs matching DLSU-D school colors (Green for Onsite classes, Blue for Online classes):

```bash
python scripts/plot_schedules.py datasets/sample_scenarios.csv --out schedule_pngs --start-hour 7 --end-hour 22
```