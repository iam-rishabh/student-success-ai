# Testing, Validation, and Minimal Data Models — Implementation Plan v2.0

> **Revision Note:** This plan is a comprehensive upgrade of v1.0. It closes identified technical bottlenecks, adds missing ML model logic per `specifications.md`, and provides agent-executable detail across all layers.

---

## Table of Contents
1. [Critical Issues Found in v1.0](#1-critical-issues-found-in-v10)
2. [Revised Architectural Modes](#2-revised-architectural-modes)
3. [ML Pipeline Deep Dive (Primary Focus)](#3-ml-pipeline-deep-dive-primary-focus)
4. [Data Models — Revised & Extended](#4-data-models--revised--extended)
5. [Backend & API Layer](#5-backend--api-layer)
6. [Scheduler & Refresh Logic](#6-scheduler--refresh-logic)
7. [Frontend Decisions](#7-frontend-decisions)
8. [Testing Suite — Complete Coverage Plan](#8-testing-suite--complete-coverage-plan)
9. [Revised Code Structure](#9-revised-code-structure)
10. [Open Questions & Decisions Required](#10-open-questions--decisions-required)
11. [Verification Plan](#11-verification-plan)

---

## 1. Critical Issues Found in v1.0

These are hard blockers or silent logic gaps that **must be resolved before any code is written**.

### 1.1 ML Model Gaps (High Priority)

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| M1 | No multi-output model design — specs require **3 outputs simultaneously**: P(placement \| 3mo), P(placement \| 6mo), P(placement \| 12mo) + salary range | Wrong model architecture entirely | Use a multi-output wrapper or separate calibrated heads per horizon |
| M2 | `XGBoost` listed as "primary" with no calibration step | Raw XGBoost probabilities are poorly calibrated; uncalibrated scores fed to lenders violates explainability intent | Add `CalibratedClassifierCV` (isotonic or platt) post-training |
| M3 | No feature engineering spec — CGPA alone is insufficient; specs require academic trajectory (trend), not just a point value | Model will underfit; trajectory signals missed | Engineer `cgpa_trend`, `semester_delta`, `rank_percentile` features explicitly |
| M4 | SHAP/LIME described as "simulated" in MVP — but the spec says explainability is **mandatory for lender outputs**, not optional | Non-compliant with RBI guidelines and spec §3.3 | Even in MVP, run real SHAP (TreeExplainer is fast; acceptable for local testing) |
| M5 | Psycho-behavioral scores listed as optional inputs but **no fallback logic** defined for absent consent | Model inference breaks or silently uses zero-fill | Explicit `None`-safe feature pipelines; psych features must be masked not zero-filled |
| M6 | No model versioning or experiment tracking in testing mode | Can't reproduce or compare test runs | Use MLflow locally even in simulation mode; log all runs |
| M7 | Salary range prediction described with no algorithm detail — classification vs regression undefined | Agent cannot implement this | Specify: use `XGBRegressor` for point estimate + quantile regression (q=0.15, q=0.85) for range bands |
| M8 | No cold-start strategy for new institutes with no historical placement data | Model will produce garbage predictions for pilot institutes | Define: fall back to national sector-level priors; flag predictions as "low-confidence" in output schema |
| M9 | Drift detection described only in production — but spec says "quarterly retraining on closed-loop data" must be validated | No way to simulate or test drift logic in MVP | Implement PSI (Population Stability Index) and feature drift checks even in simulation |
| M10 | No threshold calibration for risk tier classification (low/medium/high) | Default 0.5 threshold is almost always wrong; risk tiers will be miscalibrated | Derive thresholds from ROC curve at desired TPR/FPR operating points; store as config |

### 1.2 Data Model Gaps

| # | Issue | Fix |
|---|-------|-----|
| D1 | `InternshipRecord` has "performance notes" as a free-text field — entirely unstructured and un-featurizable | Add structured sub-fields: `role_relevance_score` (1–5), `employer_tier` (enum), `duration_weeks` (int) |
| D2 | No `InstituteEmbedding` or tier normalization — "Institute tier" is mentioned in specs but undefined | Add `institute_tier` as an ordinal enum + `historic_placement_rate_3mo/6mo/12mo` as float fields |
| D3 | `MacroIndicators` schema missing — sector-level hiring index, region job density not modeled | Add `MacroIndicators` schema (see §4) |
| D4 | `ConsentLayer` has no timestamp or version field | DPDP 2023 requires auditable consent; add `consented_at: datetime`, `consent_version: str`, `revoked_at: Optional[datetime]` |
| D5 | No `PredictionOutput` schema — model outputs are not validated before reaching API consumers | Add `SuccessScoreOutput` Pydantic model with typed fields for all 5 output dimensions |

### 1.3 Architectural / Infrastructure Gaps

| # | Issue | Fix |
|---|-------|-----|
| A1 | Feature Store is only mentioned for production — but in simulation, there is no specification for how features are assembled before inference | Implement a lightweight `FeatureAssembler` class used in both modes; swap the data source (mock vs store) via DI |
| A2 | No idempotency design for the scheduler — if a refresh job fires twice (e.g., APScheduler restart), duplicate ML inferences may corrupt state | Add job ID + last-run timestamp guard in `scheduler.py` |
| A3 | No async/sync boundary plan — FastAPI is async but `XGBoost` inference and SHAP are synchronous CPU-bound operations | Wrap inference in `asyncio.run_in_executor` with a `ThreadPoolExecutor` to prevent event loop blocking |
| A4 | No logging/observability layer defined | Add structured logging (JSON format via `structlog`) + request-level trace IDs for all API calls |

---

## 2. Revised Architectural Modes

### Mode A — Simulation / Testing (MVP/Local)

```
Data Layer:   In-memory SQLite + mocked JSON fixtures (realistic synthetic data)
Scheduler:    APScheduler — interval: 2 minutes (configurable via env var REFRESH_INTERVAL_SECONDS)
ML Inference: Local XGBoost + Random Forest + Logistic Regression ensemble
              Real SHAP (TreeExplainer) — NOT mocked
              MLflow local tracking (./mlruns)
Feature Store: FeatureAssembler class reading from SQLite
Calibration:  CalibratedClassifierCV (isotonic) applied post-fit on local validation split
Thresholds:   Derived from local synthetic data ROC; stored in ./config/thresholds.json
Drift:        PSI computed on each refresh cycle; alert if PSI > 0.2
```

### Mode B — Production (Deployable)

```
Data Layer:   PostgreSQL (relational) + Redis Feature Store (or Feast for managed feature serving)
Scheduler:    Celery + Redis/RabbitMQ broker — retraining cadence: every 45 days OR drift-triggered
ML Inference: MLflow Model Registry (S3 backend) — versioned XGBoost/Ensemble artifacts
              SHAP computed in dedicated worker pods (offloaded from API)
              Explainability cache layer (Redis TTL = 24h) for repeated student lookups
Feature Store: Feast or custom Redis feature store with online/offline split
Calibration:  Pre-applied during MLflow model registration step
Thresholds:   Stored in Postgres config table; versioned per model run
Consent:      DPDP-compliant consent engine with audit trail in Postgres
```

### Mode Switching

All mode-specific dependencies are injected via `src/core/dependencies.py`. The environment variable `APP_ENV=simulation|production` determines which implementations are bound. No `if/else` blocks in business logic.

---

## 3. ML Pipeline Deep Dive (Primary Focus)

> This section is the authoritative specification for all ML implementation. Any ambiguity elsewhere is resolved by this section.

### 3.1 Problem Formulation

The system solves **two distinct ML tasks simultaneously**:

**Task 1 — Placement Probability (Multi-Horizon Classification)**
- Three binary classifiers: P(placed | 3mo), P(placed | 6mo), P(placed | 12mo)
- Each classifier is independently trained but shares a common feature pipeline
- Outputs are calibrated probabilities in [0, 1]
- Risk tier derived from P(placed | 12mo): HIGH if < 0.40, MEDIUM if 0.40–0.70, LOW if > 0.70
  - These thresholds are **initial defaults**; recalibrate on real data after pilot

**Task 2 — Salary Range Estimation (Quantile Regression)**
- Point estimate: `XGBRegressor` predicting median expected salary (in ₹ LPA)
- Lower bound: Quantile regression at q=0.15 (15th percentile)
- Upper bound: Quantile regression at q=0.85 (85th percentile)
- Output: `{ median_lpa: float, lower_lpa: float, upper_lpa: float }`

### 3.2 Feature Engineering Specification

All features must be computed deterministically by `FeatureAssembler`. The agent must implement each feature as a named transform.

#### Academic Features
```
cgpa_normalized         : cgpa / 10.0  (or / 100.0 if percentage-based; detect via field flag)
cgpa_trend              : average semester-over-semester CGPA delta (last 3 semesters)
cgpa_consistency        : std dev of semester CGPA (lower = more consistent)
course_type_encoded     : label-encoded course type (MBA, B.Tech, MCA, etc.)
year_of_study           : integer (1–4 for UG, 1–2 for PG)
certifications_count    : integer count of verified skill certifications
```

#### Internship Features
```
internship_duration_weeks   : total weeks across all internships
internship_count            : number of distinct internships
employer_tier_best          : highest tier employer (ordinal: 1=MNC, 2=mid-size, 3=startup, 4=none)
role_relevance_avg          : mean role_relevance_score across internships (1–5 scale)
has_verified_internship     : boolean (institute-verified vs student-uploaded)
```

#### Institute Features
```
institute_tier              : ordinal (1=Tier-1, 2=Tier-2, 3=Tier-3)
placement_rate_3mo          : historical 3-month placement rate (0.0–1.0)
placement_rate_6mo          : historical 6-month placement rate
placement_rate_12mo         : historical 12-month placement rate
placement_rate_trend        : YoY change in 12mo placement rate (latest – prior year)
median_salary_lpa           : institute-level median salary for batch
recruiter_participation     : normalized recruiter count for last annual batch
```

#### Macro / Labor-Market Features
```
sector_hiring_index         : normalized hiring activity index for student's target sector (0.0–1.0)
region_job_density          : job postings per 1000 graduates in student's target region
macro_gdp_growth_proxy      : recent quarter GDP growth rate (public data feed)
sector_demand_trend         : 3-month change in sector_hiring_index
```

#### Behavioral / Psycho Features (Optional — Must Be Nullable)
```
grit_score                  : Grit-S 8-item scale score (1.0–5.0) | None if not consented
growth_mindset_score        : 3–6 item scale (1.0–6.0) | None if not consented
self_efficacy_score         : 6-item academic scale (1.0–7.0) | None if not consented
job_portal_activity_index   : normalized weekly activity proxy (0.0–1.0) | None if not consented
resume_update_count         : integer count of resume updates in last 90 days | None if not consented
```

**Critical Rule for Optional Features:** These must be passed through a `NullableFeatureImputer` that applies **median imputation using institute-cohort medians**, NOT global means. Zero-filling psycho scores will bias the model. Mark imputed rows with `is_imputed_psych: bool` for SHAP audit trail.

### 3.3 Ensemble Architecture

```
Input Features (assembled by FeatureAssembler)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│               Feature Preprocessing Pipeline             │
│  • StandardScaler for continuous features               │
│  • OrdinalEncoder for ordinal categoricals              │
│  • OneHotEncoder for nominal categoricals               │
│  • NullableFeatureImputer for optional signals          │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│         Ensemble (Per Task)       │
│                                   │
│  XGBoostClassifier  (weight: 0.5) │  ← Primary; high performance on tabular
│  RandomForestClassif(weight: 0.3) │  ← Variance reduction; handles missing
│  LogisticRegression (weight: 0.2) │  ← Interpretable baseline; audit anchor
│                                   │
│  Combination: soft-voting         │
└──────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│    CalibratedClassifierCV        │
│    method='isotonic', cv=5       │  ← Converts raw ensemble scores to
└──────────────────────────────────┘     well-calibrated probabilities
         │
         ▼
┌──────────────────────────────────┐
│       Threshold Config           │
│    (./config/thresholds.json)    │  ← Risk tier assignment per horizon
└──────────────────────────────────┘
```

**Salary Estimation (separate pipeline):**
```
Same preprocessing pipeline
         │
         ▼
XGBRegressor (objective='reg:squarederror')    → median_lpa
XGBRegressor (objective='reg:quantileerror',   → lower_lpa
              quantile_alpha=0.15)
XGBRegressor (objective='reg:quantileerror',   → upper_lpa
              quantile_alpha=0.85)
```

### 3.4 Explainability Implementation

**Do NOT mock SHAP in any mode.** `TreeExplainer` on a local XGBoost model runs in < 100ms for a single student.

```python
# Required implementation in src/ml/explainer.py

import shap

class PlacementExplainer:
    def __init__(self, xgb_model, feature_names: list[str]):
        self.explainer = shap.TreeExplainer(xgb_model)
        self.feature_names = feature_names

    def explain(self, feature_vector: np.ndarray) -> dict:
        shap_values = self.explainer.shap_values(feature_vector)
        # Return top-3 positive and top-3 negative drivers
        # Map to plain-English labels (see feature_label_map.json)
        # Output format required by SuccessScoreOutput.shap_drivers
        return {
            "top_positive_drivers": [...],   # e.g. ["Strong institute placement rate"]
            "top_negative_drivers": [...],   # e.g. ["Low internship exposure"]
            "raw_shap": dict(zip(self.feature_names, shap_values[0]))
        }
```

**LIME** is supplementary (used for sanity-checking SHAP on individual edge cases), not for every inference. Implement as an on-demand endpoint, not inline with prediction.

**Plain-English Label Mapping** (`src/ml/feature_label_map.json`): Every feature name must have a human-readable label. This is a mandatory artifact. Example:
```json
{
  "placement_rate_12mo": "Institute 12-month placement track record",
  "sector_hiring_index": "Current sector hiring activity",
  "cgpa_trend": "Academic performance trajectory",
  "internship_duration_weeks": "Internship experience depth"
}
```

### 3.5 Model Training Flow (Simulation Mode)

```
1. Generate synthetic student records (src/data/mock_generator.py)
   - 500 records minimum for meaningful training in simulation
   - Realistic distributions: CGPA ~ N(7.2, 0.8), clipped [4.0, 10.0]
   - Placement labels derived from weighted rule-based logic (not random)
   - Salary labels derived from institute_tier + course_type + sector_hiring_index

2. Feature assembly via FeatureAssembler

3. Train-validation split: 80/20 stratified by placement_label

4. Fit preprocessing pipeline on train split only (no leakage)

5. Train all 3 classifiers + 3 regressors

6. Calibrate classifiers on validation split

7. Compute SHAP values on validation set (log to MLflow)

8. Derive risk thresholds from validation ROC:
   - HIGH threshold: FPR < 0.15 operating point
   - LOW threshold: TPR > 0.90 operating point
   - Save to config/thresholds.json

9. Log all metrics to MLflow:
   - AUC-ROC per horizon
   - Calibration curve (Brier score)
   - Salary MAE and quantile coverage (should be ~70% for 15th-85th band)
   - Feature importance rankings
```

### 3.6 Cold-Start Strategy

For new institutes with < 2 years of historical placement data:

```
1. Use national sector-level placement priors (pre-computed from public NIRF/AICTE data)
2. Set institute_tier based on available signals (NAAC grade, NIRF rank if available)
3. Substitute missing placement_rate_* with sector_cohort_medians
4. Flag all predictions: confidence_level = "LOW" | "MEDIUM" | "HIGH"
   - LOW: new institute (<2yr data) OR >30% features imputed
   - MEDIUM: partial data OR psych features absent
   - HIGH: full feature set available
5. Never suppress low-confidence predictions — always show with explicit flag to lender
```

### 3.7 Drift Detection (Required in Both Modes)

```python
# src/ml/drift.py

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index. PSI > 0.2 = significant drift."""
    ...

class DriftMonitor:
    def check_feature_drift(self, reference_df, current_df) -> dict:
        """Returns per-feature PSI dict + overall_drift_flag bool"""
        ...
    
    def check_prediction_drift(self, reference_probs, current_probs) -> dict:
        """Checks if score distribution has shifted"""
        ...
```

**Thresholds:**
- PSI < 0.10 → No action
- PSI 0.10–0.20 → Log warning, monitor
- PSI > 0.20 → Trigger retraining job (even in simulation mode, log this event)

---

## 4. Data Models — Revised & Extended

All schemas live in `src/models/schemas.py`. All fields follow data minimization (DPDP 2023).

### 4.1 `AcademicProfile`
```python
class AcademicProfile(BaseModel):
    course_type: CourseTypeEnum           # MBA, BTech, MCA, BBA, etc.
    year_of_study: int = Field(ge=1, le=6)
    cgpa: float = Field(ge=0.0, le=10.0)  # Normalized to 10-point scale
    cgpa_scale: Literal["10", "100"]      # Detect percentage vs GPA
    semester_cgpas: list[float] = Field(min_length=1, max_length=12)
    certifications: list[str] = Field(default_factory=list, max_length=20)
    relevant_coursework: list[str] = Field(default_factory=list)
```

### 4.2 `InternshipRecord`
```python
class InternshipRecord(BaseModel):
    duration_weeks: int = Field(ge=0, le=104)
    employer_tier: EmployerTierEnum       # MNC | MidSize | Startup | None
    role_relevance_score: int = Field(ge=1, le=5)   # 1=unrelated, 5=highly relevant
    is_institute_verified: bool
    # No free-text performance notes stored (data minimization)
```

### 4.3 `InstituteContext`
```python
class InstituteContext(BaseModel):
    institute_tier: Literal[1, 2, 3]
    placement_rate_3mo: float = Field(ge=0.0, le=1.0)
    placement_rate_6mo: float = Field(ge=0.0, le=1.0)
    placement_rate_12mo: float = Field(ge=0.0, le=1.0)
    placement_rate_yoy_delta: float       # Positive = improving
    median_salary_lpa: float = Field(ge=0.0)
    recruiter_participation_normalized: float = Field(ge=0.0, le=1.0)
    data_vintage_months: int              # How old is this data? Used for confidence_level
```

### 4.4 `MacroIndicators` ← NEW (was missing in v1.0)
```python
class MacroIndicators(BaseModel):
    target_sector: SectorEnum             # IT, BFSI, Healthcare, Manufacturing, etc.
    sector_hiring_index: float = Field(ge=0.0, le=1.0)
    sector_demand_trend_3mo: float        # Delta over last 3 months
    region_job_density: float = Field(ge=0.0)   # Jobs per 1000 graduates
    macro_gdp_growth_pct: float           # Latest quarter GDP growth
    data_as_of: date                      # Staleness tracking
```

### 4.5 `PsychoSignals` ← NEW (was missing in v1.0)
```python
class PsychoSignals(BaseModel):
    grit_score: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    growth_mindset_score: Optional[float] = Field(default=None, ge=1.0, le=6.0)
    self_efficacy_score: Optional[float] = Field(default=None, ge=1.0, le=7.0)
    administered_at: Optional[datetime] = None
    # Never store raw item responses — only aggregated scores
```

### 4.6 `BehavioralSignals` ← NEW (was missing in v1.0)
```python
class BehavioralSignals(BaseModel):
    job_portal_activity_index: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    resume_update_count_90d: Optional[int] = Field(default=None, ge=0)
    interview_pipeline_stage: Optional[InterviewStageEnum] = None
    # Raw behavioral data not stored; only summarized proxies
```

### 4.7 `ConsentLayer` (Revised)
```python
class ConsentLayer(BaseModel):
    academic_data_consent: bool
    internship_data_consent: bool
    psycho_signals_consent: bool = False
    behavioral_signals_consent: bool = False
    lender_sharing_consent: bool
    consented_at: datetime
    consent_version: str                  # e.g., "v1.2" — maps to consent text in registry
    revoked_at: Optional[datetime] = None

    @model_validator(mode='after')
    def validate_mandatory_consents(self):
        if not self.academic_data_consent or not self.lender_sharing_consent:
            raise ValueError("Core consents are mandatory for system participation")
        return self
```

### 4.8 `StudentIngestionPayload` (Revised)
```python
class StudentIngestionPayload(BaseModel):
    student_id: str = Field(pattern=r'^STU-[A-Z0-9]{8}$')  # Pseudonymized ID
    academic: AcademicProfile
    internships: list[InternshipRecord] = Field(default_factory=list)
    institute: InstituteContext
    macro: MacroIndicators
    consent: ConsentLayer
    psych: Optional[PsychoSignals] = None
    behavioral: Optional[BehavioralSignals] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode='after')
    def validate_optional_signals_against_consent(self):
        if self.psych and not self.consent.psycho_signals_consent:
            raise ValueError("Psych signals provided without consent")
        if self.behavioral and not self.consent.behavioral_signals_consent:
            raise ValueError("Behavioral signals provided without consent")
        return self
```

### 4.9 `SuccessScoreOutput` ← NEW (was entirely missing in v1.0)
```python
class PlacementHorizonScore(BaseModel):
    horizon_months: Literal[3, 6, 12]
    probability: float = Field(ge=0.0, le=1.0)  # Calibrated
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]

class SalaryEstimate(BaseModel):
    median_lpa: float
    lower_lpa: float    # 15th percentile
    upper_lpa: float    # 85th percentile

class ShapDriver(BaseModel):
    feature_label: str  # Plain-English label from feature_label_map.json
    direction: Literal["positive", "negative"]
    magnitude: float    # Absolute SHAP value

class SuccessScoreOutput(BaseModel):
    student_id: str
    placement_scores: list[PlacementHorizonScore]  # 3 entries (3/6/12mo)
    salary_estimate: SalaryEstimate
    shap_drivers: list[ShapDriver]   # Top 6 (3 positive, 3 negative)
    confidence_level: Literal["LOW", "MEDIUM", "HIGH"]
    is_exception_alert: bool         # True if HIGH risk on any horizon
    next_best_actions: list[str]     # Advisory strings; max 3
    model_version: str               # MLflow run ID
    predicted_at: datetime
```

---

## 5. Backend & API Layer

### 5.1 FastAPI Endpoints

```
POST /v1/students/ingest
    Body: StudentIngestionPayload
    Response: { student_id, status, ingested_at }

POST /v1/predictions/score
    Body: { student_id: str }
    Response: SuccessScoreOutput
    Note: Triggers FeatureAssembler → Ensemble → SHAP → Output schema

GET  /v1/predictions/{student_id}/explain
    Response: Full SHAP breakdown + LIME comparison (on-demand)

GET  /v1/alerts/exceptions
    Query params: risk_tier=HIGH, horizon=12
    Response: Paginated list of SuccessScoreOutput for flagged students
    Note: "Exceptions-only" alerting per spec §3.4

GET  /v1/health
    Response: { status, model_version, last_refresh_at, drift_psi }

POST /v1/consent/revoke
    Body: { student_id: str }
    Action: Marks consent revoked; purges optional signal data; retains only pseudonymized core record
```

### 5.2 Async/Sync Boundary

```python
# src/api/routers/predictions.py

@router.post("/score")
async def get_success_score(payload: ScoreRequest, predictor: Predictor = Depends()):
    # Run CPU-bound inference in thread pool to not block event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        predictor.executor,  # ThreadPoolExecutor(max_workers=4)
        predictor.predict,
        payload.student_id
    )
    return result
```

### 5.3 Dependency Injection Map

```python
# src/core/dependencies.py

APP_ENV = os.getenv("APP_ENV", "simulation")

def get_feature_store():
    if APP_ENV == "simulation":
        return SQLiteFeatureStore()
    return RedisFeatureStore()

def get_model_registry():
    if APP_ENV == "simulation":
        return LocalMLflowRegistry()
    return RemoteMLflowRegistry()

def get_scheduler():
    if APP_ENV == "simulation":
        return APSchedulerService(interval_seconds=120)
    return CelerySchedulerService(interval_days=45)
```

---

## 6. Scheduler & Refresh Logic

```python
# src/jobs/scheduler.py

class RefreshJob:
    def __init__(self, feature_store, model_registry, drift_monitor):
        self.job_id = uuid4()
        self.last_run: Optional[datetime] = None

    def run(self):
        # Idempotency guard
        if self.last_run and (datetime.utcnow() - self.last_run).seconds < MIN_INTERVAL:
            logger.warning("Refresh attempted before minimum interval. Skipping.")
            return

        # 1. Pull fresh data from feature store
        # 2. Run PSI drift check against reference distribution
        # 3. IF drift detected OR scheduled retraining: retrain ensemble
        # 4. IF retraining: recalibrate, re-derive thresholds, log to MLflow
        # 5. Update last_run timestamp
        # 6. Publish "model_refreshed" event for API to pick up new version
```

---

## 7. Frontend Decisions

No changes from v1.0 stack (Next.js + TailwindCSS + Shadcn UI). Additional specifics:

- **Lender Dashboard:** Must render `SuccessScoreOutput` with SHAP waterfall chart per student (Recharts). Display `confidence_level` badge prominently. Exception alerts use a dedicated filtered view.
- **Student View:** Show only `next_best_actions` + timeline probability (no raw probabilities or SHAP scores shown to students — per spec, outputs are advisory, not alarming).
- **Consent Flow:** Gate all optional signal collection behind explicit UI toggles. Map to `ConsentLayer` fields. `consent_version` must match backend registry.

---

## 8. Testing Suite — Complete Coverage Plan

### 8.1 `tests/test_data_models.py`

```
- Valid payload ingestion (all fields populated)
- Boundary: cgpa=0.0, cgpa=10.0, cgpa=-0.1 (must fail), cgpa=10.1 (must fail)
- Mandatory consent false → ValidationError
- Psych data without psych consent → ValidationError
- student_id format validation (regex pattern)
- semester_cgpas empty list → ValidationError
- Consent revoked_at before consented_at → ValidationError
- SuccessScoreOutput: probability > 1.0 must fail
- SalaryEstimate: lower_lpa > upper_lpa must fail (add cross-field validator)
```

### 8.2 `tests/test_ml_pipeline.py`

```
- Feature assembly: all 27 named features present in output vector
- Nullable feature imputation: None psych scores → cohort median, not zero
- Ensemble output shape: (n_samples, 3) for 3 classifiers, not (n_samples, 1)
- Calibration: output probabilities in [0, 1] after calibration
- SHAP: shap_values.shape == feature_vector.shape
- SHAP: sum of SHAP values ≈ (prediction - expected_value) within epsilon
- Salary: lower_lpa <= median_lpa <= upper_lpa for all samples
- Cold-start: institute with data_vintage_months > 24 → confidence_level = "LOW"
- Risk tier thresholds loaded from config; not hardcoded
- Model version logged to MLflow on every training run
- Latency: single-student inference < 500ms (including SHAP) on local hardware
- Latency: batch of 50 students < 5s
```

### 8.3 `tests/test_refresh_logic.py`

```
- 2-minute interval fires correctly (mock time)
- Idempotency: double-fire within interval → second run skipped
- PSI > 0.2 → retraining triggered
- PSI < 0.1 → no retraining
- After retraining: model_version in registry is updated
- Drift monitor: synthetic distribution shift detected correctly
- Scheduler recovers gracefully from a failed training run (no state corruption)
```

### 8.4 `tests/test_api.py` ← NEW

```
- POST /ingest: valid payload → 200 + student_id
- POST /ingest: invalid cgpa → 422 with field-level error
- POST /score: unknown student_id → 404
- POST /score: response matches SuccessScoreOutput schema exactly
- GET /alerts/exceptions: only HIGH-risk students returned
- POST /consent/revoke: optional signal fields purged from store
- All endpoints return request trace ID in response headers
- Async: concurrent 10 POST /score requests complete without event loop blocking
```

### 8.5 `tests/test_explainability.py` ← NEW

```
- SHAP drivers: always exactly 3 positive + 3 negative in output
- Feature labels: all SHAP driver labels present in feature_label_map.json
- Plain-English labels: no raw feature names (snake_case) appear in API response
- SHAP values: consistent between two identical inputs (determinism check)
```

---

## 9. Revised Code Structure

```
src/
├── models/
│   └── schemas.py              # All Pydantic models (§4)
├── ml/
│   ├── feature_engineering.py  # 27 named feature transforms
│   ├── feature_assembler.py    # FeatureAssembler class + NullableFeatureImputer
│   ├── ensemble.py             # Multi-output ensemble + calibration wrapper
│   ├── salary_estimator.py     # Quantile regression models
│   ├── explainer.py            # SHAP TreeExplainer + plain-English mapping
│   ├── drift.py                # PSI + DriftMonitor
│   ├── trainer.py              # Full training flow (§3.5)
│   └── predictor.py            # Unified inference interface
├── data/
│   ├── mock_generator.py       # Synthetic data generation (500+ records)
│   └── feature_label_map.json  # Human-readable feature names
├── config/
│   ├── thresholds.json         # Risk tier thresholds (derived, not hardcoded)
│   └── settings.py             # Pydantic Settings (env var parsing)
├── api/
│   ├── main.py                 # FastAPI app init + lifespan handler
│   ├── routers/
│   │   ├── ingestion.py
│   │   ├── predictions.py
│   │   ├── alerts.py
│   │   └── consent.py
│   └── middleware.py           # Trace ID injection + structured logging
├── core/
│   ├── dependencies.py         # DI bindings (simulation vs production)
│   └── logging.py              # structlog JSON config
├── jobs/
│   └── scheduler.py            # Idempotent refresh job (§6)
└── stores/
    ├── base.py                 # Abstract FeatureStore interface
    ├── sqlite_store.py         # Simulation implementation
    └── redis_store.py          # Production implementation

tests/
├── conftest.py                 # Shared fixtures, synthetic data factory
├── test_data_models.py
├── test_ml_pipeline.py
├── test_refresh_logic.py
├── test_api.py
└── test_explainability.py

config/
├── thresholds.json
└── mlruns/                     # Local MLflow tracking (gitignored)
```

---

## 10. Open Questions & Decisions Required

> **[DECISION REQUIRED — Blocking]**

1. **Institute Data Sourcing:** `placement_rate_3mo/6mo/12mo` and `median_salary_lpa` for `InstituteContext` — will this be manually seeded per institute for the pilot, or is there an existing data export from the borrower app / LMS? This directly affects mock data realism and cold-start fallback design.

2. **Sector Classification:** `target_sector` for `MacroIndicators` — who classifies the student's target sector (self-reported? inferred from course type? LMS metadata)? This determines whether it's a feature input or a derived field.

3. **Consent Version Registry:** `consent_version` in `ConsentLayer` maps to specific consent text shown to users. Does Compliance own this registry, or is it embedded in code? Affects `POST /consent/revoke` purge logic.

4. **ORM for Production:** SQLAlchemy + PostgreSQL config is deferred in v1.0. Decision: proceed with SQLAlchemy models in `src/db/` now (even if unused in simulation) to avoid migration pain later, OR keep simulation storage truly ephemeral and add DB layer in Phase 2?

5. **Salary Label Source for Training:** During MVP with synthetic data, salary labels will be rule-derived. For production retraining, what is the ground truth source — AA-confirmed income data, placement cell reports, or student self-reported? This affects model validity claims.

---

## 11. Verification Plan

### Automated Tests
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
# Target: 100% pass rate; >85% line coverage on src/ml/ and src/models/
```

### ML-Specific Checks (run after each training cycle)
```bash
# Calibration check
python scripts/verify_calibration.py --model_version latest
# Acceptable: Brier score < 0.20; calibration curve within ±0.05 of diagonal

# Salary coverage check
python scripts/verify_quantile_coverage.py
# Acceptable: 65–75% of validation actuals fall within [lower_lpa, upper_lpa]

# SHAP consistency check
python scripts/verify_shap_consistency.py
# Acceptable: SHAP sum within 1e-4 of (prediction - base_value)

# Drift baseline
python scripts/set_drift_baseline.py
# Run once after initial training to set reference distribution for PSI
```

### Manual Verification
- Confirm all `SuccessScoreOutput.shap_drivers[].feature_label` values are human-readable (no snake_case)
- Confirm cold-start students have `confidence_level = "LOW"` and lender UI shows badge
- Confirm consent revocation purges `psych` and `behavioral` fields within one request cycle
- Confirm no raw SHAP scores or probabilities appear in the student-facing view

---

*Plan v2.0 — Last updated: April 2026. To be reviewed by ML Lead, Backend Lead, Compliance, and Risk teams before implementation begins.*