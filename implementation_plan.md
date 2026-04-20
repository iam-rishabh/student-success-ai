# Testing, Validation, and Minimal Data Models Implementation Plan

This plan addresses the requirement to provide production-grade, minimally redundant data models leveraging **FastAPI and Pydantic**, along with robust testing for each module based on `spqcifications.md`. It also outlines architectural, frontend, and backend decisions, including the data refresh timelines for testing vs. production.

## User Review Required

> [!IMPORTANT]  
> Please review the differences identified in the **Simulation vs. Deployable Architecture** sections, and the decisions outlined for the Frontend and Backend technologies.

## Architectural Differentiations & Data Refresh Strategy

We distinguish between two distinct architectural operational modes to ensure smooth prototyping without affecting the production roadmap:

### 1. Simulation / Testing Architecture (MVP/Local) 
* **Data Refresh Timeline**: Every **2 minutes**.
* **Job Scheduler**: `APScheduler` or FastAPI background tasks for immediate simulation feedback.
* **Storage**: In-memory SQLite or mocked JSON stores for high-speed simulation runs.
* **ML Inference**: Local `scikit-learn`/`XGBoost` pipeline using in-memory mock datasets and dummy pre-trained weights for instantaneous predictions.
* **Purpose**: Allows rapid inspection of the system's ability to trigger "Exceptions-only" alerts, route risk students, and adapt to newly refreshed 'synthetic' profile data in real-time.

### 2. Real Deployable Architecture (Production)
* **Data Refresh / Model Retraining Timeline**: Every **45 days** (in alignment with quarterly reviews and drift detection).
* **Job Scheduler**: Distributed task queues like `Celery` with a `Redis` or `RabbitMQ` message broker handling massive concurrent batch jobs.
* **Storage**: High-performance RDBMS (e.g., PostgreSQL for relational data) combined with a highly scalable Feature Store (e.g., Redis or Feast) for ML inputs.
* **ML Inference**: MLflow / S3 for model registry. The FastAPI backend fetches versioned `XGBoost`/`Ensemble` models. Explainability computes via externalized workers to offload API performance footprint.
* **Purpose**: High availability, strictly controlled DPDP consent mechanisms, asynchronous batch processing, and non-blocking real-time inferences.

## Frontend & Backend Component Decisions

### Backend Decisions:
1. **Core API Server**: **FastAPI** -> Guaranteed high performance, asynchronous request handling, and automatic OpenAPI schema generation for front-end integration.
2. **Data & Schema Validation**: **Pydantic V2** -> For blazing-fast, non-redundant, minimal data schema typing and validation. Invalid payloads (e.g. invalid CGPA, malformed industry signals) are rejected instantaneously with precise error feedback.
3. **ML Interoperability**: `scikit-learn` and `xgboost` wrapped inside dependency-injected Python services. Explainability handled with `shap` and `lime`.
4. **Testing framework**: `pytest` -> Utilizing parameterized fixtures, mock dependencies using `unittest.mock` to ensure production-grade evaluation coverage of edge cases, data types, and model latency bounds.

### Frontend Decisions (Lender & Student Views):
1. **Framework**: **Next.js (React)** -> Excellent for routing, SSR (Server Side Rendering), and unifying the lender dashboard alongside the student-facing view.
2. **Styling & Components**: **TailwindCSS** coupled with **Shadcn UI** for robust, modern, clean, and highly accessible user interface components without bloating the bundle.
3. **Data Visualization (For Lenders)**: **Recharts** or **Chart.js** -> Necessary to visualize the output of SHAP/LIME logic clearly ("Explainability"), plotting placement probability vs salary bands effortlessly.
4. **State Management**: **Zustand** or **React Query** -> Seamless handling of async API interactions with the FastAPI backend, easily managing refresh timings.

---

## Proposed Code Structure

### 1. Minimal Data Models & Validations (`src/models/`)

#### [NEW] [schemas.py](file:///home/rishab-das/fedora/backup1/Programs/hackthon/TensorX/src/models/schemas.py)
Production-grade Pydantic schemas validating specific constraints.
* `AcademicProfile` (cgpa strict limits), `InternshipRecord`
* `InstituteContext`, `MacroIndicators`
* `ConsentLayer` (Mandatory booleans ensuring DPDP 2023 compliance)
* `StudentIngestionPayload`

### 2. Controllers & Simulation Schedulers (`src/api/` & `src/jobs/`)

#### [NEW] [main.py](file:///home/rishab-das/fedora/backup1/Programs/hackthon/TensorX/src/api/main.py)
* FastAPI root initializing real-time validation endpoints.

#### [NEW] [scheduler.py](file:///home/rishab-das/fedora/backup1/Programs/hackthon/TensorX/src/jobs/scheduler.py)
* Implementation of the **Data Refresh** mechanics. (Contains logic capable of switching between the 2-minute testing mode vs 45-day production mode).

### 3. ML Pipeline Models (`src/ml/`)

#### [NEW] [predictor.py](file:///home/rishab-das/fedora/backup1/Programs/hackthon/TensorX/src/ml/predictor.py)
* Production-grade interface housing model instantiation.
* Wrappers for `xgboost` predictions and simulated SHAP outputs mapping to the API models.

### 4. Testing Suite (`tests/`)

#### [NEW] [test_data_models.py](file:///home/rishab-das/fedora/backup1/Programs/hackthon/TensorX/tests/test_data_models.py)
* Extensive `pytest` modules testing bounds, missing values, and strictly typing the Pydantic schemas.

#### [NEW] [test_ml_pipeline.py](file:///home/rishab-das/fedora/backup1/Programs/hackthon/TensorX/tests/test_ml_pipeline.py)
* Pytest suites explicitly evaluating latency, inference structure mappings, and edge cases.

#### [NEW] [test_refresh_logic.py](file:///home/rishab-das/fedora/backup1/Programs/hackthon/TensorX/tests/test_refresh_logic.py)
* Unit testing to verify that the 2-minute/45-day intervals trigger data invalidation/recalculation correctly.

---

## Open Questions

> [!WARNING]  
> Are there any specific database/ORM preferences (like SQLAlchemy / PostgreSQL) that you'd like me to start laying out config for, or should I leave persistence mocked for the simulation layer at this stage?

## Verification Plan

### Automated Tests
- Full `pytest` execution mimicking production validation boundaries. Ensuring 100% pass-rate on payload validations and schema integrity.

### Manual Verification
- Analyze the `Pydantic` schema constraints and confirm they align completely with high-accuracy, data minimization regulations according to specifications.
