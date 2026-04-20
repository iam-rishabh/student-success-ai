# specifications.md

**Project Title:** Student-Success Guidance Layer for Education Loan Risk Intelligence System  
**Version:** 1.1 (Updated to align with Problem Statement requirements)  
**Date:** April 2026  
**Audience:** Poonawalla Fincorp (NBFC) – Risk, Credit, Product, and Compliance Teams  
**Purpose:** This document outlines the functional, technical, compliance, and operational requirements for a **supportive, non-automated** AI-powered student-success guidance layer.  

The layer enriches **human lending decisions**, supports **portfolio monitoring**, and enables **voluntary opt-in student-support programs** for study-abroad education loans. It does **not** automate credit approvals, rejections, pricing, or any lending decisions.

## 1. Project Objectives
- Predict probability of job placement within 3, 6, and 12 months after graduation.
- Estimate expected starting salary range.
- Identify placement-risk that may impact repayment capacity.
- Provide lenders with early visibility into employability risks to support proactive interventions.
- Combine institute-level trends, student-level skills (including internship performance), industry demand signals, and optional real-time behavior data.
- Deliver simple, explainable outputs usable across different institutes, regions, and course types.

## 2. Scope
**In Scope:**
- Robust data collection pipeline including academic history, internship details, institute placement data, industry/labor-market indicators, and optional real-time student behavior signals.
- Closed-loop AI pipeline with ensemble modeling and Success Score for placement timeline and salary predictions.
- Lightweight psychological-behavioral layer (grit, growth mindset, self-efficacy) for nudge personalization.
- Explainable outputs using SHAP + LIME.
- Voluntary, opt-in student guidance with suggested next-best actions.
- Integration with existing borrower app and LMS.

**Out of Scope:**
- Automated credit decisioning or approval workflows.
- Mandatory student participation.
- Storage of sensitive psychological raw data beyond consented, minimized processing.

## 3. Functional Requirements

### 3.1 Data Collection Pipeline
- **Academic & Program Information:** Course type, year/semester, CGPA/percentage, academic consistency, skill certifications, relevant coursework.
- **Internship History:** Duration, employer type, performance notes (student-uploaded or institute-verified, consent-based).
- **Institute & Program-Level Data:** Institute tier, historic placement rates (3/6/12-month), salary benchmarks, placement-cell activity levels, recruiter participation trends (annual batch + selective feeds).
- **Industry & Labor-Market Indicators:** Sector-specific hiring trends (IT, BFSI, healthcare, etc.), region-wise job density, macroeconomic conditions affecting hiring cycles (public data feeds + feature store updates).
- **Real-time Student Behavior Signals (Optional):** Job-portal activity summaries, interview pipeline progress, resume updates, skill-up events (consented app-based proxies or self-reporting).
- **Financial & Repayment Data:** Account Aggregator (AA) + CIBIL APIs (quarterly + event-triggered).
- **Psychological-Behavioral Signals (Optional):** Short validated scales — Grit-S (8 items), Growth Mindset (3–6 items), Academic Self-Efficacy (6 items). Administered semesterly or on-demand (<5 min).
- All collection requires explicit, granular, revocable consent under DPDP Act 2023. Real-time and psych signals are fully optional.

### 3.2 Closed-Loop Pipeline
- Data ingestion → Feature store → AI processing → Human-reviewed outputs → Voluntary nudges/support actions → Actual outcomes (placement status, salary, repayment behavior) → Labeled feedback → Model retraining.
- Drift detection and quarterly retraining on closed-loop data.

### 3.3 Risk Modeling Mathematics & Algorithms
- **Core Models:** Ensemble (XGBoost primary, Random Forest for variance reduction, Logistic Regression as interpretable baseline).
- **Success Score Algorithm:**
  - Outputs: Probability of placement within 3/6/12 months, expected salary range, placement-risk tier (low/medium/high).
  - Features include: academic trajectory, internship performance, institute placement strength, sector hiring trends, region-wise job density, macro signals, and optional psycho-behavioral scores (as mediators only when consented).
- **Explainability:** SHAP (global & local) + LIME. Plain-English summaries (e.g., “low internship exposure + weak sector demand”).
- Psych scores and real-time signals improve nudge routing and engagement prediction — never used as standalone credit signals.

### 3.4 Outputs for Lenders & Students
- **Pre-sanction / Monitoring:** Employability Intelligence Brief with placement probabilities, salary bands, risk signals, and explainable drivers.
- **Early Alerts:** Exceptions-only notifications for high-risk students (e.g., delayed placement risk).
- **Suggested Next-Best Actions** (advisory only): Skill-up recommendations, resume improvement, mock interview coaching, or high-potential recruiter match suggestions (where data permits).
- All outputs are human-reviewed. No automated lending actions.

### 3.5 Student Support (Voluntary)
- In-app/chatbot personalized nudges and learning paths.
- Milestone tracking that feeds positive signals back to the risk model.
- Incentives (e.g., small interest rebates on verifiable milestones) optional to drive engagement.

## 4. Non-Functional Requirements
- **Scalability:** Support multiple institutes, courses, regions, and job markets.
- **Performance:** Target placement prediction accuracy 88–94%, salary estimates ±15–20% error margin (initial benchmarks).
- **Explainability & Auditability:** Full logging of SHAP values and consent records.
- **Security & Privacy:** Data minimization, zero-storage where possible (AA model), DPDP-compliant consent engine.

## 5. Compliance & Regulatory Requirements
- DPDP Act 2023: Explicit consent for all personal, psych, and behavioral data.
- RBI Guidelines: Adhere to Model Education Loan Scheme flexibility; fair practices code; explainable model outputs.
- Psychological and real-time behavioral data treated as sensitive and ancillary.

## 6. Implementation & Operational Requirements
- **Timeline (MVP):** 3–4.5 months (partnership route for guidance layer).
- **Cost (Partnership/Outsourcing Guidance Layer):** One-time ₹45 lakh – 1 Cr; ongoing ₹12–25 lakh/year per 1,000 students.
- **Human Overhead:** <1.5 FTE (oversight, exception review, partner management).
- **Tech Stack Preference:** White-label EdTech partner for assessments, nudges, and real-time signal handling; core risk modeling integrated with existing LMS.

## 7. Success Metrics (for Pilot & Rollout)
- Accuracy of 3/6/12-month placement predictions and salary estimates.
- Usefulness via early alerts and lender feedback.
- Student opt-in and engagement rates (target ≥35% for flagged students).
- Model robustness across varied programs and labor-market conditions.
- Impact on early delinquencies and student support outcomes (measured post-pilot).

## 8. Risks & Mitigations
- Low consent for optional signals → Mitigate with clear value communication and simple UX.
- Data sparsity for niche programs → Conservative estimates with low-confidence flags.
- Regulatory changes → Modular consent engine and regular reviews.

## 9. Dependencies
- Existing LMS/borrower app.
- Access to DigiLocker/NAD (API Setu), Account Aggregator, CIBIL APIs.
- White-label EdTech partner capable of handling internship data, real-time signals, and suggested actions.

## 10. Approval & Next Steps
- Review by Risk, Compliance, Legal, and Product teams.
- Pilot on 300–500 loans across varied institutes and courses.
- Final sign-off before full integration.

---

**Note:** This specification fully aligns with the Problem Statement by explicitly including internship history, institute operational signals, industry/labor-market indicators, real-time student behavior signals (optional), and suggested next-best actions as advisory outputs. The system remains strictly supportive, consent-driven, and human-supervised. All psychological and real-time signals are optional and used only for enrichment and personalization.