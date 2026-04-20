import uuid
import random
from datetime import datetime
from src.models.schemas import (
    StudentIngestionPayload, AcademicProfile, InternshipRecord, 
    InstituteContext, MacroIndicators, ConsentLayer, PsychoSignals,
    CourseTypeEnum, EmployerTierEnum, SectorEnum
)

def generate_mock_payload(student_id=None) -> StudentIngestionPayload:
    if student_id is None:
        # Pydantic schema expects STU- followed by 8 alphanumeric chars
        random_suffix = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=8))
        student_id = f"STU-{random_suffix}"

    academic = AcademicProfile(
        course_type=random.choice(list(CourseTypeEnum)),
        year_of_study=random.randint(1, 4),
        cgpa=round(random.uniform(5.5, 9.8), 2),
        cgpa_scale="10",
        semester_cgpas=[round(random.uniform(5.0, 10.0), 2) for _ in range(random.randint(1, 8))],
        certifications=[f"Cert-{i}" for i in range(random.randint(0, 3))],
        relevant_coursework=[f"Course-{i}" for i in range(random.randint(2, 5))]
    )

    internships = []
    for _ in range(random.randint(0, 2)):
        internships.append(InternshipRecord(
            duration_weeks=random.randint(4, 24),
            employer_tier=random.choice(list(EmployerTierEnum)),
            role_relevance_score=random.randint(1, 5),
            is_institute_verified=random.choice([True, False])
        ))

    institute = InstituteContext(
        institute_tier=random.choice([1, 2, 3]),
        placement_rate_3mo=round(random.uniform(0.3, 0.9), 2),
        placement_rate_6mo=round(random.uniform(0.4, 0.95), 2),
        placement_rate_12mo=round(random.uniform(0.5, 0.99), 2),
        placement_rate_yoy_delta=round(random.uniform(-0.1, 0.15), 2),
        median_salary_lpa=round(random.uniform(3.0, 15.0), 2),
        recruiter_participation_normalized=round(random.uniform(0.1, 1.0), 2),
        data_vintage_months=random.randint(1, 24)
    )

    macro = MacroIndicators(
        target_sector=random.choice(list(SectorEnum)),
        sector_hiring_index=round(random.uniform(0.2, 0.9), 2),
        sector_demand_trend_3mo=round(random.uniform(-0.2, 0.2), 2),
        region_job_density=round(random.uniform(10.0, 500.0), 2),
        macro_gdp_growth_pct=round(random.uniform(-2.0, 8.0), 2),
        data_as_of=datetime.utcnow().date()
    )

    consent = ConsentLayer(
        academic_data_consent=True,
        internship_data_consent=True,
        psycho_signals_consent=True,
        behavioral_signals_consent=True,
        lender_sharing_consent=True,
        consented_at=datetime.utcnow(),
        consent_version="v1.0"
    )

    psych = PsychoSignals(
        grit_score=round(random.uniform(1.0, 5.0), 2),
        growth_mindset_score=round(random.uniform(1.0, 6.0), 2),
        self_efficacy_score=round(random.uniform(1.0, 7.0), 2),
        administered_at=datetime.utcnow()
    )

    return StudentIngestionPayload(
        student_id=student_id,
        academic=academic,
        internships=internships,
        institute=institute,
        macro=macro,
        consent=consent,
        psych=psych
    )
