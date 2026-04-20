from pydantic import BaseModel, Field, model_validator
from datetime import datetime, date
from typing import Optional, Literal
from enum import Enum

class CourseTypeEnum(str, Enum):
    BTECH = "BTech"
    MBA = "MBA"
    MCA = "MCA"
    BBA = "BBA"
    BCA = "BCA"
    OTHER = "Other"

class EmployerTierEnum(str, Enum):
    MNC = "MNC"
    MID_SIZE = "MidSize"
    STARTUP = "Startup"
    NONE = "None"

class SectorEnum(str, Enum):
    IT = "IT"
    BFSI = "BFSI"
    HEALTHCARE = "Healthcare"
    MANUFACTURING = "Manufacturing"
    RETAIL = "Retail"
    OTHER = "Other"

class InterviewStageEnum(str, Enum):
    APPLIED = "Applied"
    SCREENING = "Screening"
    TECHNICAL_ROUND = "TechnicalRound"
    HR_ROUND = "HRRound"
    OFFERED = "Offered"
    REJECTED = "Rejected"

class AcademicProfile(BaseModel):
    course_type: CourseTypeEnum
    year_of_study: int = Field(ge=1, le=6)
    cgpa: float = Field(ge=0.0, le=10.0)
    cgpa_scale: Literal["10", "100"]
    semester_cgpas: list[float] = Field(min_length=1, max_length=12)
    certifications: list[str] = Field(default_factory=list, max_length=20)
    relevant_coursework: list[str] = Field(default_factory=list)

class InternshipRecord(BaseModel):
    duration_weeks: int = Field(ge=0, le=104)
    employer_tier: EmployerTierEnum
    role_relevance_score: int = Field(ge=1, le=5)
    is_institute_verified: bool

class InstituteContext(BaseModel):
    institute_tier: Literal[1, 2, 3]
    placement_rate_3mo: float = Field(ge=0.0, le=1.0)
    placement_rate_6mo: float = Field(ge=0.0, le=1.0)
    placement_rate_12mo: float = Field(ge=0.0, le=1.0)
    placement_rate_yoy_delta: float
    median_salary_lpa: float = Field(ge=0.0)
    recruiter_participation_normalized: float = Field(ge=0.0, le=1.0)
    data_vintage_months: int

class MacroIndicators(BaseModel):
    target_sector: SectorEnum
    sector_hiring_index: float = Field(ge=0.0, le=1.0)
    sector_demand_trend_3mo: float
    region_job_density: float = Field(ge=0.0)
    macro_gdp_growth_pct: float
    data_as_of: date

class PsychoSignals(BaseModel):
    grit_score: Optional[float] = Field(default=None, ge=1.0, le=5.0)
    growth_mindset_score: Optional[float] = Field(default=None, ge=1.0, le=6.0)
    self_efficacy_score: Optional[float] = Field(default=None, ge=1.0, le=7.0)
    administered_at: Optional[datetime] = None

class BehavioralSignals(BaseModel):
    job_portal_activity_index: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    resume_update_count_90d: Optional[int] = Field(default=None, ge=0)
    interview_pipeline_stage: Optional[InterviewStageEnum] = None

class ConsentLayer(BaseModel):
    academic_data_consent: bool
    internship_data_consent: bool
    psycho_signals_consent: bool = False
    behavioral_signals_consent: bool = False
    lender_sharing_consent: bool
    consented_at: datetime
    consent_version: str
    revoked_at: Optional[datetime] = None

    @model_validator(mode='after')
    def validate_mandatory_consents(self):
        if not self.academic_data_consent or not self.lender_sharing_consent:
            raise ValueError("Core consents are mandatory for system participation")
        if self.revoked_at and self.revoked_at < self.consented_at:
            raise ValueError("Revoked time cannot be before consented time")
        return self

class StudentIngestionPayload(BaseModel):
    student_id: str = Field(pattern=r'^STU-[A-Z0-9]{8}$')
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

class PlacementHorizonScore(BaseModel):
    horizon_months: Literal[3, 6, 12]
    probability: float = Field(ge=0.0, le=1.0)
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]

class SalaryEstimate(BaseModel):
    median_lpa: float = Field(ge=0.0)
    lower_lpa: float = Field(ge=0.0)
    upper_lpa: float = Field(ge=0.0)

    @model_validator(mode='after')
    def validate_bounds(self):
        if self.lower_lpa > self.median_lpa or self.median_lpa > self.upper_lpa:
            raise ValueError("Invalid salary bounds: lower_lpa <= median_lpa <= upper_lpa required")
        return self

class ShapDriver(BaseModel):
    feature_label: str
    direction: Literal["positive", "negative"]
    magnitude: float

class SuccessScoreOutput(BaseModel):
    student_id: str
    placement_scores: list[PlacementHorizonScore]
    salary_estimate: SalaryEstimate
    shap_drivers: list[ShapDriver]
    confidence_level: Literal["LOW", "MEDIUM", "HIGH"]
    is_exception_alert: bool
    next_best_actions: list[str]
    model_version: str
    predicted_at: datetime
