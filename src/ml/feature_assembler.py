import pandas as pd
import numpy as np
from typing import List
from src.models.schemas import StudentIngestionPayload

class FeatureAssembler:
    """Assembles structured StudentIngestionPayload into a flat feature DataFrame"""
    
    def __init__(self, institute_cohort_medians: dict = None):
        # Medians used for median imputing missing optional signals
        self.medians = institute_cohort_medians or {}
        
    def assemble(self, payloads: List[StudentIngestionPayload]) -> pd.DataFrame:
        features = []
        for p in payloads:
            f = {}
            
            # Academic Features
            f["cgpa_normalized"] = p.academic.cgpa / 10.0 if p.academic.cgpa_scale == "10" else p.academic.cgpa / 100.0
            semesters = p.academic.semester_cgpas
            if len(semesters) > 1:
                f["cgpa_trend"] = sum(semesters[i] - semesters[i-1] for i in range(1, len(semesters))) / (len(semesters) - 1)
                f["cgpa_consistency"] = np.std(semesters)
            else:
                f["cgpa_trend"] = 0.0
                f["cgpa_consistency"] = 0.0
            
            f["course_type_encoded"] = p.academic.course_type.value
            f["year_of_study"] = p.academic.year_of_study
            f["certifications_count"] = len(p.academic.certifications)
            
            # Internship Features
            f["internship_duration_weeks"] = sum(i.duration_weeks for i in p.internships)
            f["internship_count"] = len(p.internships)
            
            tier_weights = {"MNC": 1, "MidSize": 2, "Startup": 3, "None": 4}
            best_tier = min([tier_weights[i.employer_tier.value] for i in p.internships] + [4])
            f["employer_tier_best"] = best_tier
            
            if p.internships:
                f["role_relevance_avg"] = sum(i.role_relevance_score for i in p.internships) / len(p.internships)
                f["has_verified_internship"] = any(i.is_institute_verified for i in p.internships)
            else:
                f["role_relevance_avg"] = 0
                f["has_verified_internship"] = False
                
            # Institute Features
            f["institute_tier"] = p.institute.institute_tier
            f["placement_rate_3mo"] = p.institute.placement_rate_3mo
            f["placement_rate_6mo"] = p.institute.placement_rate_6mo
            f["placement_rate_12mo"] = p.institute.placement_rate_12mo
            f["placement_rate_yoy_delta"] = p.institute.placement_rate_yoy_delta
            f["median_salary_lpa"] = p.institute.median_salary_lpa
            f["recruiter_participation_normalized"] = p.institute.recruiter_participation_normalized
            
            # Macro Features
            f["sector_hiring_index"] = p.macro.sector_hiring_index
            f["region_job_density"] = p.macro.region_job_density
            f["macro_gdp_growth_pct"] = p.macro.macro_gdp_growth_pct
            f["sector_demand_trend"] = p.macro.sector_demand_trend_3mo
            
            # Psycho-Behavioral Features (Optional - NullableFeatureImputer logic inline)
            # Impute with cohort medians if provided, else safe defaults based on standard distributions
            f["grit_score"] = p.psych.grit_score if (p.psych and p.psych.grit_score) else self.medians.get("grit_score", 3.0)
            f["growth_mindset_score"] = p.psych.growth_mindset_score if (p.psych and p.psych.growth_mindset_score) else self.medians.get("growth_mindset_score", 3.5)
            f["self_efficacy_score"] = p.psych.self_efficacy_score if (p.psych and p.psych.self_efficacy_score) else self.medians.get("self_efficacy_score", 4.0)
            f["job_portal_activity_index"] = p.behavioral.job_portal_activity_index if (p.behavioral and p.behavioral.job_portal_activity_index) else self.medians.get("job_portal_activity_index", 0.5)
            f["resume_update_count"] = p.behavioral.resume_update_count_90d if (p.behavioral and p.behavioral.resume_update_count_90d is not None) else self.medians.get("resume_update_count", 0)
            
            features.append(f)
            
        return pd.DataFrame(features)
