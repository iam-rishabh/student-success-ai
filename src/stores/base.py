from abc import ABC, abstractmethod
from typing import List, Optional
from src.models.schemas import StudentIngestionPayload

class FeatureStore(ABC):
    @abstractmethod
    def save_student(self, payload: StudentIngestionPayload):
        pass

    @abstractmethod
    def get_student(self, student_id: str) -> Optional[StudentIngestionPayload]:
        pass

    @abstractmethod
    def get_all_students(self) -> List[StudentIngestionPayload]:
        pass

    @abstractmethod
    def revoke_student_signals(self, student_id: str):
        pass
