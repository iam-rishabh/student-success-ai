import sqlite3
import json
from typing import List, Optional
from datetime import datetime
from src.stores.base import FeatureStore
from src.models.schemas import StudentIngestionPayload

class SQLiteFeatureStore(FeatureStore):
    def __init__(self, db_path=":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self):
        c = self.conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                payload_json TEXT
            )
        ''')
        self.conn.commit()

    def save_student(self, payload: StudentIngestionPayload):
        c = self.conn.cursor()
        c.execute('REPLACE INTO students (student_id, payload_json) VALUES (?, ?)', 
                  (payload.student_id, payload.model_dump_json()))
        self.conn.commit()

    def get_student(self, student_id: str) -> Optional[StudentIngestionPayload]:
        c = self.conn.cursor()
        c.execute('SELECT payload_json FROM students WHERE student_id = ?', (student_id,))
        row = c.fetchone()
        if row:
            return StudentIngestionPayload.model_validate_json(row[0])
        return None

    def get_all_students(self) -> List[StudentIngestionPayload]:
        c = self.conn.cursor()
        c.execute('SELECT payload_json FROM students')
        rows = c.fetchall()
        return [StudentIngestionPayload.model_validate_json(row[0]) for row in rows]

    def revoke_student_signals(self, student_id: str):
        payload = self.get_student(student_id)
        if payload:
            payload.psych = None
            payload.behavioral = None
            payload.consent.psycho_signals_consent = False
            payload.consent.behavioral_signals_consent = False
            payload.consent.revoked_at = datetime.utcnow()
            self.save_student(payload)
