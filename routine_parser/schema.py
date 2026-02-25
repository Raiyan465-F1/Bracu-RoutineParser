from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class StudentInfo:
    name: str | None = None
    student_id: str | None = None


@dataclass
class ClassEntry:
    course_code: str
    section: str | None
    faculty: str | None
    room: str | None
    day: str
    start_time: str
    end_time: str
    class_type: str = "theory"
    confidence: float = 0.0
    raw_text: str = ""


@dataclass
class ExamEntry:
    course_code: str
    exam_type: str
    date: str
    start_time: str
    end_time: str
    confidence: float = 0.0
    raw_text: str = ""


@dataclass
class ParseMeta:
    source_image: str
    parser_version: str = "0.1.0"
    template_id: str = "unknown"
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class RoutineParseResult:
    student: StudentInfo = field(default_factory=StudentInfo)
    classes: list[ClassEntry] = field(default_factory=list)
    exams: list[ExamEntry] = field(default_factory=list)
    meta: ParseMeta = field(default_factory=lambda: ParseMeta(source_image=""))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
