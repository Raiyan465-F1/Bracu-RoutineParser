from __future__ import annotations

import re
from dataclasses import dataclass


COURSE_PATTERN = re.compile(r"([A-Z]{3}\d{3}[A-Z]?)")
COURSE_OCR_PATTERN = re.compile(r"([A-Z]{3}[0-9OIJLSBZ]{3}[A-Z]?)")
SECTION_PATTERN = re.compile(r"(?:^|[-\s])(\d{1,2})(?:$|[-\s])")


@dataclass
class ParsedCell:
    course_code: str | None = None
    section: str | None = None
    faculty: str | None = None
    room: str | None = None


def normalize_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("\n", " ").replace("—", "-").replace("–", "-")
    return " ".join(cleaned.split())


def _normalize_ocr_course(candidate: str) -> str:
    """
    Convert OCR-confused course codes like ECO1O1 -> ECO101.
    """
    if len(candidate) < 6:
        return candidate
    head = candidate[:3]
    digit_map = {
        "O": "0",
        "I": "1",
        "J": "3",
        "L": "1",
        "S": "5",
        "B": "8",
        "Z": "2",
    }
    tail = "".join(digit_map.get(ch, ch) for ch in candidate[3:6])
    suffix = candidate[6:]
    return f"{head}{tail}{suffix}"


def parse_class_payload(payload: str) -> ParsedCell:
    """
    Parse values from strings like:
      CSE340-24 - AFQ-09H-35C
      CSE423L-10 - TBA-09F-24L
    """
    text = normalize_text(payload)
    result = ParsedCell()

    if not text:
        return result

    course_match = COURSE_PATTERN.search(text)
    if not course_match:
        ocr_course_match = COURSE_OCR_PATTERN.search(text)
        if ocr_course_match:
            normalized = _normalize_ocr_course(ocr_course_match.group(1))
            if COURSE_PATTERN.fullmatch(normalized):
                result.course_code = normalized
                remainder = text[ocr_course_match.end() :].strip(" -")
            else:
                remainder = text
        else:
            remainder = text
    if course_match:
        result.course_code = course_match.group(1)
        remainder = text[course_match.end() :].strip(" -")

    section_match = SECTION_PATTERN.search(text)
    if section_match:
        result.section = section_match.group(1).zfill(2)
        # Prefer section located right after course code.
        section_head = re.match(r"^(\d{1,2})\b", remainder)
        if section_head:
            remainder = remainder[section_head.end() :].strip(" -")
            result.section = section_head.group(1).zfill(2)

    # Heuristic split: expected remainder resembles AFQ-09H-35C or TBA-09F-27L.
    right = remainder

    right_parts = [p.strip() for p in right.split("-") if p.strip()]
    if right_parts:
        faculty_candidate = right_parts[0].split()[0]
        if re.fullmatch(r"[A-Z]{2,5}", faculty_candidate):
            result.faculty = faculty_candidate
            if len(right_parts) > 1:
                result.room = "-".join(right_parts[1:])
        else:
            # fallback to legacy behavior for unusual OCR strings
            result.faculty = right_parts[0]
            if len(right_parts) > 1:
                result.room = "-".join(right_parts[1:])

    return result
