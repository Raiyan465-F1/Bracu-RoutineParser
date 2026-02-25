from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from routine_parser.course_section_parser import CourseSectionParser
from routine_parser.pipeline import RoutineParser


app = FastAPI(
    title="Bracu RoutineParser API",
    version="1.0.0",
    description="Parse BRACU routine screenshots into structured data.",
)

_full_parser = RoutineParser()
_course_section_parser = CourseSectionParser()

_ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def _save_upload_to_temp(upload: UploadFile) -> Path:
    if not upload.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    suffix = Path(upload.filename).suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    content = upload.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return Path(tmp.name)


@app.get("/health")
def health() -> dict[str, bool]:
    return {"ok": True}


@app.post("/parse/full")
def parse_full(file: UploadFile = File(...)) -> dict[str, object]:
    temp_path = _save_upload_to_temp(file)
    try:
        result = _full_parser.parse_image(str(temp_path))
        return result.to_dict()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse routine: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/parse/course-sections")
def parse_course_sections(file: UploadFile = File(...)) -> dict[str, object]:
    temp_path = _save_upload_to_temp(file)
    try:
        return _course_section_parser.parse_image(str(temp_path))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse course sections: {exc}",
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)
