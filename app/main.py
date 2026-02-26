from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

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


def get_course_sections_by_image(image_path: str) -> dict[str, object]:
    """
    Parse course+section pairs from a saved image path.
    """
    return _course_section_parser.parse_image(image_path)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Bracu RoutineParser UI</title>
    <style>
      body { font-family: system-ui, sans-serif; margin: 2rem; max-width: 760px; }
      h1 { margin-bottom: 0.5rem; }
      p { color: #555; }
      .box { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-top: 1rem; }
      button { padding: 0.6rem 1rem; border-radius: 6px; border: 1px solid #ccc; cursor: pointer; }
      pre { background: #111; color: #f3f3f3; padding: 1rem; border-radius: 8px; overflow: auto; }
    </style>
  </head>
  <body>
    <h1>Bracu RoutineParser</h1>
    <p>Upload a routine image and extract course code + section.</p>
    <div class="box">
      <input id="fileInput" type="file" accept=".png,.jpg,.jpeg,.webp" />
      <button id="runBtn">Parse Course Sections</button>
    </div>
    <h3>Response</h3>
    <pre id="output">{ "course_sections": [] }</pre>
    <script>
      const runBtn = document.getElementById("runBtn");
      const fileInput = document.getElementById("fileInput");
      const output = document.getElementById("output");

      runBtn.addEventListener("click", async () => {
        const file = fileInput.files[0];
        if (!file) {
          output.textContent = JSON.stringify({ error: "Please choose an image file first." }, null, 2);
          return;
        }
        const form = new FormData();
        form.append("file", file);
        output.textContent = "Parsing...";
        try {
          const res = await fetch("/parse/course-sections/by-image", { method: "POST", body: form });
          const data = await res.json();
          output.textContent = JSON.stringify(data, null, 2);
        } catch (e) {
          output.textContent = JSON.stringify({ error: String(e) }, null, 2);
        }
      });
    </script>
  </body>
</html>"""


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
        return get_course_sections_by_image(str(temp_path))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse course sections: {exc}",
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/parse/course-sections/by-image")
def parse_course_sections_by_image(file: UploadFile = File(...)) -> dict[str, object]:
    temp_path = _save_upload_to_temp(file)
    try:
        return get_course_sections_by_image(str(temp_path))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse course sections: {exc}",
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)
