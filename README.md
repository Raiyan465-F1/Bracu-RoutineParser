# Bracu RoutineParser

Bracu RoutineParser extracts class routine data from BRACU routine images.

It currently supports:
- Connect downloaded routines (screenshot)
- Connect screenshots
- Preprereg screenshots

The parser can return:
- Full routine entries (day, time, course, section, faculty, room)
- Minimal output (course code + section only) for autofill workflows

This project is exposed as a FastAPI service.

## Features

- EasyOCR-based OCR pipeline (CPU-friendly)
- Handles multiple screenshot styles
- Detects and crops exam schedule section in mixed images
- Course+section-only parser for quick website integration

## Project structure

- `app/main.py` - FastAPI entrypoint
- `routine_parser/` - parser core logic
- `examples/run_parser.py` - full parser CLI
- `examples/run_course_section_parser.py` - minimal parser CLI


## Quick setup

Python version:
- `3.11.11` (from `.python-version`)

```bash
python --version
```

Install dependencies:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install -r requirements.txt
```

## API usage

Run server:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Endpoints:

- `GET /health`
- `POST /parse/full` (form-data key: `file`)
- `POST /parse/course-sections` (form-data key: `file`)

Example request:

```bash
curl -X POST "http://localhost:8000/parse/course-sections" -F "file=@image/image.png"
```

## Railway deploy command

Set the start command to:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## Notes and limitations

- OCR quality depends on screenshot clarity.
- Borderless/low-quality images may still need manual correction.
- Parser is tuned for BRACU routine formats and may not generalize to unrelated layouts.
