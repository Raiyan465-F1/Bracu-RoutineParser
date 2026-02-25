# Bracu RoutineParser

Bracu RoutineParser extracts class routine data from BRACU routine images.

It currently supports:
- Connect downloaded routines (screenshot)
- Connect screenshots
- Preprereg screenshots

The parser can return:
- Full routine entries (day, time, course, section, faculty, room)
- Minimal output (course code + section only) for autofill workflows

## Features

- EasyOCR-based OCR pipeline (CPU-friendly)
- Handles multiple screenshot styles
- Detects and crops exam schedule section in mixed images
- Course+section-only parser for quick website integration

## Project structure

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

## Usage

### Full parser

```bash
python examples/run_parser.py image/image.png
```

Returns structured routine JSON including classes, meta warnings, and parser details.

### Course + section only (recommended for autofill)

```bash
python examples/run_course_section_parser.py image/image.png
```

Returns:

```json
{
  "course_sections": [
    { "course_code": "CSE330", "section": "06" }
  ],
  "meta": {
    "source_image": "image/image.png",
    "template_id": "light_grid",
    "warnings": []
  }
}
```

## Notes and limitations

- OCR quality depends on screenshot clarity.
- Borderless/low-quality images may still need manual correction.
- Parser is tuned for BRACU routine formats and may not generalize to unrelated layouts.
