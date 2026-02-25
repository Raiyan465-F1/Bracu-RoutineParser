from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from routine_parser import RoutineParser


def main() -> None:
    parser = argparse.ArgumentParser(description="Draft routine parser")
    parser.add_argument("image_path", help="Path to routine screenshot")
    parser.add_argument(
        "--ocr-preview-text",
        default="",
        help="Optional OCR preview text for routing hints",
    )
    args = parser.parse_args()

    engine = RoutineParser()
    result = engine.parse_image(
        image_path=args.image_path,
        ocr_preview_text=args.ocr_preview_text,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
