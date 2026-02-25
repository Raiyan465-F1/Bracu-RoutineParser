from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from routine_parser.ocr import OcrEngine, OcrToken
from routine_parser.parsers import normalize_text, parse_class_payload
from routine_parser.table_grid import GridCell, detect_grid_cells
from routine_parser.extractors.base import RoutineExtractor
from routine_parser.schema import ClassEntry, ParseMeta, RoutineParseResult

DAY_ALIASES = {
    "sun": "Sunday",
    "sunday": "Sunday",
    "mon": "Monday",
    "monday": "Monday",
    "tue": "Tuesday",
    "tuesday": "Tuesday",
    "wed": "Wednesday",
    "wednesday": "Wednesday",
    "thu": "Thursday",
    "thursday": "Thursday",
    "fri": "Friday",
    "friday": "Friday",
    "sat": "Saturday",
    "saturday": "Saturday",
}
DAY_NAMES = sorted(set(DAY_ALIASES.values()))

TIME_TOKEN_PATTERN = re.compile(r"(\d{1,2}[:.]\d{2})\s*([AaPp][Mm])?")


def _to_24h(time_value: str, meridian: str) -> str:
    normalized = time_value.replace(".", ":")
    dt = datetime.strptime(f"{normalized} {meridian.upper()}", "%I:%M %p")
    return dt.strftime("%H:%M")


def _hour_value(time_value: str) -> int:
    return int(re.split(r"[:.]", time_value)[0])


def _extract_time_range(value: str) -> tuple[str | None, str | None]:
    text = normalize_text(value).replace("-", " - ")
    text = text.replace(";", ":")
    # OCR often drops ":" in 200/430 style times.
    text = re.sub(r"\b(\d{1,2})(\d{2})\s*([AaPp][Mm])", r"\1:\2 \3", text)
    text = re.sub(r"\b(\d{1,2})(\d{2})\b", r"\1:\2", text)
    matches = TIME_TOKEN_PATTERN.findall(text)
    if len(matches) < 2:
        return None, None

    start_raw, start_meridian = matches[0][0], matches[0][1]
    end_raw, end_meridian = matches[1][0], matches[1][1]

    if not start_meridian and not end_meridian:
        return None, None
    if not start_meridian and end_meridian:
        start_meridian = end_meridian
    if start_meridian and not end_meridian:
        end_hour = _hour_value(end_raw)
        start_hour = _hour_value(start_raw)
        # Common timetable pattern: 11:00 AM - 12:20 PM (OCR often drops PM).
        if start_meridian.upper() == "AM" and end_hour == 12 and start_hour <= 11:
            end_meridian = "PM"
        else:
            end_meridian = start_meridian

    try:
        start = _to_24h(start_raw, start_meridian)
        end = _to_24h(end_raw, end_meridian)
    except ValueError:
        return None, None

    return start, end


def _normalize_day(value: str) -> str | None:
    cleaned = normalize_text(value).lower()
    for token in cleaned.split():
        bare = re.sub(r"[^a-z]", "", token)
        if bare in DAY_ALIASES:
            return DAY_ALIASES[bare]
    # Fuzzy fallback for OCR noise like "wedresday".
    words = [re.sub(r"[^a-z]", "", word) for word in cleaned.split()]
    words = [word for word in words if len(word) >= 3]
    for word in words:
        best_match = None
        best_score = 0.0
        for alias in DAY_ALIASES:
            score = SequenceMatcher(None, word, alias).ratio()
            if score > best_score:
                best_score = score
                best_match = alias
        if best_match and best_score >= 0.78:
            return DAY_ALIASES[best_match]
    return None


def _has_time_signal(value: str) -> bool:
    start, end = _extract_time_range(value)
    if start and end:
        return True
    text = normalize_text(value).lower()
    return ":" in text and ("am" in text or "pm" in text)


def _pick_header_row(text_by_cell: dict[tuple[int, int], tuple[str, float, GridCell]]) -> int | None:
    row_scores: Counter[int] = Counter()
    for (row_idx, _col_idx), (text, _confidence, _cell) in text_by_cell.items():
        if _normalize_day(text):
            row_scores[row_idx] += 1
    if row_scores:
        return row_scores.most_common(1)[0][0]
    if not text_by_cell:
        return None
    return min(row for row, _ in text_by_cell.keys())


def _pick_time_col(text_by_cell: dict[tuple[int, int], tuple[str, float, GridCell]]) -> int | None:
    col_scores: Counter[int] = Counter()
    for (_row_idx, col_idx), (text, _confidence, _cell) in text_by_cell.items():
        if _has_time_signal(text):
            col_scores[col_idx] += 1
    if col_scores:
        return col_scores.most_common(1)[0][0]
    if not text_by_cell:
        return None
    return min(col for _, col in text_by_cell.keys())


def _write_debug_payload(payload: dict[str, object]) -> None:
    if os.getenv("ROUTINE_DEBUG", "").lower() not in {"1", "true", "yes"}:
        return
    target = os.getenv("ROUTINE_DEBUG_PATH", "debug/routine_debug.json")
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _summarize_cell_texts(
    text_by_cell: dict[tuple[int, int], tuple[str, float, GridCell]]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (row_idx, col_idx), (text, confidence, cell) in sorted(text_by_cell.items()):
        rows.append(
            {
                "row": row_idx,
                "col": col_idx,
                "text": text,
                "confidence": round(confidence, 4),
                "bbox": [cell.x, cell.y, cell.w, cell.h],
            }
        )
    return rows


def _cluster_token_lines(tokens: list[OcrToken], tolerance: int = 12) -> list[list[OcrToken]]:
    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda token: (token.bbox[1], token.bbox[0]))
    lines: list[list[OcrToken]] = []
    for token in ordered:
        y_center = (token.bbox[1] + token.bbox[3]) // 2
        if not lines:
            lines.append([token])
            continue
        last_line = lines[-1]
        last_y = sum((t.bbox[1] + t.bbox[3]) // 2 for t in last_line) // len(last_line)
        if abs(y_center - last_y) <= tolerance:
            last_line.append(token)
        else:
            lines.append([token])
    for line in lines:
        line.sort(key=lambda token: token.bbox[0])
    return lines


def _line_text(line: list[OcrToken]) -> str:
    return normalize_text(" ".join(token.text for token in line if token.text))


def _line_bbox(line: list[OcrToken]) -> tuple[int, int, int, int]:
    xs1 = [token.bbox[0] for token in line]
    ys1 = [token.bbox[1] for token in line]
    xs2 = [token.bbox[2] for token in line]
    ys2 = [token.bbox[3] for token in line]
    return min(xs1), min(ys1), max(xs2), max(ys2)


def _detect_exam_schedule_cutoff(tokens: list[OcrToken]) -> int | None:
    """
    Find the y-position where an exam schedule section starts.
    Returns the top y of that line if found.
    """
    lines = _cluster_token_lines(tokens)
    for line in lines:
        text = _line_text(line).lower()
        simple = re.sub(r"[^a-z0-9 ]", " ", text)
        simple = " ".join(simple.split())
        if "exam schedule" in simple:
            _x1, y1, _x2, _y2 = _line_bbox(line)
            return y1
        # OCR sometimes breaks these words; accept line-level co-occurrence.
        if "exam" in simple and "sched" in simple:
            _x1, y1, _x2, _y2 = _line_bbox(line)
            return y1
        # Fallback for OCR where the title is missed but exam rows are visible.
        has_date = bool(re.search(r"20\d{2}", simple))
        has_exam_type = ("mid" in simple) or ("final" in simple)
        if has_date and has_exam_type:
            _x1, y1, _x2, _y2 = _line_bbox(line)
            return y1
    return None


def _extract_courses_without_grid(
    tokens: list[OcrToken], result: RoutineParseResult
) -> bool:
    """
    Fallback extractor for borderless templates where contour-based cell detection fails.
    """
    lines = _cluster_token_lines(tokens)
    if not lines:
        return False

    # Identify day labels and their x-centers.
    day_points: list[tuple[str, int, int]] = []  # (day, x_center, y_center)
    for token in tokens:
        day = _normalize_day(token.text)
        if not day:
            continue
        x_center = (token.bbox[0] + token.bbox[2]) // 2
        y_center = (token.bbox[1] + token.bbox[3]) // 2
        day_points.append((day, x_center, y_center))

    if not day_points:
        return False

    # Keep header-like day tokens (top-most instance per day).
    day_groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for day, x, y in day_points:
        day_groups[day].append((x, y))

    day_centers: list[tuple[str, int]] = []
    for day, points in day_groups.items():
        points.sort(key=lambda item: item[1])
        top_slice = points[: max(1, len(points) // 2)]
        avg_x = int(sum(point[0] for point in top_slice) / len(top_slice))
        day_centers.append((day, avg_x))
    day_centers.sort(key=lambda item: item[1])

    # Use left side of first day as rough boundary for time column.
    min_day_x = min(x for _, x in day_centers)
    time_lines: list[tuple[str, str, int]] = []
    for line in lines:
        text = _line_text(line)
        if not text:
            continue
        start, end = _extract_time_range(text)
        if not (start and end):
            continue
        x1, y1, x2, y2 = _line_bbox(line)
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        if x_center < (min_day_x - 20):
            time_lines.append((start, end, y_center))

    if not time_lines:
        # fallback: accept any detected time lines
        for line in lines:
            text = _line_text(line)
            start, end = _extract_time_range(text)
            if start and end:
                _x1, y1, _x2, y2 = _line_bbox(line)
                time_lines.append((start, end, (y1 + y2) // 2))

    if not time_lines:
        return False

    # Bucket non-header/non-time tokens by nearest (day, time-row).
    bucket_tokens: dict[tuple[str, int], list[OcrToken]] = defaultdict(list)
    header_day_token_ids = {
        id(token)
        for token in tokens
        if _normalize_day(token.text) and ((token.bbox[1] + token.bbox[3]) // 2) < min(y for _, _, y in day_points) + 80
    }

    for token in tokens:
        token_id = id(token)
        if token_id in header_day_token_ids:
            continue

        x_center = (token.bbox[0] + token.bbox[2]) // 2
        y_center = (token.bbox[1] + token.bbox[3]) // 2
        if x_center < (min_day_x - 20):
            # likely time column text
            continue

        closest_day = min(day_centers, key=lambda item: abs(item[1] - x_center))[0]
        closest_time_idx = min(
            range(len(time_lines)),
            key=lambda idx: abs(time_lines[idx][2] - y_center),
        )
        bucket_tokens[(closest_day, closest_time_idx)].append(token)

    for (day, time_idx), items in bucket_tokens.items():
        if not items:
            continue
        items.sort(key=lambda token: (token.bbox[1], token.bbox[0]))
        text = normalize_text(" ".join(token.text for token in items))
        parsed = parse_class_payload(text)
        if not parsed.course_code:
            continue

        start, end, _y = time_lines[time_idx]
        confidence = sum(token.confidence for token in items) / len(items)
        class_type = "lab" if parsed.course_code.endswith("L") else "theory"
        result.classes.append(
            ClassEntry(
                course_code=parsed.course_code,
                section=parsed.section,
                faculty=parsed.faculty,
                room=parsed.room,
                day=day,
                start_time=start,
                end_time=end,
                class_type=class_type,
                confidence=confidence,
                raw_text=text,
            )
        )

    return bool(result.classes)


def _token_in_cell(token: OcrToken, cell: GridCell) -> bool:
    x1, y1, x2, y2 = token.bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cell.x <= cx <= cell.x + cell.w and cell.y <= cy <= cell.y + cell.h


class GridRoutineExtractor(RoutineExtractor):
    """
    Draft extractor for grid-like class routine screenshots.

    NOTE:
    This class is intentionally scaffold-level. Replace TODO sections with:
      1. table detection
      2. cell OCR
      3. day/time mapping
      4. class payload parsing
    """

    def __init__(self) -> None:
        self.ocr = OcrEngine(language="en")

    def extract(self, image_path: str) -> RoutineParseResult:
        result = RoutineParseResult(
            meta=ParseMeta(
                source_image=image_path,
                warnings=[],
            )
        )

        if cv2 is None:
            result.meta.warnings.append("opencv is not available; grid extraction skipped.")
            return result

        image = cv2.imread(image_path)
        if image is None:
            result.meta.warnings.append("Failed to load image.")
            return result

        if not self.ocr.available:
            if self.ocr.last_error and "used " not in self.ocr.last_error:
                result.meta.warnings.append(self.ocr.last_error)
            result.meta.warnings.append("OCR engine unavailable; install/repair OCR backend.")
            return result

        tokens = self.ocr.extract_tokens(image)
        if not tokens:
            if self.ocr.last_error and "used " not in self.ocr.last_error:
                result.meta.warnings.append(self.ocr.last_error)
            result.meta.warnings.append("OCR produced no tokens.")
            return result

        exam_cutoff = _detect_exam_schedule_cutoff(tokens)
        if exam_cutoff is not None:
            image_height = image.shape[0]
            # Ignore suspiciously early cutoffs to avoid truncating routine body.
            if exam_cutoff > int(image_height * 0.35):
                crop_end = max(1, exam_cutoff - 6)
                cropped = image[:crop_end, :]
                cropped_tokens = self.ocr.extract_tokens(cropped)
                if cropped_tokens:
                    image = cropped
                    tokens = cropped_tokens
                    result.meta.warnings.append("Exam schedule section detected and cropped.")

        cells = detect_grid_cells(image)
        if not cells:
            # Primary path for borderless templates: time/day anchored extraction.
            anchor_result = RoutineParseResult(meta=ParseMeta(source_image=image_path))
            anchor_extracted = _extract_courses_without_grid(tokens=tokens, result=anchor_result)
            if not anchor_extracted:
                result.meta.warnings.append("No table cells detected.")
                result.meta.warnings.append("Anchor extraction found no class rows.")
                return result
            result.classes.extend(anchor_result.classes)
            _write_debug_payload(
                {
                    "source_image": image_path,
                    "mode": "anchor_primary_no_grid",
                    "token_count": len(tokens),
                    "class_count": len(result.classes),
                    "classes": [entry.__dict__ for entry in result.classes],
                }
            )
            return result

        header_row = min(cell.row_idx for cell in cells)
        time_col = min(cell.col_idx for cell in cells)

        day_by_col: dict[int, str] = {}
        time_by_row: dict[int, tuple[str, str]] = {}
        text_by_cell: dict[tuple[int, int], tuple[str, float, GridCell]] = {}

        for cell in cells:
            cell_tokens = [token for token in tokens if _token_in_cell(token, cell)]
            if not cell_tokens:
                continue
            cell_tokens.sort(key=lambda t: (t.bbox[1], t.bbox[0]))
            text = normalize_text(" ".join(t.text for t in cell_tokens))
            if not text:
                continue
            confidence = sum(t.confidence for t in cell_tokens) / len(cell_tokens)
            text_by_cell[(cell.row_idx, cell.col_idx)] = (text, confidence, cell)

            if cell.row_idx == header_row and cell.col_idx != time_col:
                day_value = _normalize_day(text)
                if day_value:
                    day_by_col[cell.col_idx] = day_value

            if cell.col_idx == time_col and cell.row_idx != header_row:
                start_end = _extract_time_range(text)
                if start_end[0] and start_end[1]:
                    time_by_row[cell.row_idx] = (start_end[0], start_end[1])

        inferred_header_row = _pick_header_row(text_by_cell)
        inferred_time_col = _pick_time_col(text_by_cell)
        if inferred_header_row is not None:
            header_row = inferred_header_row
        if inferred_time_col is not None:
            time_col = inferred_time_col

        # Rebuild axis maps with inferred row/column anchors.
        day_by_col = {}
        time_by_row = {}
        for (row_idx, col_idx), (text, _confidence, _cell) in text_by_cell.items():
            if row_idx == header_row and col_idx != time_col:
                day_value = _normalize_day(text)
                if day_value:
                    day_by_col[col_idx] = day_value
            if col_idx == time_col and row_idx != header_row:
                start_end = _extract_time_range(text)
                if start_end[0] and start_end[1]:
                    time_by_row[row_idx] = (start_end[0], start_end[1])

        if not day_by_col:
            # Fallback: map day columns by position order when header OCR is poor.
            content_cols = sorted({col_idx for (_row_idx, col_idx) in text_by_cell if col_idx != time_col})
            if len(content_cols) >= 5:
                ordered_days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
                for idx, col_idx in enumerate(content_cols[: len(ordered_days)]):
                    day_by_col[col_idx] = ordered_days[idx]
                result.meta.warnings.append("Day headers inferred by column order fallback.")
        if not day_by_col:
            result.meta.warnings.append("No day headers mapped.")
        if not time_by_row:
            result.meta.warnings.append("No time slots mapped.")

        for (row_idx, col_idx), (text, confidence, _cell) in text_by_cell.items():
            if row_idx == header_row or col_idx == time_col:
                continue
            day = day_by_col.get(col_idx)
            slot = time_by_row.get(row_idx)
            if not day or not slot:
                continue

            parsed = parse_class_payload(text)
            if not parsed.course_code:
                continue

            class_type = "lab" if parsed.course_code.endswith("L") else "theory"
            result.classes.append(
                ClassEntry(
                    course_code=parsed.course_code,
                    section=parsed.section,
                    faculty=parsed.faculty,
                    room=parsed.room,
                    day=day,
                    start_time=slot[0],
                    end_time=slot[1],
                    class_type=class_type,
                    confidence=confidence,
                    raw_text=text,
                )
            )

        debug_payload = {
            "source_image": image_path,
            "cell_count": len(cells),
            "token_count": len(tokens),
            "header_row": header_row,
            "time_col": time_col,
            "day_by_col": {str(key): value for key, value in sorted(day_by_col.items())},
            "time_by_row": {
                str(key): {"start": value[0], "end": value[1]}
                for key, value in sorted(time_by_row.items())
            },
            "class_count": len(result.classes),
            "cell_texts": _summarize_cell_texts(text_by_cell),
        }
        _write_debug_payload(debug_payload)

        if not result.classes:
            anchor_result = RoutineParseResult(meta=ParseMeta(source_image=image_path))
            anchor_extracted = _extract_courses_without_grid(tokens=tokens, result=anchor_result)
            if anchor_extracted:
                result.classes.extend(anchor_result.classes)
                result.meta.warnings.append(
                    "Grid parsing missed rows; used anchor fallback extraction."
                )
            else:
                if self.ocr.last_error and "used " not in self.ocr.last_error:
                    result.meta.warnings.append(self.ocr.last_error)
                result.meta.warnings.append("No class entries extracted from detected cells.")

        return result
