from __future__ import annotations

from routine_parser.pipeline import RoutineParser


class CourseSectionParser:
    """
    Minimal parser output for website autofill use-case.
    Returns unique course_code + section pairs only.
    """

    def __init__(self) -> None:
        self._parser = RoutineParser()

    def parse_image(self, image_path: str, ocr_preview_text: str = "") -> dict[str, object]:
        result = self._parser.parse_image(image_path=image_path, ocr_preview_text=ocr_preview_text)

        seen: set[tuple[str, str]] = set()
        course_sections: list[dict[str, str]] = []
        for row in result.classes:
            if not row.course_code or not row.section:
                continue
            key = (row.course_code.upper(), row.section.zfill(2))
            if key in seen:
                continue
            seen.add(key)
            course_sections.append({"course_code": key[0], "section": key[1]})

        course_sections.sort(key=lambda item: (item["course_code"], item["section"]))
        return {
            "course_sections": course_sections,
            "meta": {
                "source_image": result.meta.source_image,
                "template_id": result.meta.template_id,
                "warnings": result.meta.warnings,
            },
        }
