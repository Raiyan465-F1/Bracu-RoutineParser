from __future__ import annotations

from routine_parser.extractors.base import RoutineExtractor
from routine_parser.extractors.template_grid import GridRoutineExtractor
from routine_parser.router import TemplateId, route_template
from routine_parser.schema import RoutineParseResult


class RoutineParser:
    """
    Orchestrates template routing and extraction.
    """

    def __init__(self) -> None:
        shared_grid_extractor = GridRoutineExtractor()
        self.extractors: dict[TemplateId, RoutineExtractor] = {
            TemplateId.LIGHT_GRID: shared_grid_extractor,
            TemplateId.DARK_GRID: shared_grid_extractor,
            TemplateId.DUAL_TABLE: shared_grid_extractor,
            TemplateId.UNKNOWN: shared_grid_extractor,
        }

    def parse_image(self, image_path: str, ocr_preview_text: str = "") -> RoutineParseResult:
        decision = route_template(image_path=image_path, ocr_preview_text=ocr_preview_text)
        extractor = self.extractors.get(decision.template_id, self.extractors[TemplateId.UNKNOWN])
        result = extractor.extract(image_path=image_path)
        result.meta.template_id = decision.template_id.value
        result.meta.confidence = decision.confidence
        result.meta.warnings.extend([f"router: {reason}" for reason in decision.reasons])
        return result
