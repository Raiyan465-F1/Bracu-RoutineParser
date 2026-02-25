from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency for draft
    cv2 = None


class TemplateId(str, Enum):
    LIGHT_GRID = "light_grid"
    DARK_GRID = "dark_grid"
    DUAL_TABLE = "dual_table"
    UNKNOWN = "unknown"


@dataclass
class RoutingDecision:
    template_id: TemplateId
    confidence: float
    reasons: list[str]


def route_template(image_path: str, ocr_preview_text: str = "") -> RoutingDecision:
    """
    Lightweight image routing.
    v1 uses simple brightness + keyword heuristics and can be replaced later
    with an ML classifier without changing extractor interfaces.
    """
    reasons: list[str] = []

    text = (ocr_preview_text or "").lower()
    if "exam schedule" in text:
        reasons.append("detected phrase: exam schedule")
        return RoutingDecision(TemplateId.DUAL_TABLE, 0.9, reasons)

    if cv2 is None:
        reasons.append("opencv unavailable; defaulting to light grid")
        return RoutingDecision(TemplateId.LIGHT_GRID, 0.5, reasons)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        reasons.append("image load failed")
        return RoutingDecision(TemplateId.UNKNOWN, 0.0, reasons)

    mean_intensity = float(image.mean())
    reasons.append(f"mean intensity={mean_intensity:.2f}")

    if mean_intensity < 95:
        reasons.append("dark background style")
        return RoutingDecision(TemplateId.DARK_GRID, 0.8, reasons)

    reasons.append("light background style")
    return RoutingDecision(TemplateId.LIGHT_GRID, 0.8, reasons)
