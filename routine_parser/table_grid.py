from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


@dataclass
class GridCell:
    row_idx: int
    col_idx: int
    x: int
    y: int
    w: int
    h: int


def _cluster_positions(values: list[int], tolerance: int) -> list[int]:
    if not values:
        return []
    values = sorted(values)
    clusters: list[list[int]] = [[values[0]]]
    for value in values[1:]:
        if abs(value - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    return [int(sum(cluster) / len(cluster)) for cluster in clusters]


def _nearest_index(value: int, anchors: list[int]) -> int:
    if not anchors:
        return 0
    return min(range(len(anchors)), key=lambda idx: abs(anchors[idx] - value))


def detect_grid_cells(image: Any) -> list[GridCell]:
    """
    Detect rectangular table cells and assign coarse row/column indexes.
    """
    if cv2 is None or image is None:
        return []

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    inverted = cv2.bitwise_not(gray)
    binary = cv2.adaptiveThreshold(
        inverted,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        17,
        -6,
    )

    height, width = gray.shape[:2]
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(width // 18, 25), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(height // 18, 25))
    )

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    grid_mask = cv2.add(horizontal_lines, vertical_lines)
    grid_mask = cv2.dilate(grid_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    contours, _ = cv2.findContours(grid_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[tuple[int, int, int, int]] = []
    min_w = max(width // 30, 30)
    min_h = max(height // 30, 24)
    max_w = int(width * 0.95)
    max_h = int(height * 0.95)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_w or h < min_h:
            continue
        if w > max_w or h > max_h:
            continue
        candidates.append((x, y, w, h))

    if not candidates:
        return []

    # Deduplicate near-identical rectangles.
    candidates.sort(key=lambda item: (item[1], item[0], item[2] * item[3]))
    deduped: list[tuple[int, int, int, int]] = []
    for x, y, w, h in candidates:
        is_duplicate = False
        for dx, dy, dw, dh in deduped:
            if abs(x - dx) <= 4 and abs(y - dy) <= 4 and abs(w - dw) <= 6 and abs(h - dh) <= 6:
                is_duplicate = True
                break
        if not is_duplicate:
            deduped.append((x, y, w, h))

    row_anchors = _cluster_positions(
        [y for _, y, _, _ in deduped], tolerance=max(height // 80, 8)
    )
    col_anchors = _cluster_positions(
        [x for x, _, _, _ in deduped], tolerance=max(width // 80, 8)
    )

    cells: list[GridCell] = []
    for x, y, w, h in deduped:
        row_idx = _nearest_index(y, row_anchors)
        col_idx = _nearest_index(x, col_anchors)
        cells.append(GridCell(row_idx=row_idx, col_idx=col_idx, x=x, y=y, w=w, h=h))

    return sorted(cells, key=lambda c: (c.row_idx, c.col_idx, c.y, c.x))


def extract_cell_image(image: Any, cell: GridCell, padding: int = 2) -> Any:
    if image is None:
        return None
    height, width = image.shape[:2]
    x1 = max(cell.x + padding, 0)
    y1 = max(cell.y + padding, 0)
    x2 = min(cell.x + cell.w - padding, width)
    y2 = min(cell.y + cell.h - padding, height)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]
