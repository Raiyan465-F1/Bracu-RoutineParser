from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None  # type: ignore[assignment]


@dataclass
class OcrToken:
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]


def preprocess_for_ocr(image: Any) -> Any:
    """
    Lightweight preprocessing that works across light/dark routine screenshots.
    """
    if cv2 is None or image is None:
        return image

    if len(image.shape) == 3:
        bgr = image
    else:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Keep natural text edges for modern OCR backends (docTR/EasyOCR).
    # Heavy binarization hurt recognition on schedule headers/time labels.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return cv2.bilateralFilter(enhanced, 5, 25, 25)


class OcrEngine:
    def __init__(self, language: str = "en") -> None:
        self.language = language
        self._easy_engine = None
        self.active_backend: str = "easyocr"
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        return cv2 is not None and easyocr is not None

    def _get_easy_engine(self):
        if easyocr is None:
            return None
        if self._easy_engine is None:
            self._easy_engine = easyocr.Reader([self.language], gpu=False)
        return self._easy_engine

    def _extract_tokens_easyocr(self, image: Any) -> list[OcrToken]:
        reader = self._get_easy_engine()
        if reader is None or image is None:
            return []
        results = reader.readtext(image, detail=1, paragraph=False)
        tokens: list[OcrToken] = []
        for item in results:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            points, text, confidence = item[0], item[1], item[2]
            if not text:
                continue
            xs = [int(p[0]) for p in points]
            ys = [int(p[1]) for p in points]
            tokens.append(
                OcrToken(
                    text=str(text).strip(),
                    confidence=float(confidence),
                    bbox=(min(xs), min(ys), max(xs), max(ys)),
                )
            )
        return tokens

    def extract_tokens(self, image: Any) -> list[OcrToken]:
        """
        Returns OCR tokens with simple rectangular bbox per token.
        """
        if not self.available or image is None:
            if easyocr is None:
                self.last_error = "easyocr is not installed"
            elif cv2 is None:
                self.last_error = "opencv (cv2) is not installed"
            return []

        prepared = preprocess_for_ocr(image)
        try:
            tokens = self._extract_tokens_easyocr(prepared)
        except Exception as exc:  # pragma: no cover - runtime backend failures
            self.last_error = f"easyocr inference failed: {type(exc).__name__}: {exc}"
            return []
        if not tokens:
            self.last_error = "easyocr produced no tokens"
            return []
        self.last_error = "used easyocr"
        return tokens

    def extract_text(self, image: Any) -> tuple[str, float]:
        tokens = self.extract_tokens(image)
        if not tokens:
            return "", 0.0
        text = " ".join(token.text for token in tokens if token.text)
        confidence = sum(token.confidence for token in tokens) / len(tokens)
        return text.strip(), confidence
