from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover
    PaddleOCR = None  # type: ignore[assignment]

try:
    import easyocr
except Exception:  # pragma: no cover
    easyocr = None  # type: ignore[assignment]

try:
    from doctr.io import DocumentFile
    from doctr.models import ocr_predictor
except Exception:  # pragma: no cover
    DocumentFile = None  # type: ignore[assignment]
    ocr_predictor = None  # type: ignore[assignment]


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
    def __init__(self, language: str = "en", backend: str | None = None) -> None:
        self.language = language
        requested = backend or os.getenv("OCR_BACKEND", "doctr,easyocr,paddle")
        self.backend_order = [name.strip().lower() for name in requested.split(",") if name.strip()]

        self._paddle_engine = None
        self._easy_engine = None
        self._doctr_engine = None
        self.active_backend: str | None = None
        self.last_error: str | None = None

    @property
    def available(self) -> bool:
        if cv2 is None:
            return False
        return any(self._backend_available(name) for name in self.backend_order)

    def _backend_available(self, name: str) -> bool:
        if name == "doctr":
            return DocumentFile is not None and ocr_predictor is not None
        if name == "easyocr":
            return easyocr is not None
        if name == "paddle":
            return PaddleOCR is not None
        return False

    def _get_paddle_engine(self):
        if self._paddle_engine is not None:
            return self._paddle_engine
        if PaddleOCR is None:
            return None
        try:
            # PaddleOCR 2.x supports show_log.
            self._paddle_engine = PaddleOCR(
                use_angle_cls=False, lang=self.language, show_log=False
            )
        except Exception:
            try:
                # PaddleOCR 3.x rejects show_log and raises ValueError.
                self._paddle_engine = PaddleOCR(use_angle_cls=False, lang=self.language)
            except Exception as exc:
                self.last_error = f"paddle init failed: {type(exc).__name__}: {exc}"
                self._paddle_engine = None
        return self._paddle_engine

    def _get_easy_engine(self):
        if easyocr is None:
            return None
        if self._easy_engine is None:
            self._easy_engine = easyocr.Reader([self.language], gpu=False)
        return self._easy_engine

    def _get_doctr_engine(self):
        if self._doctr_engine is not None:
            return self._doctr_engine
        if ocr_predictor is None:
            return None
        # CPU-only is controlled by installed torch build in the venv.
        self._doctr_engine = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
        )
        return self._doctr_engine

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

    def _extract_tokens_doctr(self, image: Any) -> list[OcrToken]:
        predictor = self._get_doctr_engine()
        if predictor is None or image is None or DocumentFile is None:
            return []

        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        doc = DocumentFile.from_images([rgb])
        output = predictor(doc)

        tokens: list[OcrToken] = []
        if not output.pages:
            return tokens

        page = output.pages[0]
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    text = str(word.value).strip()
                    if not text:
                        continue
                    # geometry is normalized: ((xmin, ymin), (xmax, ymax))
                    (xmin, ymin), (xmax, ymax) = word.geometry
                    x1 = int(xmin * width)
                    y1 = int(ymin * height)
                    x2 = int(xmax * width)
                    y2 = int(ymax * height)
                    confidence = float(getattr(word, "confidence", 0.0) or 0.0)
                    tokens.append(
                        OcrToken(text=text, confidence=confidence, bbox=(x1, y1, x2, y2))
                    )
        return tokens

    def _extract_tokens_paddle(self, image: Any) -> list[OcrToken]:
        engine = self._get_paddle_engine()
        if engine is None or image is None:
            return []

        try:
            try:
                # PaddleOCR 2.x accepted cls=False.
                output = engine.ocr(image, cls=False)
            except TypeError:
                # PaddleOCR 3.x removed cls from predict()/ocr kwargs.
                output = engine.ocr(image)
        except Exception as exc:
            self.last_error = f"paddle inference failed: {type(exc).__name__}: {exc}"
            return []
        if not output:
            return []

        tokens: list[OcrToken] = []
        # PaddleOCR 2.x: output[0] = [[poly, (text, score)], ...]
        if isinstance(output, list) and output and isinstance(output[0], list):
            for line in output[0]:
                if not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                points = line[0]
                text_conf = line[1]
                if not isinstance(text_conf, (list, tuple)) or len(text_conf) < 2:
                    continue
                text, confidence = text_conf[0], text_conf[1]
                if not text:
                    continue

                xs = [int(p[0]) for p in points]
                ys = [int(p[1]) for p in points]
                x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

                tokens.append(
                    OcrToken(
                        text=str(text).strip(),
                        confidence=float(confidence),
                        bbox=(x1, y1, x2, y2),
                    )
                )
        # PaddleOCR 3.x may return list[dict] with rec_text/rec_score/dt_polys
        elif isinstance(output, list) and output and isinstance(output[0], dict):
            for item in output:
                text = str(item.get("rec_text", "")).strip()
                if not text:
                    continue
                confidence = float(item.get("rec_score", 0.0))
                polys = item.get("dt_polys")
                bbox = (0, 0, 0, 0)
                if polys is not None:
                    try:
                        points = list(polys)
                        if points:
                            xs = [int(p[0]) for p in points]
                            ys = [int(p[1]) for p in points]
                            bbox = (min(xs), min(ys), max(xs), max(ys))
                    except Exception:
                        bbox = (0, 0, 0, 0)
                tokens.append(OcrToken(text=text, confidence=confidence, bbox=bbox))
        else:
            self.last_error = f"unrecognized OCR output type: {type(output).__name__}"
            return []
        return tokens

    def extract_tokens(self, image: Any) -> list[OcrToken]:
        """
        Returns OCR tokens with simple rectangular bbox per token.
        """
        if not self.available or image is None:
            return []

        prepared = preprocess_for_ocr(image)
        failures: list[str] = []

        for backend in self.backend_order:
            try:
                if backend == "doctr":
                    tokens = self._extract_tokens_doctr(prepared)
                elif backend == "easyocr":
                    tokens = self._extract_tokens_easyocr(prepared)
                elif backend == "paddle":
                    tokens = self._extract_tokens_paddle(prepared)
                else:
                    failures.append(f"unknown backend '{backend}'")
                    continue
            except Exception as exc:
                failures.append(f"{backend} failed: {type(exc).__name__}: {exc}")
                continue

            if tokens:
                self.active_backend = backend
                self.last_error = (
                    f"used {backend}" if not failures else f"{'; '.join(failures)}; used {backend}"
                )
                return tokens

            if self.last_error:
                failures.append(self.last_error)
            else:
                failures.append(f"{backend} produced no tokens")

        self.active_backend = None
        self.last_error = "; ".join(failures) if failures else "no OCR backend available"
        return []

    def extract_text(self, image: Any) -> tuple[str, float]:
        tokens = self.extract_tokens(image)
        if not tokens:
            return "", 0.0
        text = " ".join(token.text for token in tokens if token.text)
        confidence = sum(token.confidence for token in tokens) / len(tokens)
        return text.strip(), confidence
