from __future__ import annotations

from abc import ABC, abstractmethod

from routine_parser.schema import RoutineParseResult


class RoutineExtractor(ABC):
    @abstractmethod
    def extract(self, image_path: str) -> RoutineParseResult:
        raise NotImplementedError
