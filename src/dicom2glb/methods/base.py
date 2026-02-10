"""Abstract base class for conversion methods."""

from __future__ import annotations

from abc import ABC, abstractmethod

from dicom2glb.core.types import ConversionResult, MethodParams
from dicom2glb.core.volume import DicomVolume


class ConversionMethod(ABC):
    """Base class for all DICOM-to-mesh conversion methods."""

    name: str = ""
    description: str = ""
    recommended_for: str = ""
    requires_ai: bool = False

    @abstractmethod
    def convert(self, volume: DicomVolume, params: MethodParams) -> ConversionResult:
        """Convert a DICOM volume to mesh data."""
        ...

    def supports_animation(self) -> bool:
        """Whether this method supports temporal animation."""
        return False

    @classmethod
    def check_dependencies(cls) -> tuple[bool, str]:
        """Check if required dependencies are installed.

        Returns (available, message).
        """
        return True, "No additional dependencies required."
