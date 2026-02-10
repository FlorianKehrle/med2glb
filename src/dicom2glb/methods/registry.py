"""Method registry with decorator-based registration."""

from __future__ import annotations

from typing import Type

from dicom2glb.methods.base import ConversionMethod

_registry: dict[str, Type[ConversionMethod]] = {}


def register_method(name: str):
    """Decorator to register a conversion method."""

    def decorator(cls: Type[ConversionMethod]):
        cls.name = name
        _registry[name] = cls
        return cls

    return decorator


def get_method(name: str) -> ConversionMethod:
    """Get an instance of a registered method by name."""
    if name not in _registry:
        available = ", ".join(_registry.keys())
        raise ValueError(f"Unknown method '{name}'. Available: {available}")
    return _registry[name]()


def list_methods() -> list[dict[str, str]]:
    """List all registered methods with their info."""
    methods = []
    for name, cls in _registry.items():
        available, msg = cls.check_dependencies()
        methods.append(
            {
                "name": name,
                "description": cls.description,
                "recommended_for": cls.recommended_for,
                "available": available,
                "dependency_message": msg,
            }
        )
    return methods


def _ensure_methods_loaded():
    """Import method modules to trigger registration."""
    try:
        import dicom2glb.methods.marching_cubes  # noqa: F401
    except ImportError:
        pass
    try:
        import dicom2glb.methods.classical  # noqa: F401
    except ImportError:
        pass
    try:
        import dicom2glb.methods.totalseg  # noqa: F401
    except ImportError:
        pass
    try:
        import dicom2glb.methods.medsam2  # noqa: F401
    except ImportError:
        pass
