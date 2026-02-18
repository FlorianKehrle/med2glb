"""Gallery mode: individual GLBs, lightbox grid, and spatial fan."""

from med2glb.gallery.individual import build_individual_glbs
from med2glb.gallery.lightbox import build_lightbox_glb
from med2glb.gallery.loader import load_all_slices
from med2glb.gallery.spatial import build_spatial_glb

__all__ = [
    "load_all_slices",
    "build_individual_glbs",
    "build_lightbox_glb",
    "build_spatial_glb",
]
