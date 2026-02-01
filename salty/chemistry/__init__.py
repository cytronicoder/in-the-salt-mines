"""Henderson-Hasselbalch regression and buffer region selection logic.

This module provides chemistry-specific calculations for the IA investigation,
including ionic strength calculations, buffer region selection, and H-H fitting.
"""

from .buffer_region import select_buffer_region
from .hh_model import fit_henderson_hasselbalch
from .ionic_strength import ionic_strength_general, ionic_strength_nacl

__all__ = [
    "select_buffer_region",
    "fit_henderson_hasselbalch",
    "ionic_strength_nacl",
    "ionic_strength_general",
]
