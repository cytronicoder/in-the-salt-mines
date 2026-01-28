"""Chemistry-specific models for the two-stage pKa_app workflow.

The chemistry subpackage contains Henderson–Hasselbalch regression and buffer
region selection logic. The results are explicitly interpreted as apparent
pKa_app values influenced by ionic strength via activity coefficients and are
therefore comparative rather than thermodynamic constants.
"""

from .buffer_region import select_buffer_region
from .hh_model import fit_henderson_hasselbalch

__all__ = ["select_buffer_region", "fit_henderson_hasselbalch"]
