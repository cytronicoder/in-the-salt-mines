"""Generate the IB-ready publication figure set and matching caption files."""

from __future__ import annotations

import os
from typing import Dict, List

import pandas as pd

from salty.reporting import generate_ia_caption_texts, write_caption_files

from .qc_plots import plot_temperature_and_calibration_qc
from .style import apply_global_style, figure_base_path
from .summary_plots import (
    plot_hh_linearization_and_diagnostics,
    plot_pka_app_vs_nacl_and_I,
)
from .titration_plots import (
    plot_derivative_equivalence_by_nacl,
    plot_titration_overlays_by_nacl,
)


def _render_ia_bundle(
    results: List[Dict],
    results_df: pd.DataFrame,
    output_dir: str | None,
    summary_csv_path: str,
) -> Dict[str, str]:
    apply_global_style(font_scale=1.0, context="paper")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    titration_dir = os.path.join(output_dir, "titration") if output_dir else None
    methods_dir = (
        os.path.join(output_dir, "methods_or_derivations") if output_dir else None
    )
    summary_dir = os.path.join(output_dir, "summary") if output_dir else None
    diagnostics_dir = os.path.join(output_dir, "diagnostics") if output_dir else None
    qc_dir = os.path.join(output_dir, "qc") if output_dir else None

    for directory in (titration_dir, methods_dir, summary_dir, diagnostics_dir, qc_dir):
        if directory:
            os.makedirs(directory, exist_ok=True)

    paths: Dict[str, str] = {}
    paths["titration_overlays_by_nacl"] = plot_titration_overlays_by_nacl(
        results,
        output_dir=titration_dir,
    )
    paths["derivative_equivalence_by_nacl"] = plot_derivative_equivalence_by_nacl(
        results,
        output_dir=methods_dir,
    )
    fig3_path, fig3_meta = plot_pka_app_vs_nacl_and_I(
        results_df=results_df,
        results=results,
        summary_csv_path=summary_csv_path,
        output_dir=summary_dir,
        return_metadata=True,
    )
    paths["pka_app_vs_nacl_and_I"] = fig3_path
    paths["hh_linearization_and_diagnostics"] = plot_hh_linearization_and_diagnostics(
        results,
        output_dir=diagnostics_dir,
    )
    fig5_path, fig5_meta = plot_temperature_and_calibration_qc(
        results,
        results_df=results_df,
        output_dir=qc_dir,
        return_metadata=True,
    )
    paths["temperature_and_calibration_qc"] = fig5_path

    captions = generate_ia_caption_texts(
        results=results,
        results_df=results_df,
        figure3_fit=fig3_meta,
        figure5_meta=fig5_meta,
    )
    caption_dir = (
        summary_dir
        if summary_dir
        else str(figure_base_path("ia_captions", kind="supplemental").parent)
    )
    write_caption_files(captions, caption_dir)
    return paths


def generate_ia_figure_set(
    results: List[Dict],
    results_df: pd.DataFrame,
    output_dir: str | None = None,
    iteration_output_dir: str | None = None,
    summary_csv_path: str = "output/ia/processed_summary_with_sd.csv",
) -> Dict[str, str]:
    """Generate Figure 1-5 + captions into IA and optional iteration folders."""
    primary = _render_ia_bundle(
        results=results,
        results_df=results_df,
        output_dir=output_dir,
        summary_csv_path=summary_csv_path,
    )
    if iteration_output_dir:
        _render_ia_bundle(
            results=results,
            results_df=results_df,
            output_dir=iteration_output_dir,
            summary_csv_path=summary_csv_path,
        )
    return primary
