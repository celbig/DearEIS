# DearEIS is licensed under the GPLv3 or later (https://www.gnu.org/licenses/gpl-3.0.html).
# Copyright 2026 DearEIS developers
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# The licenses of DearEIS' dependencies and/or sources of portions of code are included in
# the LICENSES folder.

from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import to_hex
from numpy import (
    floating,
    integer,
    issubdtype,
    ndarray,
    pi,
)
from pyimpspec.plot import mpl
from deareis.data import (
    DRTPeakSeries,
    DRTResult,
    DataSet,
    FitResult,
    PlotSettings,
    Project,
    SimulationResult,
    KramersKronigResult,
)
from deareis.config.defaults import DEFAULT_COLORS
from deareis.enums import PlotType


def _plot_drt_peak_series(
    series: DRTPeakSeries,
    axis: Axes,
    color: List[float],
    label: Optional[str],
    frequency: bool,
):
    has_imaginary: bool = series.has_imaginary()
    analysis = series.peak_analysis
    first: bool = True
    part: str
    index: int
    tau: ndarray
    gammas: ndarray
    for part, index, tau, gammas in series.get_curves():
        x: ndarray = (1 / (2 * pi * tau)) if frequency else tau
        if index >= 0:
            axis.fill_between(
                x,
                0.0,
                gammas,
                color=[_ / 255.0 for _ in analysis.get_peak_color(part, index)],
                linewidth=0.0,
                label=label if (first and label) else None,
            )
            first = False
            continue

        curve_label: Optional[str] = label or None
        if curve_label is not None and has_imaginary:
            curve_label += f", {part}"

        axis.plot(
            x,
            gammas,
            color=color,
            linestyle="--" if part == "imag." else "-",
            label=curve_label,
        )

    axis.set_xscale("log")
    if axis.get_xlabel() == "":
        axis.set_xlabel(r"$f\ (\mathrm{Hz})$" if frequency else r"$\tau\ (\mathrm{s})$")

    if axis.get_ylabel() == "":
        axis.set_ylabel(r"$\gamma\ (\Omega)$")


MPL_MARKERS: Dict[int, str] = {
    dpg.mvPlotMarker_Down: "v",
    dpg.mvPlotMarker_Left: "<",
    dpg.mvPlotMarker_Right: ">",
    dpg.mvPlotMarker_Up: "^",
    dpg.mvPlotMarker_Asterisk: "*",
    dpg.mvPlotMarker_Circle: "o",
    dpg.mvPlotMarker_Cross: "x",
    dpg.mvPlotMarker_Diamond: "D",
    dpg.mvPlotMarker_Plus: "+",
    dpg.mvPlotMarker_Square: "s",
}

_UNFILLED_MARKERS: List[int] = [
    dpg.mvPlotMarker_Cross,
    dpg.mvPlotMarker_Plus,
]


def plot(
    settings: PlotSettings,
    project: Project,
    x_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
    y_limits: Optional[Tuple[Optional[float], Optional[float]]] = None,
    title: bool = True,
    legend: Optional[bool] = None,
    legend_loc: Union[int, str] = 0,
    grid: bool = False,
    tight_layout: bool = False,
    figure: Optional[Figure] = None,
    axes: List[Axes] = None,
    num_per_decade: int = 100,
    subcircuits: bool = False,
    subcircuit_offset: str = "none",
) -> Tuple[Figure, Axes]:
    """
    Plot a complex plot containing one or more items from a project based on the provided settings.

    Parameters
    ----------
    settings: PlotSettings
        The settings for the plot.

    project: Project
        The project that the plot is a part of.

    x_limits: Optional[Tuple[Optional[float], Optional[float]]], optional
        The lower and upper limits of the x-axis.

    y_limits: Optional[Tuple[Optional[float], Optional[float]]], optional
        The lower and upper limits of the y-axis.

    title: bool, optional
        Whether or not to include the title in the figure.

    legend: Optional[bool], optional
        Whether or not to include a legend in the figure.

    legend_loc: Union[int, str], optional
        The position of the legend in the figure. See matplotlib's documentation for valid values.

    grid: bool, optional
        Whether or not to include a grid in the figure.

    tight_layout: bool, optional
        Whether or not to apply a tight layout that the sizes of the reduces margins.

    figure: Optional[|Figure|], optional
        The matplotlib.figure.Figure instance to use when plotting the data.

    axes: List[Axes], optional
        The matplotlib.axes.Axes instance to use when plotting the data.

    num_per_decade: int, optional
        If any circuit fits, circuit simulations, or Kramers-Kronig test results are included in the plot, then this parameter can be used to change how many points are used to draw the line (i.e. how smooth or angular the line looks).

    subcircuits: bool, optional
        Whether or not to plot the contribution of each component of the top-level series connection of a fitted circuit (e.g., R, (RC), and (RW) for R(RC)(RW)).
        Only applies to circuit fits in Nyquist plots of the impedance.

    subcircuit_offset: str, optional
        Where the contributions of the components are placed on the real axis ('none' or 'series_resistance').
        See :func:`~pyimpspec.mpl.plot_subcircuits`.

    Returns
    -------
    Tuple[|Figure|, List[|Axes|]]
    """
    assert type(settings) is PlotSettings, settings
    assert type(project) is Project, project
    assert x_limits is None or (
        type(x_limits) is tuple
        and len(x_limits) == 2
        and all(
            map(
                lambda _: _ is None
                or issubdtype(type(_), integer)
                or issubdtype(type(_), floating),
                x_limits,
            )
        )
    ), x_limits
    assert y_limits is None or (
        type(y_limits) is tuple
        and len(y_limits) == 2
        and all(
            map(
                lambda _: _ is None
                or issubdtype(type(_), integer)
                or issubdtype(type(_), floating),
                y_limits,
            )
        )
    ), y_limits
    assert type(title) is bool, title
    assert type(legend) is bool or legend is None, legend
    assert issubdtype(type(legend_loc), integer) or type(legend_loc) is str, legend_loc
    assert type(grid) is bool, grid
    assert type(tight_layout) is bool, tight_layout
    assert type(subcircuits) is bool, subcircuits
    assert subcircuit_offset in ("none", "series_resistance"), subcircuit_offset
    assert type(figure) is Figure or figure is None
    axis: Axes
    if figure is None:
        assert axes is None
        figure, axis = plt.subplots()
        axes = [axis]
    else:
        assert len(axes) > 0
        axis = axes[0]
    assert axis is not None
    assert (
        issubdtype(type(num_per_decade), integer) and num_per_decade >= 1
    ), num_per_decade
    plot_type: PlotType = settings.get_type()
    admittance: bool = plot_type in (
        PlotType.ADMITTANCE_REAL,
        PlotType.ADMITTANCE_IMAGINARY,
        PlotType.BODE_ADMITTANCE_MAGNITUDE,
        PlotType.BODE_ADMITTANCE_PHASE,
        PlotType.NYQUIST_ADMITTANCE,
    )

    selected_series: List[
        Tuple[str, Union[DataSet, KramersKronigResult, DRTResult, FitResult, SimulationResult]]
    ] = []

    uuid: str
    for uuid in settings.series_order:
        series: Optional[
            Union[DataSet, KramersKronigResult, DRTResult, FitResult, SimulationResult]
        ]
        series = settings.find_series(
            uuid=uuid,
            data_sets=project.get_data_sets(),
            tests=project.get_all_tests(),
            zhits=project.get_all_zhits(),
            drts=project.get_all_drts(),
            fits=project.get_all_fits(),
            simulations=project.get_simulations(),
        )
        if series is None:
            continue

        # Filter out stuff that cannot be plotted
        is_drt_plot: bool = plot_type in (PlotType.DRT, PlotType.DRT_FREQUENCY)
        if plot_type == PlotType.DRT and type(series) not in (
            DRTResult,
            DRTPeakSeries,
        ):
            continue
        elif type(series) is DRTPeakSeries and not is_drt_plot:
            continue

        selected_series.append(
            (
                uuid,
                series,
            )
        )

    num_series: int = len(selected_series)
    for i, (uuid, series) in enumerate(selected_series):
        label: Optional[str] = settings.get_series_label(uuid) or series.get_label()
        if label.strip() == "" and label != "":  # type: ignore
            label = ""

        if label is not None and legend is None:
            legend = True

        color: List[float] = list(
            map(lambda _: _ / 255.0, settings.get_series_color(uuid))
        )
        has_line: bool = settings.get_series_line(uuid)
        marker: Optional[str] = MPL_MARKERS.get(settings.get_series_marker(uuid))

        scatter_data: List[ndarray]
        line_data: List[ndarray]
        if plot_type in (PlotType.NYQUIST_IMPEDANCE, PlotType.NYQUIST_ADMITTANCE):
            if (
                subcircuits
                and plot_type == PlotType.NYQUIST_IMPEDANCE
                and type(series) is FitResult
            ):
                # Drawn first so that the fit is drawn on top of the contributions
                mpl.plot_subcircuits(
                    series,
                    offset=subcircuit_offset,
                    colors=[
                        to_hex([_ / 255.0 for _ in DEFAULT_COLORS[f"nyquist_subcircuit_{j}"]])
                        for j in range(1, 6)
                    ],
                    num_per_decade=num_per_decade,
                    legend=False,
                    figure=figure,
                    axes=[axis],
                    adjust_axes=False,
                )

            if has_line:
                mpl.plot_nyquist(
                    series,
                    colors={"impedance": color},
                    markers={"impedance": marker},
                    line=True,
                    label=label,
                    legend=False,
                    figure=figure,
                    axes=[axis],
                    num_per_decade=num_per_decade,
                    adjust_axes=i == num_series - 1,
                    admittance=admittance,
                )

            if marker is not None:
                mpl.plot_nyquist(
                    series,
                    colors={"impedance": color},
                    markers={"impedance": marker},
                    line=False,
                    label="" if has_line else label,
                    legend=False,
                    figure=figure,
                    axes=[axis],
                    num_per_decade=-1,
                    adjust_axes=i == num_series - 1,
                    admittance=admittance,
                )

        elif plot_type in (
            PlotType.BODE_IMPEDANCE_MAGNITUDE,
            PlotType.BODE_ADMITTANCE_MAGNITUDE,
        ):
            if has_line:
                mpl.plot_magnitude(
                    series,
                    colors={"magnitude": color},
                    markers={"magnitude": marker},
                    line=True,
                    label=label,
                    legend=False,
                    figure=figure,
                    axes=[axis],
                    num_per_decade=num_per_decade,
                    adjust_axes=i == num_series - 1,
                    admittance=admittance,
                )

            if marker is not None:
                mpl.plot_magnitude(
                    series,
                    colors={"magnitude": color},
                    markers={"magnitude": marker},
                    line=False,
                    label="" if has_line else label,
                    legend=False,
                    figure=figure,
                    axes=[axis],
                    num_per_decade=-1,
                    adjust_axes=i == num_series - 1,
                    admittance=admittance,
                )

        elif plot_type in (
            PlotType.BODE_IMPEDANCE_PHASE,
            PlotType.BODE_ADMITTANCE_PHASE,
        ):
            if has_line:
                mpl.plot_phase(
                    series,
                    colors={"phase": color},
                    markers={"phase": marker},
                    line=True,
                    label=label,
                    legend=False,
                    figure=figure,
                    axes=[axis],
                    num_per_decade=num_per_decade,
                    adjust_axes=i == num_series - 1,
                    admittance=admittance,
                )

            if marker is not None:
                mpl.plot_phase(
                    series,
                    colors={"phase": color},
                    markers={"phase": marker},
                    line=False,
                    label="" if has_line else label,
                    legend=False,
                    figure=figure,
                    axes=[axis],
                    num_per_decade=-1,
                    adjust_axes=i == num_series - 1,
                    admittance=admittance,
                )

        elif plot_type in (PlotType.DRT, PlotType.DRT_FREQUENCY):
            if isinstance(series, DRTPeakSeries):
                _plot_drt_peak_series(
                    series,
                    axis=axis,
                    color=color,
                    label=label,
                    frequency=plot_type == PlotType.DRT_FREQUENCY,
                )
            elif isinstance(series, DRTResult):
                mpl.plot_gamma(
                    series,
                    colors={"gamma": color},
                    label=label,
                    legend=False,
                    figure=figure,
                    axes=[axis],
                    adjust_axes=i == num_series - 1,
                    frequency=plot_type == PlotType.DRT_FREQUENCY,
                )

        elif plot_type in (PlotType.IMPEDANCE_REAL, PlotType.ADMITTANCE_REAL):
            if has_line:
                mpl.plot_real(
                    series,
                    colors={"real": color},
                    markers={"real": marker},
                    line=True,
                    label=label,
                    legend=False,
                    figure=figure,
                    axes=[axis],
                    num_per_decade=num_per_decade,
                    adjust_axes=i == num_series - 1,
                    admittance=admittance,
                )
            if marker is not None:
                mpl.plot_real(
                    series,
                    colors={"real": color},
                    markers={"real": marker},
                    line=False,
                    label="" if has_line else label,
                    legend=False,
                    figure=figure,
                    axes=[axis],
                    num_per_decade=-1,
                    adjust_axes=i == num_series - 1,
                    admittance=admittance,
                )

        elif plot_type in (PlotType.IMPEDANCE_IMAGINARY, PlotType.ADMITTANCE_IMAGINARY):
            if has_line:
                mpl.plot_imaginary(
                    series,
                    colors={"imaginary": color},
                    markers={"imaginary": marker},
                    line=True,
                    label=label,
                    legend=False,
                    figure=figure,
                    axes=[axis],
                    num_per_decade=num_per_decade,
                    adjust_axes=i == num_series - 1,
                    admittance=admittance,
                )

            if marker is not None:
                mpl.plot_imaginary(
                    series,
                    colors={"imaginary": color},
                    markers={"imaginary": marker},
                    line=False,
                    label="" if has_line else label,
                    legend=False,
                    figure=figure,
                    axes=[axis],
                    num_per_decade=-1,
                    adjust_axes=i == num_series - 1,
                    admittance=admittance,
                )
        else:
            raise NotImplementedError(f"Unsupported plot type: {plot_type=}")

    if x_limits is not None:
        axis.set_xlim(x_limits)

    if y_limits is not None:
        axis.set_ylim(y_limits)

    if title and settings.get_label() != "":
        figure.suptitle(settings.get_label())

    if legend:
        axis.legend(loc=legend_loc)

    axis.grid(visible=grid)

    if tight_layout:
        figure.tight_layout()

    return (
        figure,
        axes,
    )
