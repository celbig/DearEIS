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

from dataclasses import (
    dataclass,
    field,
    fields,
)
from typing import (
    Callable,
    Dict,
    List,
    Tuple,
    Optional,
)
from numpy import (
    angle,
    array,
    floating,
    abs as np_abs,
    full,
    isnan,
    log as ln,
    pi,
    trapezoid,
    issubdtype,
    nan,
)
from scipy.signal import find_peaks
from pandas import DataFrame
from pyimpspec.analysis.utility import _calculate_pseudo_chisqr
from pyimpspec import (
    ComplexImpedances,
    ComplexResiduals,
    DRTPeak,
    DRTPeakGaussian,
    DRTPeakHavriliakNegami,
    DRTPeakZARC,
    DRTPeaks,
    Frequencies,
    Gamma,
    Gammas,
    Impedances,
    Indices,
    Phases,
    Residuals,
    TimeConstant,
    TimeConstants,
)
from deareis.data.fitting import FitResult
from deareis.enums import (
    CrossValidationMethod,
    DRTLowFrequencyPrior,
    DRTMethod,
    DRTMode,
    DRTPeakShape,
    RBFShape,
    RBFType,
    TRNNLSLambdaMethod,
    cross_validation_method_to_label,
    drt_method_to_label,
    drt_mode_to_label,
    drt_peak_shape_to_value,
    label_to_cross_validation_method,
    label_to_drt_method,
    label_to_drt_mode,
    label_to_rbf_shape,
    label_to_rbf_type,
    label_to_tr_nnls_lambda_method,
    rbf_shape_to_label,
    rbf_type_to_label,
    tr_nnls_lambda_method_to_label,
)
from deareis.utility import (
    format_timestamp,
    rename_dict_entry,
)
from deareis.data import DataSet


VERSION: int = 5

# Default colors (RGBA, 0-255) of DRT peaks (Paul Tol's "muted" qualitative color scheme).
DRT_PEAK_PALETTE: List[Tuple[float, float, float, float]] = [
    (204.0, 102.0, 119.0, 110.0),
    (51.0, 34.0, 136.0, 110.0),
    (221.0, 204.0, 119.0, 110.0),
    (17.0, 119.0, 51.0, 110.0),
    (136.0, 204.0, 238.0, 110.0),
    (136.0, 34.0, 85.0, 110.0),
    (68.0, 170.0, 153.0, 110.0),
    (153.0, 153.0, 51.0, 110.0),
    (170.0, 68.0, 153.0, 110.0),
]
SETTINGS_VERSION: int = 5  # Version 5 added the settings of the Bayesian method (also used in config files)


def _parse_settings_v5(dictionary: dict) -> dict:
    dictionary.setdefault("bayes_cutoff_frequency", 0.0)
    dictionary.setdefault("bayes_low_frequency_prior", DRTLowFrequencyPrior.FREE)
    dictionary.setdefault("bayes_warmup", 200)
    dictionary.setdefault("bayes_samples", 200)

    return dictionary


def _parse_settings_v4(dictionary: dict) -> dict:
    lambda_value: float = dictionary.get("lambda_value", 1e-3)

    if "cross_validation_method" not in dictionary:
        if lambda_value <= 0.0:
            dictionary["cross_validation_method"] = CrossValidationMethod.MGCV
        else:
            dictionary["cross_validation_method"] = CrossValidationMethod.NONE

    if "tr_nnls_lambda_method" not in dictionary:
        if lambda_value <= 0.0:
            dictionary["tr_nnls_lambda_method"] = (
                TRNNLSLambdaMethod.LC
                if lambda_value < -1.5
                else TRNNLSLambdaMethod.CUSTOM
            )
        else:
            dictionary["tr_nnls_lambda_method"] = TRNNLSLambdaMethod.NONE

    return dictionary


def _parse_settings_v3(dictionary: dict) -> dict:
    if "fit" not in dictionary:
        dictionary["fit"] = None

    if "circuit" in dictionary:
        del dictionary["circuit"]

    if "timeout" not in dictionary:
        dictionary["timeout"] = 60

    return dictionary


def _parse_settings_v2(dictionary: dict) -> dict:
    rename_dict_entry(dictionary, "W", "gaussian_width")

    dictionary["fit"] = None

    del dictionary["circuit"]

    return dictionary


def _parse_settings_v1(dictionary: dict) -> dict:
    dictionary["circuit"] = ""

    dictionary["W"] = 0.15

    dictionary["num_per_decade"] = 100

    return dictionary


@dataclass(frozen=True)
class DRTSettings:
    """
    The settings to use when performing a DRT analysis.

    Parameters
    ----------
    method: DRTMethod
        The method to use to perform the analysis.

    mode: DRTMode
        The mode or type of data  (i.e., complex, real, or imaginary) to use.
        TR-NNLS and TR-RBF methods only.

    lambda_value: float
        The Tikhonov regularization parameter to use.
        TR-NNLS and TR-RBF methods only.

    rbf_type: RBFType
        The radial basis function to use for discretization.
        BHT and TR-RBF methods only.

    derivative_order: int
        The derivative order to use when calculating the penalty in the Tikhonov regularization.
        BHT and TR-RBF methods only.

    rbf_shape: RBFShape
        The shape to use with the radial basis function discretization.
        BHT and TR-RBF methods only.

    shape_coeff: float
        The shape coefficient.
        BHT and TR-RBF methods only.

    inductance: bool
        Whether or not to include an inductive term in the calculations.
        TR-RBF method only.

    credible_intervals: bool
        Whether or not to calculate Bayesian credible intervals.
        TR-RBF method only.

    timeout: int
        The number of seconds to wait for the calculation of credible intervals to complete.
        TR-RBF method only.

    num_samples: int
        The number of samples to use when calculating:
        - the Bayesian credible intervals (TR-RBF method)
        - the Jensen-Shannon distance (BHT method)

    num_attempts: int
        The number of attempts to make to find a solution.
        BHT method only.

    maximum_symmetry: float
        The maximum vertical peak-to-peak symmetry allowed.
        Used to discard results with strong oscillations.
        Smaller values provide stricter conditions.
        BHT method only.

    fit: Optional[FitResult]
        The FitResult for a circuit that contains one or more "(RQ)" or "(RC)" elements connected in series.
        An optional series resistance may also be included.
        For example, a circuit with a CDC representation of "R(RQ)(RQ)(RC)" would be a valid circuit.
        m(RQ)fit method only.

    gaussian_width: float
        The width of the Gaussian curve that is used to approximate the DRT of an "(RC)" element.
        m(RQ)fit method only.

    num_per_decade: int
        The number of points per decade to use when calculating a DRT.
        m(RQ)fit method only.

    cross_validation_method: CrossValidationMethod
        The cross-validation method used by the TR-RBF method to automatically determine a suitable value for lambda.

    tr_nnls_lambda_method: TRNNLSLambdaMethod
        The method used by the TR-NNLS method to automatically pick a lambda value

    bayes_cutoff_frequency: float
        The frequency (in hertz) below which the DRT is not constrained to be non-negative.
        A value of zero means that the DRT is non-negative at all frequencies.
        Bayesian method only.

    bayes_low_frequency_prior: DRTLowFrequencyPrior
        The prior on the DRT below the cut-off frequency.
        Bayesian method only.

    bayes_warmup: int
        The number of warmup iterations of each HMC chain (used when credible intervals are calculated).
        Bayesian method only.

    bayes_samples: int
        The number of samples drawn by each HMC chain (used when credible intervals are calculated).
        Bayesian method only.
    """

    method: DRTMethod
    mode: DRTMode
    lambda_value: float
    rbf_type: RBFType
    derivative_order: int
    rbf_shape: RBFShape
    shape_coeff: float
    inductance: bool
    credible_intervals: bool
    timeout: int
    num_samples: int
    num_attempts: int
    maximum_symmetry: float
    fit: Optional[FitResult]
    gaussian_width: float
    num_per_decade: int
    cross_validation_method: CrossValidationMethod
    tr_nnls_lambda_method: TRNNLSLambdaMethod
    bayes_cutoff_frequency: float = 0.0
    bayes_low_frequency_prior: DRTLowFrequencyPrior = DRTLowFrequencyPrior.FREE
    bayes_warmup: int = 200
    bayes_samples: int = 200

    def __repr__(self) -> str:
        return f"DRTSettings ({hex(id(self))})"

    def to_dict(self) -> dict:
        return {
            "version": SETTINGS_VERSION,
            "method": int(self.method),
            "mode": int(self.mode),
            "lambda_value": self.lambda_value,
            "rbf_type": int(self.rbf_type),
            "derivative_order": self.derivative_order,
            "rbf_shape": int(self.rbf_shape),
            "shape_coeff": self.shape_coeff,
            "inductance": self.inductance,
            "credible_intervals": self.credible_intervals,
            "timeout": self.timeout,
            "num_samples": self.num_samples,
            "num_attempts": self.num_attempts,
            "maximum_symmetry": self.maximum_symmetry,
            "fit": self.fit.to_dict(session=False) if self.fit is not None else None,
            "gaussian_width": self.gaussian_width,
            "num_per_decade": self.num_per_decade,
            "cross_validation_method": self.cross_validation_method,
            "tr_nnls_lambda_method": self.tr_nnls_lambda_method,
            "bayes_cutoff_frequency": self.bayes_cutoff_frequency,
            "bayes_low_frequency_prior": int(self.bayes_low_frequency_prior),
            "bayes_warmup": self.bayes_warmup,
            "bayes_samples": self.bayes_samples,
        }

    @classmethod
    def from_dict(Class, dictionary: dict) -> "DRTSettings":
        assert type(dictionary) is dict
        assert "version" in dictionary

        version: int = dictionary["version"]
        del dictionary["version"]

        assert version <= SETTINGS_VERSION, f"{version=} > {SETTINGS_VERSION=}"
        parsers: Dict[int, Callable] = {
            1: _parse_settings_v1,
            2: _parse_settings_v2,
            3: _parse_settings_v3,
            4: _parse_settings_v4,
            5: _parse_settings_v5,
        }
        assert version in parsers, f"{version=} not in {parsers.keys()=}"

        v: int
        p: Callable
        for v, p in parsers.items():
            if v < version:
                continue
            
            dictionary = p(dictionary)

        assert "method" in dictionary
        assert "mode" in dictionary
        assert "lambda_value" in dictionary
        assert "rbf_type" in dictionary
        assert "derivative_order" in dictionary
        assert "rbf_shape" in dictionary
        assert "shape_coeff" in dictionary
        assert "inductance" in dictionary
        assert "credible_intervals" in dictionary
        assert "timeout" in dictionary
        assert "num_samples" in dictionary
        assert "num_attempts" in dictionary
        assert "maximum_symmetry" in dictionary
        assert "fit" in dictionary
        assert "gaussian_width" in dictionary
        assert "num_per_decade" in dictionary
        assert "cross_validation_method" in dictionary
        assert "tr_nnls_lambda_method" in dictionary
        assert "bayes_cutoff_frequency" in dictionary
        assert "bayes_low_frequency_prior" in dictionary
        assert "bayes_warmup" in dictionary
        assert "bayes_samples" in dictionary

        dictionary["method"] = DRTMethod(dictionary["method"])
        dictionary["mode"] = DRTMode(dictionary["mode"])
        dictionary["rbf_type"] = RBFType(dictionary["rbf_type"])
        dictionary["rbf_shape"] = RBFShape(dictionary["rbf_shape"])
        dictionary["cross_validation_method"] = CrossValidationMethod(
            dictionary["cross_validation_method"]
        )
        dictionary["tr_nnls_lambda_method"] = TRNNLSLambdaMethod(
                dictionary["tr_nnls_lambda_method"]
        )
        dictionary["bayes_low_frequency_prior"] = DRTLowFrequencyPrior(
            dictionary["bayes_low_frequency_prior"]
        )
        
        if dictionary["fit"] is not None:
            dictionary["fit"] = FitResult.from_dict(dictionary["fit"])

        return Class(**dictionary)


def _parse_result_v5(dictionary: dict) -> dict:
    if "peak_analysis" not in dictionary:
        dictionary["peak_analysis"] = None

    return dictionary


def _parse_result_v4(dictionary: dict) -> dict:
    return dictionary


def _parse_result_v3(dictionary: dict) -> dict:
    if "pseudo_chisqr" not in dictionary:
        dictionary["pseudo_chisqr"] = nan

    if "chisqr" in dictionary:
        del dictionary["chisqr"]

    return dictionary


def _parse_result_v2(dictionary: dict) -> dict:
    rename_dict_entry(dictionary, "tau", "time_constants")
    rename_dict_entry(dictionary, "gamma", "real_gammas")
    rename_dict_entry(dictionary, "imaginary_gamma", "imaginary_gammas")
    rename_dict_entry(dictionary, "real_impedance", "real_impedances")
    rename_dict_entry(dictionary, "imaginary_impedance", "imaginary_impedances")
    rename_dict_entry(dictionary, "frequency", "frequencies")
    rename_dict_entry(dictionary, "real_residual", "real_residuals")
    rename_dict_entry(dictionary, "imaginary_residual", "imaginary_residuals")
    rename_dict_entry(dictionary, "mean_gamma", "mean_gammas")
    rename_dict_entry(dictionary, "lower_bound", "lower_bounds")
    rename_dict_entry(dictionary, "upper_bound", "upper_bounds")
    dictionary["chisqr"] = nan

    return dictionary


def _parse_result_v1(dictionary: dict) -> dict:
    return dictionary


_PEAK_CLASSES = {
    DRTPeakShape.SKEW_NORMAL: DRTPeak,
    DRTPeakShape.GAUSSIAN: DRTPeakGaussian,
    DRTPeakShape.ZARC: DRTPeakZARC,
    DRTPeakShape.HAVRILIAK_NEGAMI: DRTPeakHavriliakNegami,
}


@dataclass(frozen=True)
class DRTPeakSettings:
    """
    The settings used when analyzing the peaks of a DRT.

    Parameters
    ----------
    shape: DRTPeakShape
        The function that is fitted to each peak.
        The area of a Gaussian, ZARC, or Havriliak-Negami peak is the resistance of the corresponding process.

    num_peaks: int
        The number of peaks to analyze.
        The tallest local maxima are used first and shoulders are added if there are fewer local maxima than requested (e.g., overlapping peaks).
        Zero means that all detected peaks are analyzed.
        Ignored if peak_positions is not empty.

    peak_positions: List[float]
        The approximate time constants (in seconds) of the peaks to analyze.

    disallow_skew: bool
        If true, then normal distributions are used instead of skew normal distributions.
        Only applicable to the skew normal shape.

    global_optimization: bool
        If true, then a global search is performed before the local refinement.
        This is slower but more robust when peaks overlap.

    allow_negative: bool
        If true, then negative peaks (e.g., inductive processes) are also detected and deconvoluted.
        Negative peaks have negative heights and areas.
    """

    shape: DRTPeakShape
    num_peaks: int
    peak_positions: List[float]
    disallow_skew: bool
    global_optimization: bool
    allow_negative: bool = False

    def __repr__(self) -> str:
        return f"DRTPeakSettings ({hex(id(self))})"

    def to_dict(self) -> dict:
        return {
            "version": VERSION,
            "shape": int(self.shape),
            "num_peaks": self.num_peaks,
            "peak_positions": [float(_) for _ in self.peak_positions],
            "disallow_skew": self.disallow_skew,
            "global_optimization": self.global_optimization,
            "allow_negative": self.allow_negative,
        }

    @classmethod
    def from_dict(Class, dictionary: dict) -> "DRTPeakSettings":
        assert type(dictionary) is dict
        assert "version" in dictionary
        assert dictionary["version"] <= VERSION
        assert "shape" in dictionary
        assert "num_peaks" in dictionary
        assert "peak_positions" in dictionary
        assert "disallow_skew" in dictionary
        assert "global_optimization" in dictionary

        return Class(
            shape=DRTPeakShape(dictionary["shape"]),
            num_peaks=dictionary["num_peaks"],
            peak_positions=list(dictionary["peak_positions"]),
            disallow_skew=dictionary["disallow_skew"],
            global_optimization=dictionary["global_optimization"],
            allow_negative=dictionary.get("allow_negative", False),
        )


def _peaks_to_dict(peaks: DRTPeaks) -> dict:
    return {
        "suffix": peaks.suffix,
        "peaks": [
            {f.name: float(getattr(peak, f.name)) for f in fields(peak)}
            for peak in peaks.peaks
        ],
    }


def _peaks_from_dict(
    dictionary: dict,
    time_constants: TimeConstants,
    shape: DRTPeakShape,
) -> DRTPeaks:
    Class = _PEAK_CLASSES[shape]
    return DRTPeaks(
        time_constants=time_constants,
        peaks=[Class(**kwargs) for kwargs in dictionary["peaks"]],
        suffix=dictionary["suffix"],
        shape=drt_peak_shape_to_value[shape],
    )


@dataclass
class DRTPeakAnalysis:
    """
    The results of analyzing the peaks of a DRT.

    Parameters
    ----------
    settings: DRTPeakSettings
        The settings used to perform the analysis.

    real: DRTPeaks
        The peaks of the DRT (for BHT: the DRT calculated based on the real part of the impedance data).

    imaginary: Optional[DRTPeaks]
        The peaks of the DRT calculated based on the imaginary part of the impedance data.
        Only available when the BHT method has been used.

    colors: Dict[str, List[List[float]]]
        The (RGBA, 0-255) colors assigned to the peaks of each part ("real" and "imag.").
        Peaks without an assigned color use a default color from a palette.
    """

    settings: DRTPeakSettings
    real: DRTPeaks
    imaginary: Optional[DRTPeaks] = None
    colors: Dict[str, List[List[float]]] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"DRTPeakAnalysis ({hex(id(self))})"

    def get_num_peaks(self) -> int:
        """
        Get the total number of peaks.

        Returns
        -------
        int
        """
        n: int = self.real.get_num_peaks()
        if self.imaginary is not None:
            n += self.imaginary.get_num_peaks()

        return n

    def to_dict(self) -> dict:
        return {
            "version": VERSION,
            "settings": self.settings.to_dict(),
            "real": _peaks_to_dict(self.real),
            "imaginary": (
                _peaks_to_dict(self.imaginary) if self.imaginary is not None else None
            ),
            "colors": {
                part: [[float(_) for _ in color] for color in colors]
                for part, colors in self.colors.items()
            },
        }

    @classmethod
    def from_dict(
        Class,
        dictionary: dict,
        time_constants: TimeConstants,
    ) -> "DRTPeakAnalysis":
        assert type(dictionary) is dict
        assert "version" in dictionary
        assert dictionary["version"] <= VERSION
        assert "settings" in dictionary
        assert "real" in dictionary
        assert "imaginary" in dictionary

        settings: DRTPeakSettings = DRTPeakSettings.from_dict(dictionary["settings"])
        imaginary: Optional[DRTPeaks] = None
        if dictionary["imaginary"] is not None:
            imaginary = _peaks_from_dict(
                dictionary["imaginary"], time_constants, settings.shape
            )

        return Class(
            settings=settings,
            real=_peaks_from_dict(dictionary["real"], time_constants, settings.shape),
            imaginary=imaginary,
            colors={
                part: [list(map(float, color)) for color in colors]
                for part, colors in (dictionary.get("colors") or {}).items()
            },
        )

    def get_parts(self) -> List[Tuple[str, DRTPeaks]]:
        """
        Get the analyzed parts as (label, peaks) tuples.
        The label is "real" or "imag.".

        Returns
        -------
        List[Tuple[str, DRTPeaks]]
        """
        parts: List[Tuple[str, DRTPeaks]] = [("real", self.real)]
        if self.imaginary is not None:
            parts.append(("imag.", self.imaginary))

        return parts

    def get_peak_color(self, part: str, index: int) -> List[float]:
        """
        Get the color (RGBA, 0-255) of a peak.

        Parameters
        ----------
        part: str
            "real" or "imag.".

        index: int
            The index of the peak.

        Returns
        -------
        List[float]
        """
        assert part in ("real", "imag."), part
        assert index >= 0, index
        colors: List[List[float]] = self.colors.get(part, [])
        if index < len(colors) and colors[index] is not None:
            return list(colors[index])

        offset: int = 0
        if part == "imag.":
            offset = self.real.get_num_peaks()

        return list(DRT_PEAK_PALETTE[(offset + index) % len(DRT_PEAK_PALETTE)])

    def set_peak_color(self, part: str, index: int, color: List[float]):
        """
        Set the color (RGBA, 0-255) of a peak.

        Parameters
        ----------
        part: str
            "real" or "imag.".

        index: int
            The index of the peak.

        color: List[float]
            The RGBA values (0-255).
        """
        assert part in ("real", "imag."), part
        assert index >= 0, index
        assert len(color) == 4, color
        peaks: Optional[DRTPeaks] = self.real if part == "real" else self.imaginary
        assert peaks is not None and index < peaks.get_num_peaks(), index
        colors: List[List[float]] = [
            self.get_peak_color(part, i) for i in range(0, peaks.get_num_peaks())
        ]
        colors[index] = [float(_) for _ in color]
        self.colors[part] = colors

    def get_residuals(
        self,
        drt: "DRTResult",
    ) -> Tuple[TimeConstants, Gammas, Optional[Gammas]]:
        """
        Get the difference between the DRT and the sum of the fitted peaks (i.e., gamma_DRT - sum of peaks).

        Parameters
        ----------
        drt: DRTResult
            The DRT result that was analyzed.

        Returns
        -------
        Tuple[|TimeConstants|, |Gammas|, Optional[|Gammas|]]
            The time constants, the residuals of the (real) DRT, and the residuals of the imaginary DRT (BHT only).
        """
        tau: TimeConstants = drt.time_constants
        real: Gammas = drt.real_gammas - self._get_sum(self.real, tau)
        imaginary: Optional[Gammas] = None
        if self.imaginary is not None and drt.imaginary_gammas.size > 0:
            imaginary = drt.imaginary_gammas - self._get_sum(self.imaginary, tau)

        return (tau, real, imaginary)

    @staticmethod
    def _get_sum(peaks: DRTPeaks, time_constants: TimeConstants) -> Gammas:
        if peaks.get_num_peaks() < 1:
            return full(time_constants.shape, 0.0)

        return peaks.get_gammas(num_per_decade=-1)

    def get_relative_residual(self, drt: "DRTResult", part: str = "real") -> float:
        """
        Get the relative difference between the DRT and the sum of the fitted peaks.
        The difference is calculated as the integral of abs(gamma_DRT - sum of peaks) divided by the integral of abs(gamma_DRT) with respect to ln(tau).

        Parameters
        ----------
        drt: DRTResult
            The DRT result that was analyzed.

        part: str, optional
            "real" or "imag.".

        Returns
        -------
        float
            The relative difference (0.0 to 1.0, or larger if the peaks are a poor fit). NaN if it cannot be calculated.
        """
        assert part in ("real", "imag."), part
        tau: TimeConstants
        real: Gammas
        imaginary: Optional[Gammas]
        tau, real, imaginary = self.get_residuals(drt)
        residuals: Optional[Gammas] = real if part == "real" else imaginary
        gammas: Gammas = drt.real_gammas if part == "real" else drt.imaginary_gammas
        if residuals is None or tau.size < 2:
            return nan

        x = ln(tau)
        if x[0] > x[-1]:
            x = x[::-1]
            residuals = residuals[::-1]
            gammas = gammas[::-1]

        denominator: float = float(trapezoid(np_abs(gammas), x))
        if not denominator > 0.0:
            return nan

        return float(trapezoid(np_abs(residuals), x)) / denominator

    def to_dataframe(
        self,
        columns: Optional[List[str]] = None,
        frequency: bool = False,
    ) -> DataFrame:
        """
        Get the peaks as a pandas.DataFrame object that can be used to generate, e.g., a Markdown table.

        Parameters
        ----------
        columns: Optional[List[str]], optional
            The labels to use as the column headers for the part (only included when there are imaginary peaks), the peak number, the time constants (or frequencies), the gammas, and the areas.

        frequency: bool, optional
            If true, then the positions of the peaks are given as frequencies, f = 1 / (2 * pi * tau), instead of time constants.

        Returns
        -------
        DataFrame
        """
        if columns is None:
            columns = [
                "part",
                "#",
                "f (Hz)" if frequency else "tau (s)",
                "gamma (ohm)",
                "R_peak (ohm)",
            ]

        assert isinstance(columns, list)
        assert len(columns) >= 5

        rows: List[list] = []
        label: str
        peaks: Optional[DRTPeaks]
        for label, peaks in (("real", self.real), ("imag.", self.imaginary)):
            if peaks is None:
                continue

            i: int
            for i, peak in enumerate(peaks.peaks):
                rows.append(
                    [
                        label,
                        i + 1,
                        (
                            1.0 / (2 * pi * float(peak.get_time_constant()))
                            if frequency
                            else float(peak.get_time_constant())
                        ),
                        float(peak.get_height()),
                        float(peaks.get_peak_area(i)),
                    ]
                )

        if self.imaginary is None:
            # The part is only relevant when there are two DRTs (BHT).
            return DataFrame([row[1:] for row in rows], columns=columns[1:5])

        return DataFrame(rows, columns=columns[:5])


@dataclass
class DRTResult:
    """
    An object representing the results of calculating the distribution of relaxation times in a  data set.

    Parameters
    ----------
    uuid: str
        The universally unique identifier assigned to this result.

    timestamp: float
        The Unix time (in seconds) for when the test was performed.

    time_constants: TimeConstants
        The time constants (in seconds).

    real_gammas: Gammas
        The corresponding gamma values (in ohms).

    imaginary_gammas: Gammas
        The gamma values calculated based the imaginary part of the impedance data.
        Only non-empty when the BHT method has been used.

    frequencies: Frequencies
        The frequencies of the analyzed data set.

    impedances: ComplexImpedances
        The modeled impedances.

    residuals: ComplexResiduals
        The residuals for the real and imaginary parts of the modeled impedances.

    mean_gammas: Gammas
        The mean values for gamma(tau).
        Only non-empty when the TR-RBF method has been used and the Bayesian credible intervals have been calculated.

    lower_bounds: Gammas
        The lower bound for the gamma(tau) values.
        Only non-empty when the TR-RBF method has been used and the Bayesian credible intervals have been calculated.

    upper_bounds: Gammas
        The upper bound for the gamma(tau) values.
        Only non-empty when the TR-RBF method has been used and the Bayesian credible intervals have been calculated.

    scores: Dict[str, complex]
        The scores calculated for the analyzed data set.
        Only non-empty when the BHT method has been used.

    pseudo_chisqr: float
        The calculated |pseudo chi-squared| (eq. 14 in Boukamp, 1995).

    lambda_value: float
        The regularization parameter used as part of the Tikhonov regularization.
        Only valid (i.e., positive) when the TR-NNLS or TR-RBF methods have been used.

    mask: Dict[int, bool]
        The mask that was applied to the analyzed data set.

    settings: DRTSettings
        The settings used to perform this analysis.

    peak_analysis: Optional[DRTPeakAnalysis]
        The results of analyzing the peaks of the DRT (if performed).
    """

    uuid: str
    timestamp: float
    time_constants: TimeConstants
    real_gammas: Gammas
    imaginary_gammas: Gammas
    frequencies: Frequencies
    impedances: ComplexImpedances
    residuals: ComplexResiduals
    mean_gammas: Gammas
    lower_bounds: Gammas
    upper_bounds: Gammas
    scores: Dict[str, complex]
    pseudo_chisqr: float
    lambda_value: float
    mask: Dict[int, bool]
    settings: DRTSettings
    peak_analysis: Optional[DRTPeakAnalysis] = None

    def __repr__(self) -> str:
        return f"DRTResult ({self.get_label()}, {hex(id(self))})"

    def __hash__(self) -> int:
        return int(self.uuid, 16)

    @classmethod
    def from_dict(
        Class,
        dictionary: dict,
        data: Optional[DataSet] = None,
    ) -> "DRTResult":
        """
        Create an instance from a dictionary.

        Parameters
        ----------
        dictionary: dict
            The dictionary to turn into a DRTResult object.

        data: Optional[DataSet], optional
            The DataSet object that this result is for.

        Returns
        -------
        DRTResult
        """
        assert isinstance(dictionary, dict), dictionary
        assert data is None or isinstance(data, DataSet), data
        assert "version" in dictionary
        version: int = dictionary["version"]
        del dictionary["version"]

        assert version <= VERSION, f"{version=} > {VERSION=}"
        parsers: Dict[int, Callable] = {
            1: _parse_result_v1,
            2: _parse_result_v2,
            3: _parse_result_v3,
            4: _parse_result_v4,
            5: _parse_result_v5,
        }
        assert version in parsers, f"{version=} not in {parsers.keys()=}"

        v: int
        p: Callable
        for v, p in parsers.items():
            if v < version:
                continue

            dictionary = p(dictionary)

        assert "uuid" in dictionary
        assert "timestamp" in dictionary
        assert "time_constants" in dictionary
        assert "real_gammas" in dictionary
        assert "imaginary_gammas" in dictionary
        assert "real_impedances" in dictionary
        assert "imaginary_impedances" in dictionary
        assert "frequencies" in dictionary
        assert "real_residuals" in dictionary
        assert "imaginary_residuals" in dictionary
        assert "mean_gammas" in dictionary
        assert "lower_bounds" in dictionary
        assert "upper_bounds" in dictionary
        assert "real_scores" in dictionary
        assert "imaginary_scores" in dictionary
        assert "pseudo_chisqr" in dictionary
        assert "lambda_value" in dictionary
        assert "mask" in dictionary
        assert "settings" in dictionary

        dictionary["time_constants"] = array(dictionary["time_constants"])
        dictionary["real_gammas"] = array(dictionary["real_gammas"])
        dictionary["imaginary_gammas"] = array(dictionary["imaginary_gammas"])
        dictionary["frequencies"] = array(dictionary["frequencies"])
        dictionary["mean_gammas"] = array(dictionary["mean_gammas"])
        dictionary["lower_bounds"] = array(dictionary["lower_bounds"])
        dictionary["upper_bounds"] = array(dictionary["upper_bounds"])
        dictionary["settings"] = DRTSettings.from_dict(dictionary["settings"])
        if dictionary.get("peak_analysis") is not None:
            dictionary["peak_analysis"] = DRTPeakAnalysis.from_dict(
                dictionary["peak_analysis"],
                dictionary["time_constants"],
            )
        else:
            dictionary["peak_analysis"] = None

        mask: Dict[str, bool] = dictionary["mask"]
        if data is not None:
            mask = {
                i: mask.get(str(i), False)
                for i in range(0, len(data.get_frequencies(masked=None)))
            }
        else:
            mask = {
                i: mask.get(str(i), False)
                for i in range(0, len(dictionary["frequencies"]))
            }
        dictionary["mask"] = mask

        dictionary["impedances"] = array(
            list(
                map(
                    lambda _: complex(*_),
                    zip(
                        dictionary["real_impedances"],
                        dictionary["imaginary_impedances"],
                    ),
                )
            )
        )
        del dictionary["real_impedances"]
        del dictionary["imaginary_impedances"]

        dictionary["residuals"] = array(
            list(
                map(
                    lambda _: complex(*_),
                    zip(
                        dictionary["real_residuals"],
                        dictionary["imaginary_residuals"],
                    ),
                )
            )
        )
        del dictionary["real_residuals"]
        del dictionary["imaginary_residuals"]

        dictionary["scores"] = {
            k: complex(
                dictionary["real_scores"][k],
                dictionary["imaginary_scores"][k],
            )
            for k in dictionary["real_scores"]
        }
        del dictionary["real_scores"]
        del dictionary["imaginary_scores"]

        if isnan(dictionary["pseudo_chisqr"]):
            dictionary["pseudo_chisqr"] = _calculate_pseudo_chisqr(
                Z_exp=data.get_impedances(),
                Z_fit=dictionary["impedances"],
            )

        return Class(**dictionary)

    def to_dict(self, session: bool = True) -> dict:
        """
        Return a dictionary that can be used to recreate an instance.

        Parameters
        ----------
        session: bool, optional
            If False, then a minimal dictionary is generated to reduce file size.

        Returns
        -------
        dict
        """
        assert type(session) is bool, session

        dictionary: dict = {
            "version": VERSION,
            "uuid": self.uuid,
            "timestamp": self.timestamp,
            "time_constants": list(self.time_constants),
            "real_gammas": list(self.real_gammas),
            "imaginary_gammas": list(self.imaginary_gammas),
            "real_impedances": list(self.impedances.real),
            "imaginary_impedances": list(self.impedances.imag),
            "frequencies": list(self.frequencies),
            "real_residuals": list(self.residuals.real),
            "imaginary_residuals": list(self.residuals.imag),
            "mean_gammas": list(self.mean_gammas),
            "lower_bounds": list(self.lower_bounds),
            "upper_bounds": list(self.upper_bounds),
            "real_scores": {k: v.real for k, v in self.scores.items()},
            "imaginary_scores": {k: v.imag for k, v in self.scores.items()},
            "pseudo_chisqr": self.pseudo_chisqr,
            "lambda_value": self.lambda_value,
            "mask": self.mask.copy(),
            "settings": self.settings.to_dict(),
            "peak_analysis": (
                self.peak_analysis.to_dict() if self.peak_analysis is not None else None
            ),
        }

        if not session:
            # This helps to reduce the file sizes of projects.
            dictionary["mask"] = {
                k: v for k, v in dictionary["mask"].items() if v is True
            }

        return dictionary

    def get_label(self) -> str:
        """
        Generate a label for the result.

        Returns
        -------
        str
        """
        method: str = drt_method_to_label[self.settings.method]
        timestamp: str = format_timestamp(self.timestamp)

        return f"{method} ({timestamp})"

    def get_frequencies(self) -> Frequencies:
        """
        Get the frequencies (in hertz) of the data set.

        Returns
        -------
        Frequencies
        """
        return self.frequencies

    def get_impedances(self) -> ComplexImpedances:
        """
        Get the complex impedance of the model.

        Returns
        -------
        ComplexImpedances
        """
        return self.impedances

    def get_time_constants(self) -> TimeConstants:
        """
        Get the time constants.

        Returns
        -------
        TimeConstants
        """
        return self.time_constants

    def get_gammas(self) -> Tuple[Gammas, Gammas]:
        """
        Get the gamma values.

        Returns
        -------
        Tuple[Gammas, Gammas]
        """
        return (
            self.real_gammas,
            self.imaginary_gammas,
        )

    def to_peaks_dataframe(
        self,
        threshold: float = 0.0,
        columns: Optional[List[str]] = None,
    ) -> DataFrame:
        """
        Get the peaks as a pandas.DataFrame object that can be used to generate, e.g., a Markdown table.

        Parameters
        ----------
        threshold: float, optional
            The minimum peak height threshold (relative to the height of the tallest peak) for a peak to be included.

        columns: Optional[List[str]], optional
            The labels to use as the column headers for real time constants, real gammas, imaginary time constants, and imaginary gammas.

        Returns
        -------
        DataFrame
        """
        if columns is None:
            if self.settings.method == DRTMethod.BHT:
                columns = [
                    "tau, real (s)",
                    "gamma, real (ohm)",
                    "tau, imag. (s)",
                    "gamma, imag. (ohm)",
                ]
            else:
                columns = [
                    "tau (s)",
                    "gamma (ohm)",
                ]

        assert isinstance(columns, list)
        if self.settings.method == DRTMethod.BHT:
            assert len(columns) >= 4
        else:
            assert len(columns) >= 2

        real_taus: TimeConstants
        real_gammas: Gammas
        imag_taus: TimeConstants
        imag_gammas: Gammas
        (real_taus, real_gammas, imag_taus, imag_gammas) = self.get_peaks(
            threshold=threshold
        )

        if self.settings.method != DRTMethod.BHT:
            dictionary: dict = {
                columns[0]: real_taus,
                columns[1]: real_gammas,
            }
        else:
            dictionary: dict = {
                columns[0]: real_taus,
                columns[1]: real_gammas,
                columns[2]: imag_taus,
                columns[3]: imag_gammas,
            }

        return DataFrame.from_dict(dictionary)

    def to_peak_analysis_dataframe(
        self,
        columns: Optional[List[str]] = None,
        frequency: bool = False,
    ) -> DataFrame:
        """
        Get the results of the peak analysis as a pandas.DataFrame object.
        An empty DataFrame is returned if the peaks have not been analyzed.

        Parameters
        ----------
        columns: Optional[List[str]], optional
            The labels to use as the column headers for the part (only included for BHT), the peak number, the time constants (or frequencies), the gammas, and the areas.

        frequency: bool, optional
            If true, then the positions of the peaks are given as frequencies instead of time constants.

        Returns
        -------
        DataFrame
        """
        if self.peak_analysis is None:
            return DataFrame()

        return self.peak_analysis.to_dataframe(columns=columns, frequency=frequency)

    def get_peaks(
        self,
        threshold: float = 0.0,
    ) -> Tuple[TimeConstants, Gammas, TimeConstants, Gammas]:
        """
        Get the time constants (in seconds) and gamma (in ohms) of peaks with magnitudes greater than the threshold.
        The threshold and the magnitudes are all relative to the magnitude of the highest peak.

        Parameters
        ----------
        threshold: float, optional
            The threshold for the relative magnitude (0.0 to 1.0).

        Returns
        -------
        Tuple[TimeConstants, Gammas, TimeConstants, Gammas]
        """
        assert issubdtype(type(threshold), floating), type(threshold)
        assert 0.0 <= threshold <= 1.0, threshold

        def filter_indices(gammas: Gammas) -> Indices:
            max_g: Gamma = max(gammas)
            indices: Indices = find_peaks(gammas)[0]
            return array(
                list(
                    filter(
                        lambda _: gammas[_] / max_g > threshold and gammas[_] > 0.0,
                        indices,
                    ),
                )
            )

        real_indices: Indices = filter_indices(self.real_gammas)
        real_taus: TimeConstants
        real_gammas: Gammas
        if real_indices.size > 0:
            real_taus = self.time_constants[real_indices]
            real_gammas = self.real_gammas[real_indices]
        else:
            real_taus = array([])
            real_gammas = array([])

        imag_indices: Indices
        if self.imaginary_gammas.size > 0:
            imag_indices = filter_indices(self.imaginary_gammas)
            imag_taus: TimeConstants
            imag_gammas: Gammas
            if imag_indices.size > 0:
                imag_taus = self.time_constants[imag_indices]
                imag_gammas = self.imaginary_gammas[imag_indices]
            else:
                imag_taus = array([])
                imag_gammas = array([])
        else:
            imag_taus = array([])
            imag_gammas = array([])

        if real_taus.size != imag_taus.size:

            def pad(
                t: TimeConstants,
                g: Gammas,
                w: int,
            ) -> Tuple[TimeConstants, Gammas]:
                tmp_taus: TimeConstants = full(w, nan, dtype=TimeConstant)
                tmp_gammas: Gammas = full(w, nan, dtype=Gamma)
                tmp_taus[: t.size] = t
                tmp_gammas[: g.size] = g
                return (
                    tmp_taus,
                    tmp_gammas,
                )

            max_size: int = max(real_taus.size, imag_taus.size)
            real_taus, real_gammas = pad(real_taus, real_gammas, max_size)
            imag_taus, imag_gammas = pad(imag_taus, imag_gammas, max_size)

        return (
            real_taus,
            real_gammas,
            imag_taus,
            imag_gammas,
        )

    def get_nyquist_data(self) -> Tuple[Impedances, Impedances]:
        """
        Get the data necessary to plot this DataSet as a Nyquist plot: the real and the negative imaginary parts of the impedances.

        Returns
        -------
        Tuple[Impedances, Impedances]
        """
        return (
            self.impedances.real,
            -self.impedances.imag,
        )

    def get_bode_data(self) -> Tuple[Frequencies, Impedances, Phases]:
        """
        Get the data necessary to plot this DataSet as a Bode plot: the frequencies, the absolute magnitudes of the impedances, and the negative phase angles/shifts of the impedances in degrees.

        Returns
        -------
        Tuple[Frequencies, Impedances, Phases]
        """
        return (
            self.frequencies,
            abs(self.impedances),
            -angle(self.impedances, deg=True),
        )

    def get_drt_data(self) -> Tuple[TimeConstants, Gammas, Gammas]:
        """
        Get the data necessary to plot this DRTResult as a DRT plot: the time constants and the corresponding gamma values.

        Returns
        -------
        Tuple[TimeConstants, Gammas, Gammas]
        """
        return (
            self.time_constants,
            self.real_gammas,
            self.imaginary_gammas,
        )

    def get_drt_credible_intervals_data(
        self,
    ) -> Tuple[TimeConstants, Gammas, Gammas, Gammas]:
        """
        Get the data necessary to plot the Bayesian credible intervals for this DRTResult: the time constants, the mean gamma values, the lower bound gamma values, and the upper bound gamma values.

        Returns
        -------
        Tuple[TimeConstants, Gammas, Gammas, Gammas]
        """
        if not self.mean_gammas.any():
            return (
                array([]),
                array([]),
                array([]),
                array([]),
            )

        return (
            self.time_constants,
            self.mean_gammas,
            self.lower_bounds,
            self.upper_bounds,
        )

    def get_residuals_data(self) -> Tuple[Frequencies, Residuals, Residuals]:
        """
        Get the data necessary to plot the relative residuals for this DRTResult: the frequencies and the relative residuals for the real and imaginary parts of the impedances in percents.

        Returns
        -------
        Tuple[Frequencies, Residuals, Residuals]
        """
        return (
            self.frequencies,
            self.residuals.real * 100,
            self.residuals.imag * 100,
        )

    def get_scores(self) -> Dict[str, complex]:
        """
        Get the scores for the data set.
        The scores are represented as complex values where the real and imaginary parts have magnitudes ranging from 0.0 to 1.0.
        A consistent impedance spectrum should score high.
        BHT method only.

        Returns
        -------
        Dict[str, complex]
        """
        return self.scores

    def to_scores_dataframe(
        self,
        columns: Optional[List[str]] = None,
        rows: Optional[List[str]] = None,
    ) -> Optional[DataFrame]:
        """
        Get the scores for the data set as a pandas.DataFrame object that can be used to generate, e.g., a Markdown table.
        BHT method only.

        Parameters
        ----------
        columns: Optional[List[str]], optional
            The labels for the column headers.

        rows: Optional[List[str]], optional
            The labels for the rows.

        Returns
        -------
        Optional[pandas.DataFrame]
        """
        if self.settings.method != DRTMethod.BHT:
            return None

        if columns is None:
            columns = [
                "Score",
                "Real (%)",
                "Imag. (%)",
            ]
        assert isinstance(columns, list), columns
        assert len(columns) == 3

        if rows is None:
            rows = [
                "Mean",
                "Residuals, 1 sigma",
                "Residuals, 2 sigma",
                "Residuals, 3 sigma",
                "Hellinger distance",
                "Jensen-Shannon distance",
            ]
        assert isinstance(rows, list), rows
        assert len(rows) == 6

        return DataFrame.from_dict(
            {
                columns[0]: rows,
                columns[1]: [
                    self.scores["mean"].real * 100,
                    self.scores["residuals_1sigma"].real * 100,
                    self.scores["residuals_2sigma"].real * 100,
                    self.scores["residuals_3sigma"].real * 100,
                    self.scores["hellinger_distance"].real * 100,
                    self.scores["jensen_shannon_distance"].real * 100,
                ],
                columns[2]: [
                    self.scores["mean"].imag * 100,
                    self.scores["residuals_1sigma"].imag * 100,
                    self.scores["residuals_2sigma"].imag * 100,
                    self.scores["residuals_3sigma"].imag * 100,
                    self.scores["hellinger_distance"].imag * 100,
                    self.scores["jensen_shannon_distance"].imag * 100,
                ],
            }
        )


DRT_PEAK_SERIES_KINDS: Dict[str, str] = {
    "peaks": "peaks",
    "sum": "sum of peaks",
    "residual": "difference",
}


class DRTPeakSeries:
    """
    A view of the results of a DRT peak analysis that can be plotted as a separate series (e.g., in the plotting tab).
    These objects are not serialized and are generated on demand based on a DRTResult that has a peak analysis.

    Parameters
    ----------
    drt: DRTResult
        The DRT result that has been analyzed.

    kind: str
        "peaks" (the individual peaks), "sum" (the sum of the peaks), or "residual" (the DRT minus the sum of the peaks).
    """

    SEPARATOR: str = ":"

    def __init__(self, drt: DRTResult, kind: str):
        assert isinstance(drt, DRTResult), drt
        assert drt.peak_analysis is not None
        assert kind in DRT_PEAK_SERIES_KINDS, kind
        self.drt: DRTResult = drt
        self.kind: str = kind
        self.uuid: str = self.generate_uuid(drt.uuid, kind)

    def __repr__(self) -> str:
        return f"DRTPeakSeries ({self.get_label()}, {hex(id(self))})"

    def __eq__(self, other) -> bool:
        return isinstance(other, DRTPeakSeries) and other.uuid == self.uuid

    def __hash__(self) -> int:
        return hash(self.uuid)

    @classmethod
    def generate_uuid(Class, uuid: str, kind: str) -> str:
        return f"{uuid}{Class.SEPARATOR}{kind}"

    @classmethod
    def parse_uuid(Class, uuid: str) -> Optional[Tuple[str, str]]:
        """
        Split a series UUID into the UUID of the DRT result and the kind of series.
        Returns None if the UUID does not refer to a DRTPeakSeries.
        """
        if Class.SEPARATOR not in uuid:
            return None

        drt_uuid: str
        kind: str
        drt_uuid, kind = uuid.rsplit(Class.SEPARATOR, 1)
        if kind not in DRT_PEAK_SERIES_KINDS:
            return None

        return (drt_uuid, kind)

    @property
    def peak_analysis(self) -> DRTPeakAnalysis:
        assert self.drt.peak_analysis is not None
        return self.drt.peak_analysis

    def get_label(self) -> str:
        return f"{self.drt.get_label()} - {DRT_PEAK_SERIES_KINDS[self.kind]}"

    def has_imaginary(self) -> bool:
        return self.peak_analysis.imaginary is not None

    def get_kind_label(self) -> str:
        return DRT_PEAK_SERIES_KINDS[self.kind]

    def get_curves(
        self,
        num_per_decade: int = 100,
    ) -> List[Tuple[str, int, TimeConstants, Gammas]]:
        """
        Get the curves to plot as (part, peak index, time constants, gammas) tuples.
        The peak index is -1 for the sum of the peaks and -2 for the difference.
        The part is "real" or "imag." (the latter only for BHT); see has_imaginary.
        """
        analysis: DRTPeakAnalysis = self.peak_analysis
        curves: List[Tuple[str, int, TimeConstants, Gammas]] = []
        if self.kind == "residual":
            tau, real, imaginary = analysis.get_residuals(self.drt)
            curves.append(("real", -2, tau, real))
            if imaginary is not None:
                curves.append(("imag.", -2, tau, imaginary))

            return curves

        part: str
        peaks: DRTPeaks
        for part, peaks in analysis.get_parts():
            if peaks.get_num_peaks() < 1:
                continue

            tau = peaks.get_time_constants(num_per_decade=num_per_decade)
            if self.kind == "sum":
                curves.append(
                    (part, -1, tau, peaks.get_gammas(num_per_decade=num_per_decade))
                )
                continue

            i: int
            for i in range(0, peaks.get_num_peaks()):
                curves.append(
                    (
                        part,
                        i,
                        tau,
                        peaks.get_gammas(
                            peak_indices=[i],
                            num_per_decade=num_per_decade,
                        ),
                    )
                )

        return curves
