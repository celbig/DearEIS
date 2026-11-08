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

from dataclasses import dataclass
from math import (
    exp as _exp,
    isfinite as _isfinite,
    log as _log,
)
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
from numpy import (
    argmax as _argmax,
    argsort as _argsort,
    log as _ln,
    median as _median,
    ndarray,
    pi as _pi,
    zeros as _zeros,
)
import pyimpspec as _pyimpspec
from pyimpspec import (
    BaseDRTPeak,
    Capacitor,
    Circuit,
    ConstantPhaseElement,
    DRTPeakHavriliakNegami,
    DRTPeakZARC,
    DRTPeaks,
    Element,
    HavriliakNegamiAlternative,
    Inductor,
    Parallel,
    Resistor,
    ZARC,
)
from pyimpspec.circuit.base import Connection
from deareis.data import (
    DRTResult,
    DataSet,
)
from deareis.data.drt import (
    DRTPeakAnalysis,
    DRTPeakSettings,
    DRTPeakShape,
)

try:
    from numpy import trapezoid as _trapezoid
except ImportError:  # NumPy < 2.0
    from numpy import trapz as _trapezoid


@dataclass(frozen=True)
class DRTProcess:
    """
    A process identified by the peak analysis of a distribution of relaxation times.

    Parameters
    ----------
    time_constant: float
        The time constant (in seconds) at the extremum of the peak.

    resistance: float
        The area (in ohms) of the peak.
        Negative for negative peaks (e.g., inductive processes).

    peak: Optional[BaseDRTPeak]
        The fitted peak, which provides additional parameters depending on its shape (e.g., the dispersion parameter n of a ZARC peak).
        None if the process is the result of merging several peaks.
    """

    time_constant: float
    resistance: float
    peak: Optional[BaseDRTPeak] = None


def _peaks_to_processes(peaks: DRTPeaks) -> List[DRTProcess]:
    processes: List[DRTProcess] = []
    i: int
    peak: BaseDRTPeak
    for i, peak in enumerate(peaks.peaks):
        R: float = float(peaks.get_peak_area(i))
        tau: float = float(peak.get_time_constant())
        if _isfinite(R) and _isfinite(tau) and R != 0.0 and tau > 0.0:
            processes.append(DRTProcess(tau, R, peak))

    return sorted(processes, key=lambda _: _.time_constant)


def _merge_negligible_processes(
    processes: List[DRTProcess],
    min_fraction: float,
) -> List[DRTProcess]:
    # Merge negligible peaks into their closest neighbor (in terms of ln(tau)) so that
    # their area is not lost.
    # Positive and negative peaks are never merged with each other.
    total: float = sum(abs(_.resistance) for _ in processes)
    merged: List[DRTProcess] = []
    for positive in (True, False):
        group: List[DRTProcess] = [
            _ for _ in processes if (_.resistance > 0.0) is positive
        ]
        while len(group) > 1:
            i: int = min(range(len(group)), key=lambda j: abs(group[j].resistance))
            if abs(group[i].resistance) >= min_fraction * total:
                break

            neighbors: List[int] = [j for j in (i - 1, i + 1) if 0 <= j < len(group)]
            j: int = min(
                neighbors,
                key=lambda k: abs(
                    _log(group[k].time_constant) - _log(group[i].time_constant)
                ),
            )
            group[j] = _merge_pair(group[j], group[i], negligible=True)
            del group[i]

        merged.extend(group)

    return sorted(merged, key=lambda _: _.time_constant)


def _merge_pair(a: DRTProcess, b: DRTProcess, negligible: bool = False) -> DRTProcess:
    R: float = a.resistance + b.resistance
    if negligible:
        # A negligible peak does not change the shape of the dominant peak, so the
        # dominant peak (and its time constant and shape parameters) is kept.
        dominant: DRTProcess = a if abs(a.resistance) >= abs(b.resistance) else b
        return DRTProcess(dominant.time_constant, R, dominant.peak)

    # The shape parameters (e.g., n) of the individual peaks do not describe the merged
    # process, so only the total resistance and the R-weighted mean of ln(tau) are kept.
    ln_tau: float = (
        abs(a.resistance) * _log(a.time_constant)
        + abs(b.resistance) * _log(b.time_constant)
    ) / (abs(a.resistance) + abs(b.resistance))
    return DRTProcess(_exp(ln_tau), R, None)


def extract_drt_processes(
    drt: DRTResult,
    num_peaks: int = 0,
    min_fraction: float = 0.01,
    allow_negative: bool = False,
) -> List[DRTProcess]:
    """
    Identify the processes in a DRT based on an analysis of its peaks.

    The peak analysis of the DRT result is used if one has been performed.
    Otherwise, the peaks are analyzed using ZARC peaks (the result is not stored in the DRT result).
    Only the DRT calculated based on the real part of the impedance is used.
    Peaks whose area is less than min_fraction of the total area are merged into their closest neighbor (positive and negative peaks are never merged with each other).
    Negative peaks (negative resistances) of a stored peak analysis are always included.

    Parameters
    ----------
    drt: DRTResult
        The DRT to analyze.

    num_peaks: int, optional
        The minimum number of peaks to analyze when the DRT result does not include a peak analysis.
        All detected peaks are analyzed and shoulders are added if there are fewer peaks than requested (e.g., overlapping peaks).

    min_fraction: float, optional
        The minimum area of a peak relative to the total area of all peaks.

    allow_negative: bool, optional
        If true, then negative peaks are also detected when the DRT result does not include a peak analysis.

    Returns
    -------
    List[DRTProcess]
        Sorted by increasing time constant (i.e., from high to low frequency).
    """
    assert isinstance(drt, DRTResult), drt
    assert num_peaks >= 0, num_peaks
    assert 0.0 <= min_fraction < 1.0, min_fraction

    if drt.peak_analysis is not None:
        return _merge_negligible_processes(
            _peaks_to_processes(drt.peak_analysis.real),
            min_fraction,
        )

    gammas_present: bool = (
        (drt.real_gammas != 0.0).any()
        if allow_negative
        else (drt.real_gammas > 0.0).any()
    )
    if drt.time_constants.size < 3 or not gammas_present:
        return []

    from deareis.api.drt import analyze_drt_peaks

    def analyze(n: int) -> List[DRTProcess]:
        analysis: DRTPeakAnalysis = analyze_drt_peaks(
            drt=drt,
            settings=DRTPeakSettings(
                shape=DRTPeakShape.ZARC,
                num_peaks=n,
                peak_positions=[],
                disallow_skew=False,
                global_optimization=False,
                allow_negative=allow_negative,
            ),
        )
        return _merge_negligible_processes(
            _peaks_to_processes(analysis.real),
            min_fraction,
        )

    # All detected peaks are analyzed first so that no process is lost (requesting
    # fewer peaks would ignore the smaller ones). Shoulders are only requested if
    # there are fewer peaks than requested.
    processes: List[DRTProcess] = analyze(0)
    if 0 < len(processes) < num_peaks:
        processes = analyze(num_peaks)

    return processes


def _merge_processes(processes: List[DRTProcess], num: int) -> List[DRTProcess]:
    # Merge the pair of adjacent processes with the same sign that are closest to each
    # other (in terms of ln(tau)) until there are only num processes left.
    # If there are no such pairs, then the smallest process is removed instead.
    processes = list(processes)
    while len(processes) > max(num, 1):
        pairs: List[int] = [
            j
            for j in range(len(processes) - 1)
            if (processes[j].resistance > 0.0) is (processes[j + 1].resistance > 0.0)
        ]
        if not pairs:
            i: int = min(range(len(processes)), key=lambda j: abs(processes[j].resistance))
            del processes[i]
            continue

        i = min(
            pairs,
            key=lambda j: _log(processes[j + 1].time_constant)
            - _log(processes[j].time_constant),
        )
        processes[i : i + 2] = [_merge_pair(processes[i], processes[i + 1])]

    return processes


def _estimate_series_resistance(
    drt: DRTResult,
    data: DataSet,
    offset: float = 0.0,
) -> Optional[float]:
    # R_inf is estimated as the (robust) average difference between the real part of
    # the experimental impedance and the real part of the impedance that corresponds
    # to the DRT. This is consistent with the DRT instead of relying on a single
    # (possibly noisy) high-frequency intercept.
    # The offset is subtracted from the estimate (e.g., the resistances of (RL) blocks,
    # which contribute to the high-frequency resistance).
    f: ndarray = data.get_frequencies()
    Z: ndarray = data.get_impedances()
    if f.size == 0:
        return None

    order: ndarray = _argsort(drt.time_constants)
    tau: ndarray = drt.time_constants[order]
    gamma: ndarray = drt.real_gammas[order]
    x: ndarray = _ln(tau)

    Z_drt: ndarray = _zeros(f.size, dtype=complex)
    i: int
    for i, w in enumerate(2 * _pi * f):
        Z_drt[i] = _trapezoid(gamma / (1 + 1j * w * tau), x)

    R_inf: float = float(_median(Z.real - Z_drt.real)) - offset
    if not _isfinite(R_inf):
        return None

    return max(R_inf, 0.0)


def _estimate_series_inductance(data: DataSet) -> Optional[float]:
    f: ndarray = data.get_frequencies()
    Z: ndarray = data.get_impedances()
    if f.size == 0:
        return None

    i: int = int(_argmax(f))
    if Z[i].imag <= 0.0:
        return None

    return float(Z[i].imag / (2 * _pi * f[i]))


class _Updater:
    def __init__(self, circuit: Circuit):
        self.messages: List[str] = []
        self.names: Dict[Element, str] = {
            element: f"{element.get_symbol()}_{ident}"
            for element, ident in circuit.generate_element_identifiers(
                running=False
            ).items()
        }

    def name(self, element: Element) -> str:
        return self.names.get(element, element.get_symbol())

    def set(self, element: Element, key: str, value: float):
        if not _isfinite(value):
            return

        name: str = f"{self.name(element)}.{key}"
        if element.is_fixed(key):
            self.messages.append(f"{name} is fixed and was not changed.")
            return

        lower: float = element.get_lower_limit(key)
        upper: float = element.get_upper_limit(key)
        clamped: float = min(max(value, lower), upper)
        if clamped != value:
            self.messages.append(
                f"{name} was restricted by its limits ({value:.3g} -> {clamped:.3g})."
            )

        element.set_values(**{key: clamped})


# A relaxation block is a single element or a parallel connection with a single
# time constant that can be initialized from the resistance and the time constant of
# a process in the DRT.
# An inductive block is a resistor in parallel with an inductor, (RL), which
# corresponds to a negative peak in the DRT: R*j*w*tau / (1 + j*w*tau) = R - R / (1 + j*w*tau)
# where tau = L / R.
_RelaxationBlock = Union[Element, Parallel]


def _is_inductive_block(item: Union[Element, Connection]) -> bool:
    if not isinstance(item, Parallel):
        return False

    children: List[Union[Element, Connection]] = list(item)

    return (
        len(children) == 2
        and all(isinstance(_, Element) for _ in children)
        and sorted(type(_).__name__ for _ in children) == ["Inductor", "Resistor"]
    )


def _as_relaxation_block(item: Union[Element, Connection]) -> Optional[_RelaxationBlock]:
    if isinstance(item, (ZARC, HavriliakNegamiAlternative)):
        return item

    if not isinstance(item, Parallel):
        return None

    children: List[Union[Element, Connection]] = list(item)
    if len(children) != 2 or not all(isinstance(_, Element) for _ in children):
        return None

    resistors: List[Element] = [_ for _ in children if type(_) is Resistor]
    capacitive: List[Element] = [
        _ for _ in children if type(_) in (Capacitor, ConstantPhaseElement)
    ]
    if len(resistors) != 1 or len(capacitive) != 1:
        return None

    return item


def _allows_negative(block: _RelaxationBlock) -> bool:
    # Whether the limits of all of the parameters that are initialized with a
    # resistance (or a quantity that has the same sign) allow negative values.
    if isinstance(block, (ZARC, HavriliakNegamiAlternative)):
        return block.get_lower_limit("R") < 0.0

    children: List[Element] = list(block)
    capacitive: Element = next(_ for _ in children if type(_) is not Resistor)
    resistor: Element = next(_ for _ in children if type(_) is Resistor)
    key: str = "C" if type(capacitive) is Capacitor else "Y"

    return resistor.get_lower_limit("R") < 0.0 and capacitive.get_lower_limit(key) < 0.0


def _match_processes(
    blocks: List[_RelaxationBlock],
    processes: List[DRTProcess],
) -> List[Tuple[_RelaxationBlock, DRTProcess]]:
    # Both the blocks and the processes are ordered. Negative processes can only be
    # matched to blocks that allow negative values. The order is preserved and as many
    # processes as possible are matched (longest common subsequence-like alignment).
    n: int = len(blocks)
    m: int = len(processes)
    allowed: List[List[bool]] = [
        [processes[i].resistance > 0.0 or _allows_negative(blocks[j]) for j in range(n)]
        for i in range(m)
    ]
    best: List[List[int]] = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            best[i][j] = max(best[i + 1][j], best[i][j + 1])
            if allowed[i][j]:
                best[i][j] = max(best[i][j], 1 + best[i + 1][j + 1])

    pairs: List[Tuple[_RelaxationBlock, DRTProcess]] = []
    i = j = 0
    while i < m and j < n:
        if allowed[i][j] and best[i][j] == 1 + best[i + 1][j + 1]:
            pairs.append((blocks[j], processes[i]))
            i += 1
            j += 1
        elif best[i][j] == best[i][j + 1]:
            j += 1
        else:
            i += 1

    return pairs


def _initialize_relaxation_block(
    block: _RelaxationBlock,
    process: DRTProcess,
    updater: _Updater,
):
    tau: float = process.time_constant
    R: float = process.resistance
    # Negative peaks result in negative values (e.g., R and C of an (RC)).
    negative: bool = R < 0.0
    peak: Optional[BaseDRTPeak] = process.peak
    # The dispersion parameter of a ZARC peak is the n of a Zarc element or of the CPE
    # in an (RQ) block.
    n: Optional[float] = float(peak.n) if isinstance(peak, DRTPeakZARC) else None

    if isinstance(block, HavriliakNegamiAlternative):
        updater.set(block, "R", R)
        if isinstance(peak, DRTPeakHavriliakNegami):
            # Same definition of the impedance: R / (1 + (j*omega*tau)^a)^b.
            updater.set(block, "tau", float(peak.tau))
            updater.set(block, "a", float(peak.a))
            updater.set(block, "b", float(peak.b))
        else:
            updater.set(block, "tau", tau)
            if n is not None:
                # A symmetric (ZARC) peak corresponds to b = 1.
                updater.set(block, "a", n)
                updater.set(block, "b", 1.0)
        return

    if isinstance(block, ZARC):
        updater.set(block, "R", R)
        updater.set(block, "tau", tau)
        if n is not None:
            updater.set(block, "n", n)
        return

    children: List[Element] = list(block)
    resistor: Element = next(_ for _ in children if type(_) is Resistor)
    if _is_inductive_block(block):
        inductor: Element = next(_ for _ in children if type(_) is Inductor)
        updater.set(resistor, "R", abs(R))
        R = resistor.get_value("R")
        if R > 0.0:
            updater.set(inductor, "L", tau * R)
        return

    capacitive: Element = next(_ for _ in children if _ is not resistor)

    updater.set(resistor, "R", R)
    # Use the actual value (e.g., if the resistance is fixed or was restricted by its
    # limits) so that the time constant of the block is still correct.
    R = resistor.get_value("R")
    if R == 0.0 or (R < 0.0) is not negative:
        return

    if type(capacitive) is Capacitor:
        updater.set(capacitive, "C", tau / R)
    else:
        if n is not None:
            updater.set(capacitive, "n", n)
        # Use the actual value of n for the same reason as R.
        updater.set(capacitive, "Y", tau ** capacitive.get_value("n") / R)


def estimate_initial_values(
    circuit: Circuit,
    drt: DRTResult,
    data: DataSet,
    min_fraction: float = 0.01,
) -> Tuple[Circuit, List[str]]:
    """
    Estimate initial values for the parameters of a circuit based on the distribution of relaxation times (DRT) of a data set.

    The elements and connections that are connected in series at the top level of the circuit are classified as:

    - relaxation blocks: (RC), (RQ), Zarc, and Ha.
      These are initialized from the resistances (areas) and time constants of the peaks found by the peak analysis of the DRT (see extract_drt_processes).
      The dispersion parameter of a ZARC peak is also used for the n of a CPE or Zarc, and the parameters of a Havriliak-Negami peak are used for Ha.
      The relaxation blocks are matched, from left to right, to the peaks sorted by increasing time constant (i.e., from high to low frequency).
      If the DRT result has no peak analysis, then one ZARC peak per relaxation block is analyzed.
      If there are more peaks than relaxation blocks, then the closest adjacent peaks are merged.
      If there are fewer peaks than relaxation blocks, then the remaining blocks are left unchanged.
    - inductive blocks: (RL), i.e., a resistor in parallel with an inductor, which correspond to negative peaks in the DRT (R - R / (1 + j*omega*tau) with tau = L / R).
      These are initialized from the negative peaks of the peak analysis of the DRT, which are matched, from left to right, to the peaks sorted by increasing time constant.
      R is the absolute value of the area of the peak and L is equal to R multiplied by the time constant.
      They are matched to the negative peaks with the largest areas.
      Negative peaks are only available if the stored peak analysis includes them or, if there is no stored peak analysis, if the circuit has inductive blocks (in which case they are analyzed).
    - The remaining negative peaks are matched to the relaxation blocks, together with the positive peaks and sorted by increasing time constant, and result in negative values (e.g., a negative R and a negative C for an (RC)).
      Only blocks whose limits allow negative values (a negative lower limit for R and for C or Y) are matched to negative peaks.
      Negative peaks are ignored if there are no such blocks, and only peaks with the same sign are merged.
    - the first series resistance, which is initialized based on the difference between the experimental impedance and the impedance that corresponds to the DRT (minus the resistances of the inductive blocks).
    - the first series inductance, which is initialized based on the imaginary part of the impedance at the highest frequency (if positive).

    Other elements and connections are left unchanged.
    Fixed parameters are never changed and values are restricted by the limits of the parameters.

    Parameters
    ----------
    circuit: Circuit
        The circuit to initialize. This circuit is not modified.

    drt: DRTResult
        The DRT of the data set.

    data: DataSet
        The data set that the circuit will be fitted to.

    min_fraction: float, optional
        Peaks whose area is less than this fraction of the total area of all peaks are merged into their closest neighbor.

    Returns
    -------
    Tuple[Circuit, List[str]]
        A new circuit with updated initial values and a list of messages about the estimation (e.g., parts of the circuit that were not recognized).
    """
    assert isinstance(circuit, Circuit), circuit
    assert isinstance(drt, DRTResult), drt
    assert isinstance(data, _pyimpspec.DataSet), data

    circuit = _pyimpspec.parse_cdc(circuit.serialize())
    updater: _Updater = _Updater(circuit)

    top: Connection = circuit.get_connections(recursive=False)[0]
    items: List[Union[Element, Connection]] = (
        [top] if isinstance(top, Parallel) else list(top)
    )

    blocks: List[_RelaxationBlock] = []
    inductive_blocks: List[Parallel] = []
    resistors: List[Element] = []
    inductors: List[Element] = []
    item: Union[Element, Connection]
    for item in items:
        block: Optional[_RelaxationBlock] = _as_relaxation_block(item)
        if _is_inductive_block(item):
            inductive_blocks.append(item)  # type: ignore
        elif block is not None:
            blocks.append(block)
        elif type(item) is Resistor:
            resistors.append(item)
        elif type(item) is Inductor:
            inductors.append(item)
        else:
            label: str = (
                updater.name(item)
                if isinstance(item, Element)
                else item.to_string()
            )
            updater.messages.append(f"{label} was not recognized and was not changed.")

    # Without a stored peak analysis, exactly one peak per relaxation block is
    # requested, which avoids having to merge peaks afterwards.
    # Negative peaks are only analyzed if there are blocks that can represent them.
    all_processes: List[DRTProcess] = extract_drt_processes(
        drt,
        num_peaks=len(blocks) + len(inductive_blocks),
        min_fraction=min_fraction,
        allow_negative=len(inductive_blocks) > 0,
    )
    if not all_processes:
        raise ValueError("No peaks were found in the DRT!")
    elif drt.peak_analysis is None:
        updater.messages.insert(
            0,
            f"The DRT result has no peak analysis: {len(all_processes)} ZARC peaks were analyzed for the estimate (not saved).",
        )

    # The inductive blocks are matched to the negative peaks with the largest areas.
    # The remaining negative peaks are matched to the relaxation blocks together with
    # the positive peaks, which results in negative values (e.g., R and C of an (RC)).
    inductive_processes: List[DRTProcess] = []
    if inductive_blocks:
        negative_processes: List[DRTProcess] = [
            _ for _ in all_processes if _.resistance < 0.0
        ]
        inductive_processes = sorted(
            sorted(negative_processes, key=lambda _: _.resistance)[
                : len(inductive_blocks)
            ],
            key=lambda _: _.time_constant,
        )
        all_processes = [_ for _ in all_processes if _ not in inductive_processes]
        if len(inductive_processes) < len(inductive_blocks):
            updater.messages.append(
                f"The DRT has {len(inductive_processes)} negative peaks but the circuit has {len(inductive_blocks)} inductive (RL) blocks: the last {len(inductive_blocks) - len(inductive_processes)} blocks were not changed."
            )

    processes: List[DRTProcess] = all_processes
    if blocks:
        # Negative peaks can only be matched to the blocks whose limits allow negative
        # values. They are merged if there are more negative peaks than such blocks and
        # ignored if there are no such blocks.
        negative: List[DRTProcess] = [_ for _ in processes if _.resistance < 0.0]
        positive: List[DRTProcess] = [_ for _ in processes if _.resistance > 0.0]
        capable: int = sum(1 for _ in blocks if _allows_negative(_))
        if negative and capable < 1:
            updater.messages.append(
                f"The DRT has {len(negative)} negative peaks, which were ignored because none of the relaxation blocks have limits that allow negative values."
            )
            negative = []
        elif len(negative) > capable:
            updater.messages.append(
                f"The DRT has {len(negative)} negative peaks but only {capable} relaxation blocks have limits that allow negative values: the closest negative peaks were merged."
            )
            negative = _merge_processes(negative, capable)

        if len(positive) + len(negative) > len(blocks):
            updater.messages.append(
                f"The DRT has {len(positive) + len(negative)} peaks but the circuit has {len(blocks)} relaxation blocks: the closest peaks with the same sign were merged."
            )
            positive = _merge_processes(positive, max(len(blocks) - len(negative), 1))

        processes = sorted(positive + negative, key=lambda _: _.time_constant)
        pairs: List[Tuple[_RelaxationBlock, DRTProcess]] = _match_processes(
            blocks,
            processes,
        )
        if len(pairs) < len(blocks):
            updater.messages.append(
                f"The DRT has {len(pairs)} peaks that could be matched to the {len(blocks)} relaxation blocks: the other blocks were not changed."
            )

        for block, process in pairs:
            _initialize_relaxation_block(block, process, updater)
    elif not inductive_blocks:
        updater.messages.append("No relaxation blocks (e.g., (RC), (RQ), or Zarc) were found.")
    elif processes:
        updater.messages.append(
            f"{len(processes)} peaks were ignored because the circuit has no relaxation blocks (e.g., (RC), (RQ), or Zarc)."
        )

    inductive_resistance: float = 0.0
    for block, process in zip(inductive_blocks, inductive_processes):
        _initialize_relaxation_block(block, process, updater)
        inductive_resistance += next(
            _ for _ in block if type(_) is Resistor
        ).get_value("R")

    if resistors:
        # An (RL) block contributes its resistance to the high-frequency resistance.
        R_inf: Optional[float] = _estimate_series_resistance(
            drt,
            data,
            offset=inductive_resistance,
        )
        if R_inf is not None:
            updater.set(resistors[0], "R", R_inf)
        if len(resistors) > 1:
            updater.messages.append(
                "Only the first series resistance was changed: "
                + ", ".join(updater.name(_) for _ in resistors[1:])
                + " were not changed."
            )

    if inductors:
        L: Optional[float] = _estimate_series_inductance(data)
        if L is not None:
            updater.set(inductors[0], "L", L)
        else:
            updater.messages.append(
                f"{updater.name(inductors[0])} was not changed since the imaginary impedance at the highest frequency is not positive."
            )

    return (circuit, updater.messages)
