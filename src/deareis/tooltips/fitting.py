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

from types import SimpleNamespace
from deareis.tooltips.kramers_kronig import boukamp1995


fitting = SimpleNamespace(
    **{
        "cdc": """
The equivalent circuit to fit to the data. A circuit description code (CDC) can be typed in to define the equivalent circuit. Alternatively, the graphical circuit editor can be used to construct the equivalent circuit.
    """.strip(),
        "method": """
See the documentation for the 'lmfit' Python package for information about the fitting methods.

The 'Auto' setting performs parallel fits with each of the fitting methods and returns the result with the smallest chi-squared value.
    """.strip(),
        "weight": f"""
The 'Auto' setting performs parallel fits with each of the weight functions listed above and returns the results with the smallest chi-squared value.

References:
- Boukamp (1995, {boukamp1995})
    """.strip(),
        "nfev": """
The maximum number of function evaluations to perform. This setting can be used to limit the amount of time spent performing a fit. If this setting is set to zero, then no limit is applied.
    """.strip(),
        "timeout": """
The amount of time in seconds that a single fit is allowed to take before being timed out. If this values is less than one, then no time limit is imposed.
    """.strip(),
        "perform": """
Sometimes a circuit, which ostensibly should be capable of providing a good fit, does not fit well. If the starting values of the parameters have been left at their default values, then rough adjustments may be all that is required. For example, if a circuit contains two parallel RC subcircuits in series (i.e. the CDC contains '(RC)(RC)'), then setting one of the capacitances to be one order of magnitude greater than the other may be sufficient.

Setting both the fitting method and weight function to 'Auto' is recommended as the starting point. Some of the various combinations of fitting methods and weight functions may be more forgiving than others regarding starting values for a particular circuit and thus require less work on the part of the user to achieve a good fit.

The fitting is performed using the Python package called lmfit. That package provides estimates for the error of the fitted parameters whenever possible. Circumstances when it is not possible to provide such estimates include e.g. when a parameter is near its lower/upper limit if such a limit has been defined. A parameter with a large error estimate is indicative of that parameter not really affecting the fit at all. For example, an unnecessary Warburg diffusion element may have its admittance parameter, Y, set to be a very large value and thus the element contributes very little to the overall impedance of the circuit within the frequency range of the experimental data.
    """.strip(),
        "pseudo_chisqr": f"""
Pseudo chi-squared value calculated according to eq. 14.

Reference: {boukamp1995}
    """.strip(),
        "chisqr": """
The chi-squared value of the fit.
    """.strip(),
        "red_chisqr": """
The reduced chi-squared value of the fit.
    """.strip(),
        "aic": """
The Akaike information criterion.
    """.strip(),
        "bic": """
The Bayesian information criterion.
    """.strip(),
        "nfree": """
The degrees of freedom in the fit.
    """.strip(),
        "ndata": """
The number of data points in the fit.
    """.strip(),
        "element": """
An element in the equivalent circuit. Elements marked with an asterisk, *, are nested within the subcircuit of a container element (e.g., a transmission line model (Tlm)). Hovering over the name will show a tooltip that includes information about which container element and the specific subcircuit.
    """.strip(),
        "parameter": """
A parameter in the element. Hovering over the name will show a tooltip with the parameter's unit if one has been specified.
    """.strip(),
        "parameter_value": """
The fitted value of the parameter. Hovering over the value will show a tooltip that also contains the parameter's unit.

Values are highlighted in red to indicate that either the lower or the upper limit is preventing further adjustment of the value.
    """.strip(),
        "error": """
The estimated relative standard error of the fitted value. Hovering over the value will show a tooltip with the estimated absolute standard error of the fitted value.

Relative errors greater than or equal to 5 % are highlighted in orange to draw attention since further action may be required.

Relative errors greater than 100 % are highlighted in red to indicate that the chosen equivalent circuit clearly should be modified.
    """.strip(),
        "statistics": """
Some values may be highlighted to indicate a possible issue. For example, if the number of function evaluations is equal to the chosen limit, then this may result in a poor fit.
    """.strip(),
        "subcircuits": """
Show the contribution of each component that is connected in series in the circuit (e.g., R, (RC), and (RW) for R(RC)(RW)).

The impedances of components in series add up, so the contributions sum up to the impedance of the whole circuit.

The names of the elements in the table of parameters and the title bars of the nodes in the circuit preview are colored like the contributions that they belong to.

Not available when plotting admittance.
    """.strip(),
        "subcircuit_offset": """
Choose where the contributions are placed on the real axis:
- No offset: each contribution starts from the origin and the values are exact. Resistors in series are shown as single points.
- Series R offset: the sum of the resistors in series (e.g., the ohmic resistance) is added to the real part of the other contributions.
- Cumulative R (DC): the resistors in series and the zero-frequency resistances of the faster components (i.e., those with higher characteristic frequencies) are added. Each arc starts where it would start if the time constants were well separated.
- Peak-aligned: the real parts of all of the other components at the frequency of the apex of an arc are added. The apex of each arc is placed at the same real value as the fit at that frequency, even when arcs overlap.

The offsets only shift the real part, so the shapes of the contributions are preserved but they no longer sum up to the fit. Resistors in series are not shown separately when using an offset.

The height of a peak in the data may be greater than that of the corresponding arc when arcs overlap since the other components also contribute to the imaginary part.
    """.strip(),
        "impedance_subcircuits": """
Show the contribution of each component that is connected in series in the circuit (e.g., R, (RC), and (RW) for R(RC)(RW)).

The imaginary parts are exact and sum up to the imaginary part of the fit.

The real parts are cumulative: the components are sorted from the fastest to the slowest (i.e., by decreasing characteristic frequency) and each curve includes the resistors in series and the real parts of the faster components. Each step thus shows the resistance of a process and the last curve is the real part of the fit.

The names of the elements in the table of parameters and the title bars of the nodes in the circuit preview are colored like the contributions that they belong to.

Not available when plotting admittance.
    """.strip(),
        "delete": """
Delete the current fit result.
    """.strip(),
        "apply_fitted_as_initial": """
Copy the fitted circuit so that its values can be used as initial values.
    """.strip(),
        "drt_initial_values": """
The result of a DRT analysis of the current data set that is used to estimate the initial values of the circuit.

The time constant and the resistance of each process are obtained from the position and the area of the corresponding peak found by the peak analysis of the DRT (see the DRT tab). If the DRT result has no peak analysis, then ZARC peaks are analyzed for the estimate (the result is not saved). The following parts of the circuit that are connected in series at the top level are initialized:
- Relaxation blocks: (RC), (RQ), Zarc, and Ha. They are matched, from left to right, to the peaks sorted from high to low frequency. Write the circuit so that the order of the blocks corresponds to the order of the processes in terms of frequency. If there are more peaks than blocks, then the closest peaks are merged. The n of a CPE or Zarc is obtained from a ZARC peak, and the parameters of Ha from a Havriliak-Negami peak.
- Inductive blocks: (RL), i.e., a resistor in parallel with an inductor. They correspond to negative peaks, which are only included if negative peaks were found by the peak analysis of the DRT (enable "Negative peaks" in the DRT tab) or if the circuit has such a block and there is no peak analysis. R is the absolute value of the area of the peak and L is the product of R and the time constant. Any remaining negative peaks are matched to the relaxation blocks together with the positive peaks, which results in negative values (e.g., negative R and C in an (RC)); the lower limits of those parameters are removed.
- The first resistor: the difference between the real part of the impedance of the data set and the real part of the impedance of the DRT (minus the resistances of the inductive blocks).
- The first inductor: the positive imaginary impedance at the highest frequency.

Other parts of the circuit (e.g., Warburg elements and nested connections) and fixed parameters are left unchanged. Values are restricted by the limits of the parameters.
    """.strip(),
        "apply_drt_initial_values": """
Estimate the initial values of the circuit based on the selected DRT result.
    """.strip(),
    }
)
