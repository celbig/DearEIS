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

import json
from pathlib import Path
from os.path import (
    dirname,
    join,
)
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)
from unittest import TestCase
from uuid import uuid4
from pandas import DataFrame
from numpy import (
    allclose,
    angle,
    isclose,
    isnan,
    ndarray,
    pi,
)
import matplotlib
import matplotlib.pyplot
import pyimpspec
import deareis
from deareis.config import Config
from deareis.enums import drt_peak_shape_to_value

matplotlib.use("Agg")


TEST_PROJECT_PATH: str = join(dirname(__file__), "example-project-v5.json")
TEST_DATA_PATH: str = join(dirname(__file__), "data-1.idf")
TEST_MDL_PATH: str = join(dirname(__file__), "zview-supported.mdl")


class TestData(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control_data_sets: List[pyimpspec.DataSet] = pyimpspec.parse_data(
            TEST_DATA_PATH
        )
        cls.data_sets: List[deareis.DataSet] = deareis.parse_data(TEST_DATA_PATH)

    def test_return_type(self):
        self.assertEqual(type(self.control_data_sets), type(self.data_sets))
        self.assertTrue(
            all(map(lambda _: type(_) is pyimpspec.DataSet, self.control_data_sets))
        )
        self.assertTrue(all(map(lambda _: type(_) is deareis.DataSet, self.data_sets)))
        self.assertTrue(isinstance(self.data_sets[0], pyimpspec.DataSet))

    def test_return_values(self):
        self.assertEqual(len(self.control_data_sets), 1)
        self.assertEqual(len(self.control_data_sets), len(self.data_sets))

    def test_num_points(self):
        self.assertEqual(
            self.control_data_sets[0].get_num_points(),
            self.data_sets[0].get_num_points(),
        )

    def test_frequency(self):
        self.assertTrue(
            allclose(
                self.control_data_sets[0].get_frequencies(),
                self.data_sets[0].get_frequencies(),
            )
        )

    def test_impedance(self):
        self.assertTrue(
            allclose(
                self.control_data_sets[0].get_impedances(),
                self.data_sets[0].get_impedances(),
            )
        )


class TestElements(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control_elements: Dict[
            str, Type[pyimpspec.Element]
        ] = pyimpspec.get_elements()
        cls.elements: Dict[str, Type[deareis.Element]] = deareis.get_elements()

    def test_return_type(self):
        self.assertEqual(type(self.control_elements), dict)
        self.assertTrue(
            all(map(lambda _: type(_) is str, self.control_elements.keys()))
        )
        self.assertTrue(
            all(
                map(
                    lambda _: issubclass(_, pyimpspec.Element),
                    self.control_elements.values(),
                )
            )
        )
        self.assertEqual(type(self.elements), dict)
        self.assertTrue(all(map(lambda _: type(_) is str, self.elements.keys())))
        self.assertTrue(
            all(
                map(
                    lambda _: issubclass(_, deareis.Element),
                    self.elements.values(),
                )
            )
        )
        self.assertEqual(type(self.control_elements), type(self.elements))

    def test_return_values(self):
        self.assertGreater(len(self.control_elements), 0)
        self.assertEqual(len(self.control_elements), len(self.elements))
        self.assertEqual(list(self.control_elements.keys()), list(self.elements.keys()))
        self.assertEqual(
            list(self.control_elements.values()), list(self.elements.values())
        )


class TestCDC(TestCase):
    @classmethod
    def setUpClass(cls):
        cdc: str = "R{R=53.2:sol}([R{R=325.162:ct}W]C{C=1.68E-11:dl})"
        cls.control_circuit: pyimpspec.Circuit = pyimpspec.parse_cdc(cdc)
        cls.circuit: deareis.Circuit = deareis.parse_cdc(cdc)

    def test_return_type(self):
        self.assertEqual(type(self.control_circuit), type(self.circuit))

    def test_to_string(self):
        self.assertEqual(str(self.control_circuit), str(self.circuit))
        self.assertEqual(self.control_circuit.to_string(6), self.circuit.to_string(6))
        self.assertTrue(str(self.control_circuit) == str(self.circuit) == "[R([RW]C)]")
        self.assertEqual(
            self.circuit.to_string(1),
            "[R{R=5.3E+01/0.0E+00/inf:sol}([R{R=3.3E+02/0.0E+00/inf:ct}W{Y=1.0E-03/1.0E-24/inf,n=5.0E-01F/0.0E+00/1.0E+00}]C{C=1.7E-11/1.0E-24/1.0E+03:dl})]",
        )

    def test_circuit_builder(self):
        with deareis.CircuitBuilder() as builder:
            R: deareis.Resistor = deareis.Resistor(R=53.2)
            R.set_label("sol")
            builder.add(R)
            with builder.parallel() as parallel:
                with parallel.series() as series:
                    R: deareis.Resistor = deareis.Resistor(R=325.162)
                    R.set_label("ct")
                    series.add(R)
                    series.add(deareis.Warburg())
                C: deareis.Capacitor = deareis.Capacitor(C=1.68e-11)
                C.set_label("dl")
                parallel.add(C)
        self.assertEqual(builder.to_string(), self.circuit.to_string())
        with deareis.CircuitBuilder() as builder:
            builder += deareis.Resistor(R=53.2).set_label("sol")
            with builder.parallel() as parallel:
                with parallel.series() as series:
                    series += deareis.Resistor(R=325.162).set_label("ct")
                    series += deareis.Warburg()
                parallel += deareis.Capacitor(C=1.68e-11).set_label("dl")
        self.assertEqual(builder.to_string(), self.circuit.to_string())


class TestMDL(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control_circuit: pyimpspec.Circuit = pyimpspec.parse_mdl(TEST_MDL_PATH)
        cls.circuit: deareis.Circuit = deareis.parse_mdl(TEST_MDL_PATH)

    def test_return_type(self):
        self.assertEqual(type(self.control_circuit), type(self.circuit))

    def test_to_string(self):
        self.assertEqual(self.control_circuit.serialize(), self.circuit.serialize())
        self.assertEqual(str(self.circuit), "[(RR)CLQQGWsWoLsG]")

    def test_serialize(self):
        # The GUI stores circuits as serialized CDCs, which must retain the labels and fixed parameters.
        circuit: deareis.Circuit = deareis.parse_cdc(self.circuit.serialize())
        self.assertEqual(circuit.serialize(), self.circuit.serialize())
        self.assertEqual(
            [element.get_label() for element in circuit.get_elements()],
            [element.get_label() for element in self.circuit.get_elements()],
        )
        self.assertEqual(
            [element.are_fixed() for element in circuit.get_elements()],
            [element.are_fixed() for element in self.circuit.get_elements()],
        )


class TestConfig(TestCase):
    @classmethod
    def setUpClass(cls):
        tests_dir: Path = Path(__file__).parent
        cls.config_paths: List[Path] = [
            path
            for path in sorted(tests_dir.glob("*.json"), key=lambda p: p.name)
            if path.name.startswith("example-config-v")
        ]

    def test_migration_to_latest_version(self):
        path: Path
        for path in self.config_paths:
            config: Config = Config()
            self.assertTrue(config.load(str(path)), msg=path.name)


# TODO: Update tests
class TestKramersKronig(TestCase):
    @classmethod
    def setUpClass(cls):
        control_data: pyimpspec.DataSet = pyimpspec.parse_data(TEST_DATA_PATH)[0]
        cls.data: deareis.DataSet = deareis.parse_data(TEST_DATA_PATH)[0]

        suggestion_settings: deareis.KramersKronigSuggestionSettings 
        suggestion_settings = deareis.KramersKronigSuggestionSettings(
            methods=[],
            use_mean=False,
            use_ranking=False,
            use_sum=False,
            lower_limit=0,
            upper_limit=0,
            limit_delta=0,
            m1_mu_criterion=0.7,
            m1_beta=0.75,
        )

        test_settings: deareis.KramersKronigSettings
        test_settings = deareis.KramersKronigSettings(
            test=deareis.KramersKronigTest.COMPLEX_LEASTSQ,
            mode=deareis.KramersKronigMode.AUTO,
            representation=deareis.KramersKronigRepresentation.AUTO,
            add_capacitance=True,
            add_inductance=True,
            num_RC=0,
            min_log_F_ext=-1.0,
            max_log_F_ext=1.0,
            log_F_ext=0.0,
            num_F_ext_evaluations=20,
            rapid_F_ext_evaluations=True,
            cnls_method=deareis.CNLSMethod.LEASTSQ,
            max_nfev=50,
            timeout=60,
            suggestion_settings=suggestion_settings,
        )
        
        cls.control_result: deareis.KramersKronigResult = pyimpspec.perform_kramers_kronig_test(
            control_data,
            test="complex",
            admittance=None,
            add_capacitance=True,
            add_inductance=True,
            num_RC=0,
            min_log_F_ext=-1.0,
            max_log_F_ext=1.0,
            log_F_ext=0.0,
            num_F_ext_evaluations=20,
            rapid_F_ext_evaluations=True,
            cnls_method="leastsq",
            max_nfev=50,
            timeout=60,
        )
        
        cls.result: deareis.KramersKronigResult = deareis.perform_kramers_kronig_test(
            cls.data,
            settings=test_settings,
        )
        
        test_settings = deareis.KramersKronigSettings(
            test=deareis.KramersKronigTest.COMPLEX_LEASTSQ,
            mode=deareis.KramersKronigMode.EXPLORATORY,
            representation=deareis.KramersKronigRepresentation.AUTO,
            num_RC=cls.data.get_num_points(),
            add_capacitance=True,
            add_inductance=False,
            min_log_F_ext=-1.0,
            max_log_F_ext=1.0,
            log_F_ext=0.0,
            num_F_ext_evaluations=20,
            rapid_F_ext_evaluations=True,
            cnls_method=deareis.CNLSMethod.LEASTSQ,
            max_nfev=50,
            timeout=60,
            suggestion_settings=suggestion_settings,
        )
        
        cls.control_exploratory_results: List[
            pyimpspec.KramersKronigResult
        ] = pyimpspec.perform_exploratory_kramers_kronig_tests(
            cls.data,
            test="complex",
            num_RCs=None,
            mu_criterion=0.7,
            add_capacitance=True,
            add_inductance=True,
            method="leastsq",
            max_nfev=50,
        )
        
        cls.exploratory_results: List[
            deareis.KramersKronigResult
        ] = deareis.perform_exploratory_kramers_kronig_tests(
            cls.data,
            settings=test_settings,
        )

    def test_result(self):
        self.assertIsInstance(self.control_result, pyimpspec.KramersKronigResult)
        self.assertIsInstance(self.result, deareis.KramersKronigResult)

    def test_settings(self):
        self.assertEqual(self.result.settings.test, deareis.KramersKronigTest.COMPLEX_LEASTSQ)
        self.assertEqual(self.result.settings.mode, deareis.KramersKronigMode.AUTO)
        self.assertEqual(self.result.settings.representation, deareis.KramersKronigRepresentation.AUTO)
        self.assertEqual(self.result.settings.add_capacitance, True)
        self.assertEqual(self.result.settings.add_inductance, True)
        self.assertEqual(self.result.settings.cnls_method, deareis.CNLSMethod.LEASTSQ)
        self.assertEqual(self.result.settings.max_nfev, 50)

    def test_num_RC(self):
        self.assertEqual(self.control_result.num_RC, self.result.num_RC)
        # Because the data is not noisy and the test implementation is set to
        # complex (least squares).
        self.assertGreater(self.result.num_RC, self.data.get_num_points())

    def test_pseudo_chisqr(self):
        self.assertEqual(self.control_result.pseudo_chisqr, self.result.pseudo_chisqr)

    def test_impedance(self):
        data_impedance: ndarray = self.data.get_impedances()
        test_impedance: ndarray = self.result.get_impedances()
        self.assertTrue(type(data_impedance) == type(test_impedance))
        self.assertEqual(type(test_impedance), ndarray)
        self.assertEqual(len(test_impedance), len(data_impedance))
        self.assertLess(abs(sum(data_impedance.real - test_impedance.real)), 6e-2)
        self.assertLess(abs(sum(data_impedance.imag - test_impedance.imag)), 2e-1)

    def test_exploratory_results(self):
        self.assertIsInstance(self.control_exploratory_results, tuple)
        self.assertIsInstance(self.exploratory_results, tuple)

        self.assertEqual(len(self.control_exploratory_results), 2)
        self.assertEqual(len(self.exploratory_results), 2)

        self.assertIsInstance(self.control_exploratory_results[0], list)
        self.assertIsInstance(self.exploratory_results[0], list)
        self.assertEqual(len(self.control_exploratory_results[0]), len(self.exploratory_results[0]))

        self.assertIsInstance(self.control_exploratory_results[1], tuple)
        self.assertIsInstance(self.exploratory_results[1], tuple)
        self.assertEqual(len(self.control_exploratory_results[1]), 4)
        self.assertEqual(len(self.exploratory_results[1]), 4)
        self.assertIsInstance(self.control_exploratory_results[1][0], pyimpspec.KramersKronigResult)
        self.assertIsInstance(self.exploratory_results[1][0], deareis.KramersKronigResult)
        self.assertIsInstance(self.control_exploratory_results[1][1], dict)
        self.assertIsInstance(self.exploratory_results[1][1], dict)
        self.assertIsInstance(self.control_exploratory_results[1][2], int)
        self.assertIsInstance(self.exploratory_results[1][2], int)
        self.assertIsInstance(self.control_exploratory_results[1][3], int)
        self.assertIsInstance(self.exploratory_results[1][3], int)

        for i in range(0, len(self.exploratory_results[0])):
            self.assertEqual(
                self.control_exploratory_results[0][i].num_RC,
                self.exploratory_results[0][i].num_RC,
            )

    def test_to_statistics_dataframe(self):
        df: DataFrame = self.result.to_statistics_dataframe()
        self.assertIsInstance(df, DataFrame)
        markdown_lines: List[str] = df.to_markdown().strip().split("\n")[2:]
        label_lines: List[str] = [
            "Log pseudo chi-squared",
            "Number of RC elements",
            "Log Fext (extension factor for time constant range)",
            "Parallel resistance (ohm)",
            "Parallel capacitance (F)",
            "Parallel inductance (H)",
            "Mean of residuals, real (% of |Z|)",
            "Mean of residuals, imag. (% of |Z|)",
            "SD of residuals, real (% of |Z|)",
            "SD of residuals, imag. (% of |Z|)",
            "Residuals within 1 SD, real (%)",
            "Residuals within 1 SD, imag. (%)",
            "Residuals within 2 SD, real (%)",
            "Residuals within 2 SD, imag. (%)",
            "Residuals within 3 SD, real (%)",
            "Residuals within 3 SD, imag. (%)",
            "Lilliefors test p-value, real",
            "Lilliefors test p-value, imag.",
            "Shapiro-Wilk test p-value, real",
            "Shapiro-Wilk test p-value, imag.",
            "Estimated SD of Gaussian noise (% of |Z|)",
            "One-sample Kolmogorov-Smirnov test p-value, real",
            "One-sample Kolmogorov-Smirnov test p-value, imag.",
        ]
        self.assertEqual(len(markdown_lines), len(label_lines))

        md: str
        label: str
        for md, label in zip(markdown_lines, label_lines):
            self.assertTrue(label in md)


class TestFitting(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control_data: pyimpspec.DataSet = pyimpspec.parse_data(TEST_DATA_PATH)[0]
        cls.data: deareis.DataSet = deareis.parse_data(TEST_DATA_PATH)[0]
        cls.settings: deareis.FitSettings = deareis.FitSettings(
            cdc="R{R=90}(R{R=230}C{C=7.8E-7})(R{R=570}W{Y=4.25E-4})",
            method=deareis.CNLSMethod.LEAST_SQUARES,
            weight=deareis.Weight.PROPORTIONAL,
            max_nfev=200,
            timeout=60,
        )
        cls.control_result: deareis.FitResult = deareis.fit_circuit(
            cls.control_data, cls.settings
        )
        cls.result: deareis.FitResult = deareis.fit_circuit(cls.data, cls.settings)

    def test_return_type(self):
        self.assertIsInstance(self.control_result, deareis.FitResult)
        self.assertIsInstance(self.result, deareis.FitResult)

    def test_settings(self):
        res: deareis.FitResult
        for res in [self.result, self.control_result]:
            self.assertEqual(res.settings.cdc, self.settings.cdc)
            self.assertEqual(res.settings.method, self.settings.method)
            self.assertEqual(res.settings.weight, self.settings.weight)
            self.assertEqual(res.settings.max_nfev, self.settings.max_nfev)

    def test_statistics(self):
        self.assertEqual(self.control_result.pseudo_chisqr, self.result.pseudo_chisqr)

    def test_fitted_parameters(self):
        self.assertAlmostEqual(
            deareis.FittedParameter.from_dict(
                self.result.parameters["R_1"]["R"].to_dict()
            ).get_relative_error(),
            3.7e-4,
            delta=1e-3,
        )
        expected: Dict[str, Dict[str, Tuple[float, float, bool, str]]] = {
            "R_1": {
                "R": (1.00e2, 2e0, False, "ohm"),
            },
            "R_2": {
                "R": (2.00e2, 2e0, False, "ohm"),
            },
            "C_1": {
                "C": (8.00e-7, 1e-8, False, "F"),
            },
            "R_3": {
                "R": (5.00e2, 2e0, False, "ohm"),
            },
            "W_1": {
                "Y": (4.00e-4, 1e-5, False, "S*s^n"),
                "n": (0.5, 0.1, True, ""),
            },
            "C_2": {
                "C": (1e6, 1e-7, True, "F"),
            },
        }
        for element_label, parameters in self.result.parameters.items():
            self.assertIsInstance(element_label, str)
            self.assertTrue("_" in element_label)
            self.assertIsInstance(parameters, dict)
            for parameter_label, param in parameters.items():
                self.assertIsInstance(parameter_label, str)
                self.assertIsInstance(param, deareis.FittedParameter)
                value: float
                delta: float
                fixed: bool
                unit: str
                value, delta, fixed, unit = expected[element_label][parameter_label]
                self.assertAlmostEqual(param.get_value(), value, delta=delta)
                self.assertEqual(param.is_fixed(), fixed)
                self.assertEqual(param.get_unit(), unit)
                if not isnan(param.get_error()):
                    self.assertEqual(
                        param.get_relative_error(),
                        param.get_error() / param.get_value(),
                    )

    def test_impedance(self):
        data_impedance: ndarray = self.data.get_impedances()
        fit_impedance: ndarray = self.result.get_impedances()
        self.assertTrue(type(data_impedance) == type(fit_impedance))
        self.assertEqual(type(fit_impedance), ndarray)
        self.assertEqual(len(fit_impedance), len(data_impedance))
        self.assertLess(abs(sum(data_impedance.real - fit_impedance.real)), 2)
        self.assertLess(abs(sum(data_impedance.imag - fit_impedance.imag)), 2)

    def test_markdown(self):
        markdown: str = self.result.to_parameters_dataframe().to_markdown()
        lines: List[str] = markdown.split("\n")
        self.assertTrue(lines[0].startswith("|    | Element   | Parameter   |"))
        self.assertTrue(
            lines[0].endswith("Value |   Std. err. (%) | Unit   | Fixed   |"),
            msg=f"{lines[0]=}",
        )
        self.assertTrue(
            lines[1].startswith("|---:|:----------|:------------|"),
            msg=f"{lines[1]=}",
        )
        self.assertTrue(
            lines[1].endswith(":|----------------:|:-------|:--------|"),
            msg=f"{lines[1]=}",
        )
        self.assertTrue(
            lines[2].startswith("|  0 | R_1       | R           |"),
            msg=f"{lines[2]=}",
        )
        self.assertTrue(
            lines[2].endswith("| ohm    | No      |"),
            msg=f"{lines[2]=}",
        )
        self.assertTrue(
            lines[3].startswith("|  1 | R_2       | R           |"),
            msg=f"{lines[3]=}",
        )
        self.assertTrue(
            lines[3].endswith("| ohm    | No      |"),
            msg=f"{lines[3]=}",
        )
        self.assertTrue(
            lines[4].startswith("|  2 | C_1       | C           |"),
            msg=f"{lines[4]=}",
        )
        self.assertTrue(
            lines[4].endswith("| F      | No      |"),
            msg=f"{lines[4]=}",
        )
        self.assertTrue(
            lines[5].startswith("|  3 | R_3       | R           |"),
            msg=f"{lines[5]=}",
        )
        self.assertTrue(
            lines[5].endswith("| ohm    | No      |"),
            msg=f"{lines[5]=}",
        )
        self.assertTrue(
            lines[6].startswith("|  4 | W_1       | Y           |"),
            msg=f"{lines[6]=}",
        )
        self.assertTrue(
            lines[6].endswith("| S*s^n  | No      |"),
            msg=f"{lines[6]=}",
        )

    def test_latex(self):
        latex: str = (
            self.result.to_parameters_dataframe()
            .style.format(precision=1)
            .format_index(axis=1, escape="latex")
            .to_latex(hrules=True)
        )
        lines: List[str] = latex.split("\n")
        self.assertEqual(lines[0], r"\begin{tabular}{lllrrll}")
        self.assertEqual(lines[1], r"\toprule")
        self.assertEqual(
            lines[2],
            r" & Element & Parameter & Value & Std. err. (\%) & Unit & Fixed \\",
        )
        self.assertEqual(lines[3], r"\midrule")
        self.assertTrue(lines[4].startswith(r"0 & R_1 & R & "))
        self.assertTrue(lines[4].endswith(r" & ohm & No \\"))
        self.assertTrue(lines[5].startswith(r"1 & R_2 & R & "))
        self.assertTrue(lines[5].endswith(r" & ohm & No \\"))
        self.assertTrue(lines[6].startswith(r"2 & C_1 & C & "))
        self.assertTrue(lines[6].endswith(r" & F & No \\"))
        self.assertTrue(lines[7].startswith(r"3 & R_3 & R & "))
        self.assertTrue(lines[7].endswith(r" & ohm & No \\"))
        self.assertTrue(lines[8].startswith(r"4 & W_1 & Y & "))
        self.assertTrue(lines[8].endswith(r" & S*s^n & No \\"))
        self.assertEqual(lines[9], r"5 & W_1 & n & 0.5 & nan &  & Yes \\")
        self.assertEqual(lines[10], r"\bottomrule")
        self.assertEqual(lines[11], r"\end{tabular}")

    def test_to_statistics_dataframe(self):
        df: DataFrame = self.result.to_statistics_dataframe()
        self.assertIsInstance(df, DataFrame)
        markdown: str = df.to_markdown()
        self.assertTrue("Log pseudo chi-squared" in markdown)
        self.assertTrue("Log chi-squared" in markdown)
        self.assertTrue("Log chi-squared (reduced)" in markdown)
        self.assertTrue("Akaike info. criterion" in markdown)
        self.assertTrue("Bayesian info. criterion" in markdown)
        self.assertTrue("Degrees of freedom" in markdown)
        self.assertTrue("Number of data points" in markdown)
        self.assertTrue("Number of function evaluations" in markdown)
        self.assertTrue("Method" in markdown)
        self.assertTrue("Weight" in markdown)


class TestSimulation(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parameter_values: List[float] = [
            25.0,  # R_0
            100.0,  # R_1
            1.5e-6,  # C_2
        ]
        cls.settings: deareis.SimulationSettings = deareis.SimulationSettings(
            cdc=f"R{{R={cls.parameter_values[0]}}}(R{{R={cls.parameter_values[1]}}}C{{C={cls.parameter_values[2]}}})",
            min_frequency=2.5e-1,
            max_frequency=5e5,
            num_per_decade=5,
        )
        cls.simulation: deareis.SimulationResult = deareis.simulate_spectrum(
            cls.settings
        )

    def test_return_type(self):
        self.assertEqual(type(self.simulation), deareis.SimulationResult)

    def test_frequencies(self):
        self.assertEqual(
            len(self.simulation.get_frequencies()),
            len(
                pyimpspec.analysis.utility._interpolate(
                    [self.settings.min_frequency, self.settings.max_frequency],
                    self.settings.num_per_decade,
                )
            ),
        )
        frequency: ndarray = self.simulation.get_frequencies()
        self.assertEqual(type(frequency), ndarray)

    def test_impedance(self):
        impedance: ndarray = self.simulation.get_impedances()
        self.assertEqual(type(impedance), ndarray)
        frequency: ndarray = self.simulation.get_frequencies()
        num_points: int = len(frequency)
        self.assertEqual(num_points, len(impedance))

    def test_nyquist_data(self):
        nyquist_data: Tuple[ndarray, ndarray] = self.simulation.get_nyquist_data()
        self.assertEqual(type(nyquist_data), tuple)
        self.assertEqual(len(nyquist_data), 2)
        self.assertTrue(all(map(lambda _: type(_) is ndarray, nyquist_data)))
        impedance: ndarray = self.simulation.get_impedances()
        num_points: int = impedance.size
        self.assertEqual(num_points, len(nyquist_data[0]))
        self.assertEqual(num_points, len(nyquist_data[1]))
        self.assertTrue(allclose(impedance.real, nyquist_data[0]))
        self.assertTrue(allclose(-impedance.imag, nyquist_data[1]))

    def test_bode_data(self):
        bode_data: Tuple[ndarray, ndarray, ndarray] = self.simulation.get_bode_data()
        self.assertEqual(type(bode_data), tuple)
        self.assertEqual(len(bode_data), 3)
        self.assertTrue(all(map(lambda _: type(_) is ndarray, bode_data)))
        impedance: ndarray = self.simulation.get_impedances()
        num_points: int = impedance.size
        self.assertEqual(num_points, len(bode_data[0]))
        self.assertEqual(num_points, len(bode_data[1]))
        self.assertEqual(num_points, len(bode_data[2]))
        frequency: ndarray = self.simulation.get_frequencies()
        self.assertTrue(allclose(frequency, bode_data[0]))
        self.assertTrue(allclose(abs(impedance), bode_data[1]))
        self.assertTrue(allclose(-angle(impedance, deg=True), bode_data[2]))

    def test_markdown(self):
        markdown: str = self.simulation.to_dataframe().to_markdown()
        lines: List[str] = markdown.split("\n")
        self.assertTrue(lines[0].startswith("|    | Element   | Parameter   |"))
        self.assertTrue(lines[0].endswith(" |"))
        self.assertTrue(lines[1].startswith("|---:|:----------|:------------|"))
        self.assertTrue(lines[1].endswith(":|"))
        self.assertTrue(lines[2].startswith("|  0 | R_1       | R           |"))
        self.assertTrue(lines[2].endswith(" |"))
        self.assertTrue(lines[3].startswith("|  1 | R_2       | R           |"))
        self.assertTrue(lines[3].endswith(" |"))
        self.assertTrue(lines[4].startswith("|  2 | C_1       | C           |"))
        self.assertTrue(lines[4].endswith(" |"))

    def test_latex(self):
        latex: str = (
            self.simulation.to_dataframe()
            .style.format(precision=1)
            .format_index(axis=1, escape="latex")
            .to_latex(hrules=True)
        )
        lines: List[str] = latex.split("\n")
        self.assertEqual(lines[0], r"\begin{tabular}{lllr}")
        self.assertEqual(lines[1], r"\toprule")
        self.assertEqual(lines[2], r" & Element & Parameter & Value \\")
        self.assertEqual(lines[3], r"\midrule")
        self.assertTrue(lines[4].startswith(r"0 & R_1 & R & "))
        self.assertTrue(lines[4].endswith(r" \\"))
        self.assertTrue(lines[5].startswith(r"1 & R_2 & R & "))
        self.assertTrue(lines[5].endswith(r" \\"))
        self.assertTrue(lines[6].startswith(r"2 & C_1 & C & "))
        self.assertTrue(lines[6].endswith(r" \\"))
        self.assertEqual(lines[7], r"\bottomrule")
        self.assertEqual(lines[8], r"\end{tabular}")


class TestMPL(TestCase):
    def test_mpl(self):
        project: deareis.Project = deareis.Project.from_file(TEST_PROJECT_PATH)
        plots: List[deareis.PlotSettings] = project.get_plots()
        fig, axis = deareis.mpl.plot(plots[0], project)


class TestMPLSubcircuits(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data: deareis.DataSet = deareis.parse_data(TEST_DATA_PATH)[0]
        cls.result: deareis.FitResult = deareis.fit_circuit(
            cls.data,
            deareis.FitSettings(
                cdc="R{R=90}(R{R=230}C{C=7.8E-7})(R{R=570}W{Y=4.25E-4})",
                method=deareis.CNLSMethod.LEAST_SQUARES,
                weight=deareis.Weight.PROPORTIONAL,
                max_nfev=200,
                timeout=60,
            ),
        )
        cls.project: deareis.Project = deareis.Project()
        cls.project.add_data_set(cls.data)
        cls.project.add_fit(cls.data, cls.result)

    def create_settings(self, plot_type: deareis.PlotType) -> deareis.PlotSettings:
        settings: deareis.PlotSettings = deareis.PlotSettings(
            "Plot",
            plot_type,
            [],
            {},
            {},
            {},
            {},
            {},
            uuid4().hex,
        )
        settings.add_series(self.result)
        return settings

    def get_labels(self, axes) -> List[str]:
        return [_.get_label() for _ in list(axes[0].lines) + list(axes[0].collections)]

    def tearDown(self):
        matplotlib.pyplot.close("all")

    def test_default(self):
        settings = self.create_settings(deareis.PlotType.NYQUIST_IMPEDANCE)
        _, axes = deareis.mpl.plot(settings, self.project)
        labels: List[str] = self.get_labels(axes)
        for label in ("R_1", "(R_2C_1)", "(R_3W_1)"):
            self.assertNotIn(label, labels)

    def test_subcircuits(self):
        settings = self.create_settings(deareis.PlotType.NYQUIST_IMPEDANCE)
        _, axes = deareis.mpl.plot(settings, self.project, subcircuits=True)
        labels: List[str] = self.get_labels(axes)
        for label in ("R_1", "(R_2C_1)", "(R_3W_1)"):
            self.assertIn(label, labels)

    def test_series_resistance_offset(self):
        settings = self.create_settings(deareis.PlotType.NYQUIST_IMPEDANCE)
        _, axes = deareis.mpl.plot(
            settings,
            self.project,
            subcircuits=True,
            subcircuit_offset="series_resistance",
        )
        labels: List[str] = self.get_labels(axes)
        self.assertNotIn("R_1", labels)
        self.assertIn("(R_2C_1)", labels)
        self.assertIn("(R_3W_1)", labels)

    def test_other_offsets(self):
        for offset in ("cumulative_dc", "peak"):
            settings = self.create_settings(deareis.PlotType.NYQUIST_IMPEDANCE)
            _, axes = deareis.mpl.plot(
                settings,
                self.project,
                subcircuits=True,
                subcircuit_offset=offset,
            )
            labels: List[str] = self.get_labels(axes)
            self.assertNotIn("R_1", labels)
            self.assertIn("(R_2C_1)", labels)
            self.assertIn("(R_3W_1)", labels)
            matplotlib.pyplot.close("all")

    def test_subcircuit_color_indices(self):
        from deareis.utility import get_subcircuit_color_indices

        circuit = self.result.circuit
        names = {
            circuit.get_element_name(element): index
            for element, index in get_subcircuit_color_indices(circuit).items()
        }
        self.assertEqual(names, {"R_1": 0, "R_2": 1, "C_1": 1, "R_3": 2, "W_1": 2})
        names = {
            circuit.get_element_name(element): index
            for element, index in get_subcircuit_color_indices(circuit, offset="peak").items()
        }
        self.assertEqual(names, {"R_2": 1, "C_1": 1, "R_3": 2, "W_1": 2})

    def test_other_plot_types(self):
        for plot_type in (
            deareis.PlotType.NYQUIST_ADMITTANCE,
            deareis.PlotType.BODE_IMPEDANCE_MAGNITUDE,
        ):
            settings = self.create_settings(plot_type)
            _, axes = deareis.mpl.plot(settings, self.project, subcircuits=True)
            self.assertNotIn("(R_2C_1)", self.get_labels(axes))

    def test_invalid_arguments(self):
        settings = self.create_settings(deareis.PlotType.NYQUIST_IMPEDANCE)
        with self.assertRaises(AssertionError):
            deareis.mpl.plot(settings, self.project, subcircuits=True, subcircuit_offset="invalid")

    def test_plot_subcircuits(self):
        figure, axes = deareis.mpl.plot_subcircuits(self.result)
        self.assertEqual(len(axes[0].lines) + len(axes[0].collections), 3)

    def test_plot_fit(self):
        deareis.mpl.plot_fit(self.result, self.data, subcircuits=True)


# TODO: Update tests
class TestDRT(TestCase):
    @classmethod
    def setUpClass(cls):
        control_data: pyimpspec.DataSet = pyimpspec.parse_data(TEST_DATA_PATH)[0]
        cls.data: deareis.DataSet = deareis.parse_data(TEST_DATA_PATH)[0]
        cls.control_drt: pyimpspec.DRTResult = pyimpspec.calculate_drt(control_data)
        settings: deareis.DRTSettings = deareis.DRTSettings(
            method=deareis.DRTMethod.TR_NNLS,
            mode=deareis.DRTMode.REAL,
            lambda_value=-1.0,
            rbf_type=deareis.RBFType.GAUSSIAN,
            derivative_order=1,
            rbf_shape=deareis.RBFShape.FWHM,
            shape_coeff=0.5,
            inductance=False,
            credible_intervals=False,
            timeout=60,
            num_samples=2000,
            num_attempts=10,
            maximum_symmetry=0.5,
            fit=None,
            gaussian_width=0.15,
            num_per_decade=100,
            cross_validation_method=deareis.CrossValidationMethod.MGCV,
            tr_nnls_lambda_method=deareis.TRNNLSLambdaMethod.CUSTOM,
        )
        cls.drt: deareis.DRTResult = deareis.calculate_drt(cls.data, settings)

    def test_final_lambda_value(self):
        self.assertEqual(self.control_drt.lambda_value, self.drt.lambda_value)

    def test_pseudo_chisqr(self):
        self.assertEqual(self.control_drt.pseudo_chisqr, self.drt.pseudo_chisqr)

    def test_get_time_constants(self):
        self.assertTrue(
            allclose(
                self.control_drt.get_time_constants(),
                self.drt.get_time_constants(),
            )
        )

    def test_get_gammas(self):
        self.assertEqual(self.control_drt.get_gammas().shape, (29,))
        self.assertIsInstance(self.drt.get_gammas(), tuple)
        self.assertEqual(len(self.drt.get_gammas()), 2)
        self.assertTrue(
            allclose(
                self.control_drt.get_gammas(),
                self.drt.get_gammas()[0],
            )
        )
        self.assertEqual(self.drt.get_gammas()[1].shape, (0,))

    def test_to_peaks_dataframe(self):
        df: DataFrame = self.drt.to_peaks_dataframe()
        self.assertIsInstance(df, DataFrame)
        markdown: str = df.to_markdown()
        self.assertTrue("tau (s)" in markdown)
        self.assertTrue("gamma (ohm)" in markdown)

    def test_to_scores_dataframe(self):
        df: Optional[DataFrame] = self.drt.to_scores_dataframe()
        if self.drt.settings.method != deareis.DRTMethod.BHT:
            self.assertTrue(df is None)
            return

    def test_tr_rbf(self):
        deareis.calculate_drt(
            self.data,
            settings=deareis.DRTSettings(
                method=deareis.DRTMethod.TR_RBF,
                mode=deareis.DRTMode.COMPLEX,
                lambda_value=-1.0,
                rbf_type=deareis.RBFType.GAUSSIAN,
                derivative_order=1,
                rbf_shape=deareis.RBFShape.FWHM,
                shape_coeff=0.5,
                inductance=False,
                credible_intervals=False,
                timeout=60,
                num_samples=2000,
                num_attempts=10,
                maximum_symmetry=0.5,
                fit=None,
                gaussian_width=0.15,
                num_per_decade=100,
                cross_validation_method=deareis.CrossValidationMethod.MGCV,
                tr_nnls_lambda_method=deareis.TRNNLSLambdaMethod.CUSTOM,
            )
        )

    def test_bht(self):
        deareis.calculate_drt(
            self.data,
            settings=deareis.DRTSettings(
                method=deareis.DRTMethod.BHT,
                mode=deareis.DRTMode.COMPLEX,
                lambda_value=-1.0,
                rbf_type=deareis.RBFType.GAUSSIAN,
                derivative_order=1,
                rbf_shape=deareis.RBFShape.FWHM,
                shape_coeff=0.5,
                inductance=False,
                credible_intervals=False,
                timeout=60,
                num_samples=2000,
                num_attempts=10,
                maximum_symmetry=0.5,
                fit=None,
                gaussian_width=0.15,
                num_per_decade=100,
                cross_validation_method=deareis.CrossValidationMethod.MGCV,
                tr_nnls_lambda_method=deareis.TRNNLSLambdaMethod.CUSTOM,
            )
        )

    def test_settings_without_bayes_fields(self):
        # Settings stored before the Bayesian method was added lack its fields
        from dataclasses import replace
        from deareis.config.defaults import DEFAULT_DRT_SETTINGS

        dictionary: dict = DEFAULT_DRT_SETTINGS.to_dict()
        for key in (
            "bayes_cutoff_frequency",
            "bayes_low_frequency_prior",
            "bayes_warmup",
            "bayes_samples",
        ):
            del dictionary[key]
        dictionary["version"] = 4
        settings = deareis.DRTSettings.from_dict(dictionary)
        self.assertEqual(settings, DEFAULT_DRT_SETTINGS)

        settings = replace(
            DEFAULT_DRT_SETTINGS,
            method=deareis.DRTMethod.BAYES_DRT2,
            bayes_cutoff_frequency=0.01,
            bayes_low_frequency_prior=deareis.DRTLowFrequencyPrior.NEGATIVE,
        )
        self.assertEqual(deareis.DRTSettings.from_dict(settings.to_dict()), settings)

    def test_bayes_drt2(self):
        from dataclasses import replace
        from importlib.util import find_spec
        from deareis.config.defaults import DEFAULT_DRT_SETTINGS

        if find_spec("bayes_drt2") is None or find_spec("cmdstanpy") is None:
            self.skipTest("bayes_drt2 and cmdstanpy are not installed")

        drt: deareis.DRTResult = deareis.calculate_drt(
            self.data,
            replace(
                DEFAULT_DRT_SETTINGS,
                method=deareis.DRTMethod.BAYES_DRT2,
                mode=deareis.DRTMode.COMPLEX,
                bayes_cutoff_frequency=0.01,
            ),
        )
        tau, real, imaginary = drt.get_drt_data()
        self.assertEqual(tau.shape, real.shape)
        self.assertEqual(drt.get_drt_credible_intervals_data()[0].size, 0)
        self.assertTrue(drt.get_label().startswith("Bayesian"))

    def test_mrq_fit(self):
        fit: deareis.FitResult = deareis.fit_circuit(
            data=self.data,
            settings=deareis.FitSettings(
                cdc="R(RQ)(RQ)",
                method=deareis.CNLSMethod.LEAST_SQUARES,
                weight=deareis.Weight.PROPORTIONAL,
                max_nfev=200,
                timeout=60,
            )
        )
        deareis.calculate_drt(
            self.data,
            settings=deareis.DRTSettings(
                method=deareis.DRTMethod.BHT,
                mode=deareis.DRTMode.COMPLEX,
                lambda_value=-1.0,
                rbf_type=deareis.RBFType.GAUSSIAN,
                derivative_order=1,
                rbf_shape=deareis.RBFShape.FWHM,
                shape_coeff=0.5,
                inductance=False,
                credible_intervals=False,
                timeout=60,
                num_samples=2000,
                num_attempts=10,
                maximum_symmetry=0.5,
                fit=fit,
                gaussian_width=0.15,
                num_per_decade=100,
                cross_validation_method=deareis.CrossValidationMethod.MGCV,
                tr_nnls_lambda_method=deareis.TRNNLSLambdaMethod.CUSTOM,
            )
        )


class TestInitialValues(TestCase):
    @classmethod
    def setUpClass(cls):
        from dataclasses import replace
        from numpy import logspace
        from numpy.random import default_rng
        from deareis.config.defaults import DEFAULT_DRT_SETTINGS

        cls.true_circuit: pyimpspec.Circuit = pyimpspec.parse_cdc(
            "R{R=20}(R{R=100}C{C=1e-6})(R{R=300}Q{Y=2e-3,n=0.9})L{L=1e-7}"
        )
        f: ndarray = logspace(6, -2, 81)
        Z: ndarray = cls.true_circuit.get_impedances(f)
        rng = default_rng(0)
        Z = Z * (
            1 + 0.002 * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
        )
        cls.data: deareis.DataSet = deareis.DataSet(
            frequencies=f,
            impedances=Z,
            label="Synthetic",
        )
        cls.drt: deareis.DRTResult = deareis.calculate_drt(
            cls.data,
            replace(DEFAULT_DRT_SETTINGS, method=deareis.DRTMethod.TR_RBF),
        )

    def estimate(self, cdc: str) -> Tuple[pyimpspec.Circuit, List[str]]:
        return deareis.estimate_initial_values(
            pyimpspec.parse_cdc(cdc),
            self.drt,
            self.data,
        )

    def values(self, circuit: pyimpspec.Circuit) -> List[Dict[str, float]]:
        return [_.get_values() for _ in circuit.get_elements()]

    def with_peaks(self, shape: "deareis.DRTPeakShape", num_peaks: int = 0) -> deareis.DRTResult:
        # A copy of the DRT result with a stored peak analysis.
        from copy import copy

        drt: deareis.DRTResult = copy(self.drt)
        drt.peak_analysis = deareis.analyze_drt_peaks(
            drt=drt,
            settings=deareis.DRTPeakSettings(
                shape=shape,
                num_peaks=num_peaks,
                peak_positions=[],
                disallow_skew=False,
                global_optimization=False,
            ),
        )
        return drt

    def test_processes(self):
        # Without a stored peak analysis, ZARC peaks are analyzed.
        self.assertIsNone(self.drt.peak_analysis)
        processes: List[deareis.DRTProcess] = deareis.extract_drt_processes(
            self.drt,
            num_peaks=2,
        )
        self.assertEqual(len(processes), 2)
        self.assertIsNone(self.drt.peak_analysis)
        taus: List[float] = [_.time_constant for _ in processes]
        self.assertEqual(taus, sorted(taus))
        self.assertTrue(isclose(taus[0], 1e-4, rtol=0.2))
        self.assertTrue(all(isinstance(_.peak, pyimpspec.DRTPeakZARC) for _ in processes))
        # The areas of the ZARC peaks overestimate the resistances by up to ~20 %
        # because of the broadening of the peaks by the regularization of the DRT.
        self.assertTrue(isclose(sum(_.resistance for _ in processes), 400.0, rtol=0.15))

    def test_stored_peak_analysis(self):
        drt: deareis.DRTResult = self.with_peaks(deareis.DRTPeakShape.GAUSSIAN, 2)
        processes: List[deareis.DRTProcess] = deareis.extract_drt_processes(drt)
        self.assertTrue(
            all(isinstance(_.peak, pyimpspec.DRTPeakGaussian) for _ in processes)
        )
        circuit, messages = deareis.estimate_initial_values(
            pyimpspec.parse_cdc("R(RC)(RQ)"),
            drt,
            self.data,
        )
        R0, R1, C1, R2, Q2 = self.values(circuit)
        self.assertTrue(isclose(R1["R"], processes[0].resistance))
        self.assertTrue(isclose(R2["R"], processes[1].resistance))
        self.assertFalse(any("no peak analysis" in _ for _ in messages))

    def test_return_types(self):
        circuit, messages = self.estimate("R(RC)(RQ)L")
        self.assertIsInstance(circuit, pyimpspec.Circuit)
        self.assertIsInstance(messages, list)
        self.assertTrue(all(type(_) is str for _ in messages))
        self.assertTrue(any("no peak analysis" in _ for _ in messages))

    def test_input_unchanged(self):
        original: pyimpspec.Circuit = pyimpspec.parse_cdc("R(RC)(RQ)")
        cdc: str = original.serialize()
        deareis.estimate_initial_values(original, self.drt, self.data)
        self.assertEqual(original.serialize(), cdc)

    def test_values(self):
        circuit, _ = self.estimate("R(RC)(RQ)L")
        R0, R1, C1, R2, Q2, L = self.values(circuit)
        self.assertTrue(isclose(R0["R"], 20.0, rtol=0.2))
        self.assertTrue(isclose(R1["R"], 100.0, rtol=0.25))
        self.assertTrue(isclose(R1["R"] * C1["C"], 1e-4, rtol=0.2))
        self.assertTrue(isclose(R2["R"], 300.0, rtol=0.25))
        # The n of the CPE comes from the ZARC peak.
        self.assertTrue(isclose(Q2["n"], 0.9, atol=0.05))
        self.assertTrue(isclose((R2["R"] * Q2["Y"]) ** (1 / Q2["n"]), 0.6, rtol=0.5))
        self.assertTrue(isclose(L["L"], 1e-7, rtol=0.5))

    def test_order(self):
        # The blocks follow their positions in the circuit (high to low frequency).
        circuit, _ = self.estimate("R(RQ)(RC)")
        R0, R1, Q1, R2, C2 = self.values(circuit)
        self.assertTrue(isclose(R1["R"], 100.0, rtol=0.25))
        self.assertTrue(isclose(R2["R"], 300.0, rtol=0.25))
        self.assertLess(
            (R1["R"] * Q1["Y"]) ** (1 / Q1["n"]),
            R2["R"] * C2["C"],
        )

    def test_merging(self):
        # A stored peak analysis may have more peaks than there are blocks.
        drt: deareis.DRTResult = self.with_peaks(deareis.DRTPeakShape.GAUSSIAN, 0)
        self.assertGreater(len(deareis.extract_drt_processes(drt)), 1)
        circuit, messages = deareis.estimate_initial_values(
            pyimpspec.parse_cdc("R(RC)"),
            drt,
            self.data,
        )
        R0, R1, C1 = self.values(circuit)
        self.assertTrue(isclose(R1["R"], 400.0, rtol=0.1))
        self.assertTrue(any("merged" in _ for _ in messages))

    def test_merging_without_peak_analysis(self):
        # All detected peaks are analyzed, so the smaller process is not lost.
        circuit, messages = self.estimate("R(RC)")
        R0, R1, C1 = self.values(circuit)
        self.assertTrue(isclose(R1["R"], 400.0, rtol=0.15))
        self.assertTrue(any("merged" in _ for _ in messages))

    def test_fewer_peaks_without_peak_analysis(self):
        # The DRT only has two processes: the peak that is added when three peaks are
        # requested has a negligible area, so the last block is not changed.
        cdc: str = "R(RC)(RC)(RC)"
        default: pyimpspec.Circuit = pyimpspec.parse_cdc(cdc)
        circuit, messages = self.estimate(cdc)
        values: List[Dict[str, float]] = self.values(circuit)
        self.assertNotEqual(values[1:3], self.values(default)[1:3])
        self.assertEqual(values[-2:], self.values(default)[-2:])
        self.assertTrue(any("were not changed" in _ for _ in messages))

    def test_fewer_peaks(self):
        drt: deareis.DRTResult = self.with_peaks(deareis.DRTPeakShape.GAUSSIAN, 2)
        cdc: str = "R" + "(RC)" * 8
        default: pyimpspec.Circuit = pyimpspec.parse_cdc(cdc)
        circuit, messages = deareis.estimate_initial_values(
            pyimpspec.parse_cdc(cdc),
            drt,
            self.data,
        )
        self.assertEqual(self.values(circuit)[-2:], self.values(default)[-2:])
        self.assertTrue(any("were not changed" in _ for _ in messages))

    def test_fixed_and_unrecognized(self):
        circuit, messages = self.estimate("R{R=5F}(RC)(RQ)W")
        R0, R1, C1, R2, Q2, W = self.values(circuit)
        self.assertEqual(R0["R"], 5.0)
        self.assertEqual(W, pyimpspec.parse_cdc("W").get_elements()[0].get_values())
        self.assertTrue(any("R_1.R is fixed" in _ for _ in messages))
        self.assertTrue(any("W_1 was not recognized" in _ for _ in messages))

    def test_limits(self):
        circuit, messages = self.estimate("R(R{R=1/1/50}C)(RQ)")
        R0, R1, C1, R2, Q2 = self.values(circuit)
        self.assertEqual(R1["R"], 50.0)
        # The time constant is still correct.
        self.assertTrue(isclose(R1["R"] * C1["C"], 1e-4, rtol=0.2))
        self.assertTrue(any("restricted by its limits" in _ for _ in messages))

    def test_zarc(self):
        circuit, _ = self.estimate("RZarcZarc")
        R0, Z1, Z2 = self.values(circuit)
        self.assertTrue(isclose(Z1["tau"], 1e-4, rtol=0.2))
        self.assertTrue(isclose(Z2["R"], 300.0, rtol=0.25))
        self.assertTrue(isclose(Z2["n"], 0.9, atol=0.05))

    def test_negative_peaks(self):
        # A DRT with a positive ZARC peak and a negative peak that corresponds to an
        # inductive (RL) block, which is R - R / (1 + j*w*tau) where tau = L / R.
        from copy import copy
        from pyimpspec.analysis.drt.peak_analysis import _zarc
        from numpy import (
            log as ln,
            logspace,
        )

        true_circuit: pyimpspec.Circuit = pyimpspec.parse_cdc(
            "R{R=20}Zarc{R=100,tau=1e-4,n=0.9}(R{R=30}L{L=0.03})"
        )
        f: ndarray = logspace(6, -2, 81)
        data: deareis.DataSet = deareis.DataSet(
            frequencies=f,
            impedances=true_circuit.get_impedances(f),
            label="Inductive",
        )
        drt: deareis.DRTResult = copy(self.drt)
        u: ndarray = ln(drt.time_constants)
        drt.real_gammas = _zarc(u, 100.0, ln(1e-4), 0.9) + _zarc(
            u, -30.0, ln(1e-3), 0.9
        )
        self.assertLess(drt.real_gammas.min(), 0.0)

        # Negative peaks are only analyzed when requested
        processes = deareis.extract_drt_processes(drt, num_peaks=2)
        self.assertTrue(all(_.resistance > 0.0 for _ in processes))
        processes = deareis.extract_drt_processes(drt, num_peaks=2, allow_negative=True)
        self.assertEqual([_.resistance < 0.0 for _ in processes], [False, True])
        self.assertTrue(isclose(processes[1].resistance, -30.0, rtol=0.1))
        self.assertTrue(isclose(processes[1].time_constant, 1e-3, rtol=0.1))

        circuit, messages = deareis.estimate_initial_values(
            pyimpspec.parse_cdc("RZarc(RL)"),
            drt,
            data,
        )
        R0, Z1, RL_R, RL_L = self.values(circuit)
        self.assertTrue(isclose(Z1["R"], 100.0, rtol=0.1), Z1)
        self.assertTrue(isclose(Z1["tau"], 1e-4, rtol=0.1), Z1)
        self.assertTrue(isclose(RL_R["R"], 30.0, rtol=0.1), RL_R)
        self.assertTrue(isclose(RL_L["L"], 0.03, rtol=0.2), RL_L)
        # The resistance of the (RL) block is part of the high-frequency resistance
        self.assertTrue(isclose(R0["R"], 20.0, atol=3.0), R0)

        # Peaks with different signs are not merged
        circuit, messages = deareis.estimate_initial_values(
            pyimpspec.parse_cdc("RZarc"),
            drt,
            data,
        )
        self.assertTrue(isclose(self.values(circuit)[1]["R"], 100.0, rtol=0.1))

        # Without a stored analysis and without inductive blocks, negative peaks are not analyzed
        circuit, messages = deareis.estimate_initial_values(
            pyimpspec.parse_cdc("RZarc(RC)"),
            drt,
            data,
        )
        self.assertGreater(self.values(circuit)[2]["R"], 0.0)

        # Negative peaks of a stored peak analysis are matched to relaxation blocks and
        # result in negative values
        stored: deareis.DRTResult = copy(drt)
        stored.peak_analysis = deareis.analyze_drt_peaks(
            drt=stored,
            settings=deareis.DRTPeakSettings(
                shape=deareis.DRTPeakShape.ZARC,
                num_peaks=0,
                peak_positions=[],
                disallow_skew=False,
                global_optimization=False,
                allow_negative=True,
            ),
        )
        # The default limits do not allow negative values
        circuit, messages = deareis.estimate_initial_values(
            pyimpspec.parse_cdc("RZarc(RC)"),
            stored,
            data,
        )
        R0, Z1, RC_R, RC_C = self.values(circuit)
        self.assertTrue(isclose(Z1["R"], 100.0, rtol=0.1), Z1)
        self.assertEqual(RC_R["R"], 1000.0)
        self.assertTrue(any("limits that allow negative" in _ for _ in messages), messages)

        # Blocks with limits that allow negative values are used for negative peaks
        cdc: pyimpspec.Circuit = pyimpspec.parse_cdc("RZarc(RC)")
        resistor, capacitor = cdc.get_elements()[2:]
        resistor.set_lower_limits(R=-1e3)
        capacitor.set_lower_limits(C=-1e3)
        circuit, messages = deareis.estimate_initial_values(cdc, stored, data)
        R0, Z1, RC_R, RC_C = self.values(circuit)
        self.assertTrue(isclose(Z1["R"], 100.0, rtol=0.1), Z1)
        self.assertTrue(isclose(RC_R["R"], -30.0, rtol=0.1), RC_R)
        self.assertTrue(isclose(RC_C["C"], 1e-3 / -30.0, rtol=0.2), RC_C)
        # The time constant of the block is positive
        self.assertGreater(RC_R["R"] * RC_C["C"], 0.0)

        # The negative peak is matched to the block that allows negative values even
        # if it is not the last one
        cdc = pyimpspec.parse_cdc("R(RC)Zarc")
        cdc.get_elements()[3].set_lower_limits(R=-1e3)
        circuit, messages = deareis.estimate_initial_values(cdc, stored, data)
        R0, RC_R, RC_C, Z = self.values(circuit)
        self.assertTrue(isclose(Z["R"], -30.0, rtol=0.1), Z)
        self.assertTrue(isclose(RC_R["R"], 100.0, rtol=0.1), RC_R)

    def test_havriliak_negami(self):
        drt: deareis.DRTResult = self.with_peaks(
            deareis.DRTPeakShape.HAVRILIAK_NEGAMI, 2
        )
        peak = deareis.extract_drt_processes(drt)[1].peak
        self.assertIsInstance(peak, pyimpspec.DRTPeakHavriliakNegami)
        circuit, _ = deareis.estimate_initial_values(
            pyimpspec.parse_cdc("R(RC)Ha"),
            drt,
            self.data,
        )
        Ha = self.values(circuit)[-1]
        for key in ("tau", "a", "b"):
            expected: float = float(getattr(peak, key))
            element = circuit.get_elements()[-1]
            expected = min(
                max(expected, element.get_lower_limit(key)),
                element.get_upper_limit(key),
            )
            self.assertTrue(isclose(Ha[key], expected), key)

    def test_fit(self):
        from dataclasses import replace

        settings: deareis.FitSettings = deareis.FitSettings(
            cdc="",
            method=deareis.CNLSMethod.LEASTSQ,
            weight=deareis.Weight.BOUKAMP,
            max_nfev=1000,
            timeout=0,
        )
        circuit, _ = self.estimate("R(RC)(RQ)L")
        default_fit: deareis.FitResult = deareis.fit_circuit(
            self.data,
            replace(settings, cdc=pyimpspec.parse_cdc("R(RC)(RQ)L").serialize()),
        )
        fit: deareis.FitResult = deareis.fit_circuit(
            self.data,
            replace(settings, cdc=circuit.serialize()),
        )
        self.assertLessEqual(fit.chisqr, default_fit.chisqr)
        self.assertLess(fit.nfev, default_fit.nfev)


# TODO: Update tests
class TestDRTPeakAnalysis(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data: deareis.DataSet = deareis.parse_data(TEST_DATA_PATH)[0]
        cls.tr_nnls: deareis.DRTResult = cls.calculate(deareis.DRTMethod.TR_NNLS)
        cls.bht: deareis.DRTResult = cls.calculate(deareis.DRTMethod.BHT)

    @classmethod
    def calculate(cls, method: deareis.DRTMethod) -> deareis.DRTResult:
        return deareis.calculate_drt(
            cls.data,
            settings=deareis.DRTSettings(
                method=method,
                mode=deareis.DRTMode.COMPLEX,
                lambda_value=-1.0,
                rbf_type=deareis.RBFType.GAUSSIAN,
                derivative_order=1,
                rbf_shape=deareis.RBFShape.FWHM,
                shape_coeff=0.5,
                inductance=False,
                credible_intervals=False,
                timeout=60,
                num_samples=200,
                num_attempts=2,
                maximum_symmetry=0.5,
                fit=None,
                gaussian_width=0.15,
                num_per_decade=100,
                cross_validation_method=deareis.CrossValidationMethod.MGCV,
                tr_nnls_lambda_method=deareis.TRNNLSLambdaMethod.CUSTOM,
            ),
        )

    @staticmethod
    def settings(**kwargs) -> deareis.DRTPeakSettings:
        defaults: dict = dict(
            shape=deareis.DRTPeakShape.GAUSSIAN,
            num_peaks=0,
            peak_positions=[],
            disallow_skew=False,
            global_optimization=False,
        )
        defaults.update(kwargs)

        return deareis.DRTPeakSettings(**defaults)

    def test_shapes_match_pyimpspec(self):
        shape: deareis.DRTPeakShape
        for shape in deareis.DRTPeakShape:
            analysis: deareis.DRTPeakAnalysis = deareis.analyze_drt_peaks(
                self.tr_nnls,
                self.settings(shape=shape),
            )
            self.assertIsNone(analysis.imaginary, msg=shape)
            control: pyimpspec.DRTPeaks = pyimpspec.analyze_drt_peaks(
                self.tr_nnls.time_constants,
                self.tr_nnls.real_gammas,
                shape=drt_peak_shape_to_value[shape],
            )
            self.assertEqual(analysis.real.get_num_peaks(), control.get_num_peaks())
            self.assertGreater(analysis.real.get_num_peaks(), 0, msg=shape)
            df: DataFrame = analysis.to_dataframe()
            self.assertEqual(
                list(df.columns),
                ["#", "tau (s)", "gamma (ohm)", "R_peak (ohm)"],
            )
            self.assertTrue(
                allclose(df["R_peak (ohm)"], control.to_peaks_dataframe().iloc[:, 2]),
                msg=shape,
            )

    def test_negative_peaks(self):
        shape: deareis.DRTPeakShape
        for shape in deareis.DRTPeakShape:
            default: deareis.DRTPeakAnalysis = deareis.analyze_drt_peaks(
                self.bht,
                self.settings(shape=shape),
            )
            analysis: deareis.DRTPeakAnalysis = deareis.analyze_drt_peaks(
                self.bht,
                self.settings(shape=shape, allow_negative=True),
            )
            self.assertTrue(analysis.settings.allow_negative)
            self.assertFalse(default.settings.allow_negative)
            self.assertTrue(
                all(_ > 0.0 for _ in default.to_dataframe()["R_peak (ohm)"]),
                msg=shape,
            )
            self.assertGreater(analysis.get_num_peaks(), default.get_num_peaks(), msg=shape)
            self.assertTrue(
                any(_ < 0.0 for _ in analysis.to_dataframe()["R_peak (ohm)"]),
                msg=shape,
            )
            restored = deareis.DRTPeakSettings.from_dict(analysis.settings.to_dict())
            self.assertTrue(restored.allow_negative)

        # Settings stored by older versions lack the key
        old: dict = self.settings().to_dict()
        del old["allow_negative"]
        self.assertFalse(deareis.DRTPeakSettings.from_dict(old).allow_negative)

    def test_bht_real_and_imaginary(self):
        analysis: deareis.DRTPeakAnalysis = deareis.analyze_drt_peaks(
            self.bht,
            self.settings(shape=deareis.DRTPeakShape.ZARC),
        )
        self.assertIsNotNone(analysis.imaginary)
        self.assertGreater(analysis.real.get_num_peaks(), 0)
        self.assertGreater(analysis.imaginary.get_num_peaks(), 0)
        df: DataFrame = analysis.to_dataframe()
        self.assertEqual(df.columns[0], "part")
        self.assertEqual(
            set(df["part"]),
            {"real", "imag."},
        )
        self.assertEqual(len(df), analysis.get_num_peaks())

    def test_num_peaks(self):
        analysis: deareis.DRTPeakAnalysis = deareis.analyze_drt_peaks(
            self.tr_nnls,
            self.settings(num_peaks=2),
        )
        self.assertEqual(analysis.real.get_num_peaks(), 2)

    def test_peak_positions(self):
        positions: List[float] = [1e-4, 1e-2]
        analysis: deareis.DRTPeakAnalysis = deareis.analyze_drt_peaks(
            self.tr_nnls,
            self.settings(peak_positions=positions, num_peaks=5),
        )
        self.assertEqual(analysis.real.get_num_peaks(), len(positions))

    def test_no_peaks(self):
        # All of the gamma values are zero or negative
        drt: deareis.DRTResult = self.calculate(deareis.DRTMethod.TR_NNLS)
        drt.real_gammas = drt.real_gammas * 0.0
        shape: deareis.DRTPeakShape
        for shape in deareis.DRTPeakShape:
            analysis: deareis.DRTPeakAnalysis = deareis.analyze_drt_peaks(
                drt,
                self.settings(shape=shape),
            )
            self.assertEqual(analysis.get_num_peaks(), 0, msg=shape)
            self.assertTrue(analysis.to_dataframe().empty)

        drt.real_gammas = -abs(drt.real_gammas) - 1.0
        analysis = deareis.analyze_drt_peaks(
            drt,
            self.settings(shape=deareis.DRTPeakShape.SKEW_NORMAL),
        )
        self.assertEqual(analysis.get_num_peaks(), 0)

    def test_invalid_settings(self):
        with self.assertRaises(AssertionError):
            deareis.analyze_drt_peaks(
                self.tr_nnls,
                self.settings(peak_positions=[-1.0]),
            )

        with self.assertRaises(AssertionError):
            deareis.analyze_drt_peaks(
                self.tr_nnls,
                self.settings(num_peaks=-1),
            )

    def test_to_dict_from_dict(self):
        drt: deareis.DRTResult = self.calculate(deareis.DRTMethod.BHT)
        shape: deareis.DRTPeakShape
        for shape in deareis.DRTPeakShape:
            drt.peak_analysis = deareis.analyze_drt_peaks(
                drt,
                self.settings(shape=shape),
            )
            dictionary: dict = json.loads(json.dumps(drt.to_dict()))
            result: deareis.DRTResult = deareis.DRTResult.from_dict(
                dictionary,
                data=self.data,
            )
            self.assertIsNotNone(result.peak_analysis, msg=shape)
            self.assertEqual(result.peak_analysis.settings, drt.peak_analysis.settings)
            self.assertTrue(
                allclose(
                    result.peak_analysis.to_dataframe().iloc[:, 2:],
                    drt.peak_analysis.to_dataframe().iloc[:, 2:],
                ),
                msg=shape,
            )

    def test_old_results_have_no_peak_analysis(self):
        dictionary: dict = self.tr_nnls.to_dict()
        dictionary["version"] = 4
        del dictionary["peak_analysis"]
        dictionary["settings"]["version"] = 4
        result: deareis.DRTResult = deareis.DRTResult.from_dict(
            dictionary,
            data=self.data,
        )
        self.assertIsNone(result.peak_analysis)
        self.assertTrue(result.to_peak_analysis_dataframe().empty)

    def test_frequency_dataframe(self):
        analysis: deareis.DRTPeakAnalysis = deareis.analyze_drt_peaks(
            self.tr_nnls,
            self.settings(),
        )
        tau: DataFrame = analysis.to_dataframe()
        freq: DataFrame = analysis.to_dataframe(frequency=True)
        self.assertEqual(
            list(freq.columns),
            ["#", "f (Hz)", "gamma (ohm)", "R_peak (ohm)"],
        )
        self.assertTrue(
            allclose(freq["f (Hz)"], 1.0 / (2 * pi * tau["tau (s)"]))
        )
        self.assertTrue(allclose(freq.iloc[:, 2:], tau.iloc[:, 2:]))

    def test_colors(self):
        drt: deareis.DRTResult = self.calculate(deareis.DRTMethod.BHT)
        drt.peak_analysis = deareis.analyze_drt_peaks(drt, self.settings())
        analysis: deareis.DRTPeakAnalysis = drt.peak_analysis
        self.assertEqual(analysis.colors, {})
        num_real: int = analysis.real.get_num_peaks()
        self.assertGreater(num_real, 1)
        colors: List[List[float]] = [
            analysis.get_peak_color("real", i) for i in range(0, num_real)
        ]
        self.assertNotEqual(colors[0], colors[1])
        # The imaginary peaks continue the palette
        self.assertNotEqual(analysis.get_peak_color("imag.", 0), colors[0])

        analysis.set_peak_color("real", 1, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(analysis.get_peak_color("real", 1), [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(analysis.get_peak_color("real", 0), colors[0])

        serialized: str = json.dumps(drt.to_dict())
        result: deareis.DRTResult = deareis.DRTResult.from_dict(
            json.loads(serialized),
            data=self.data,
        )
        self.assertEqual(
            result.peak_analysis.get_peak_color("real", 1),
            [1.0, 2.0, 3.0, 4.0],
        )

        # Results saved before colors were added
        dictionary: dict = json.loads(serialized)
        del dictionary["peak_analysis"]["colors"]
        result = deareis.DRTResult.from_dict(dictionary, data=self.data)
        self.assertEqual(result.peak_analysis.colors, {})
        self.assertEqual(result.peak_analysis.get_peak_color("real", 0), colors[0])

    def test_residuals(self):
        drt: deareis.DRTResult = self.calculate(deareis.DRTMethod.TR_NNLS)
        analysis: deareis.DRTPeakAnalysis = deareis.analyze_drt_peaks(
            drt,
            self.settings(),
        )
        tau, real, imaginary = analysis.get_residuals(drt)
        self.assertIsNone(imaginary)
        self.assertEqual(tau.shape, real.shape)
        self.assertTrue(
            allclose(
                real,
                drt.real_gammas - analysis.real.get_gammas(num_per_decade=-1),
            )
        )
        value: float = analysis.get_relative_residual(drt)
        self.assertTrue(0.0 <= value < 1.0, msg=value)

        # A DRT that is exactly the sum of the fitted peaks
        drt.real_gammas = analysis.real.get_gammas(num_per_decade=-1)
        self.assertAlmostEqual(analysis.get_relative_residual(drt), 0.0)

        bht: deareis.DRTResult = self.calculate(deareis.DRTMethod.BHT)
        analysis = deareis.analyze_drt_peaks(bht, self.settings())
        tau, real, imaginary = analysis.get_residuals(bht)
        self.assertIsNotNone(imaginary)
        self.assertFalse(isnan(analysis.get_relative_residual(bht, "imag.")))

    def test_peak_series(self):
        from deareis.data import DRTPeakSeries

        drt: deareis.DRTResult = self.calculate(deareis.DRTMethod.BHT)
        drt.peak_analysis = deareis.analyze_drt_peaks(drt, self.settings())
        num_peaks: int = drt.peak_analysis.get_num_peaks()
        kind: str
        for kind in ("peaks", "sum", "residual"):
            series: DRTPeakSeries = DRTPeakSeries(drt, kind)
            self.assertEqual(
                DRTPeakSeries.parse_uuid(series.uuid),
                (drt.uuid, kind),
            )
            curves = series.get_curves()
            parts = set(_[0] for _ in curves)
            self.assertEqual(parts, {"real", "imag."}, msg=kind)
            if kind == "peaks":
                self.assertEqual(len(curves), num_peaks)
            else:
                self.assertEqual(len(curves), 2)

            for _, _, tau, gammas in curves:
                self.assertEqual(tau.shape, gammas.shape)

        self.assertIsNone(DRTPeakSeries.parse_uuid(drt.uuid))
        self.assertIsNone(DRTPeakSeries.parse_uuid(f"{drt.uuid}:foo"))


class TestZHIT(TestCase):
    @classmethod
    def setUpClass(cls):
        control_data: pyimpspec.DataSet = pyimpspec.parse_data(TEST_DATA_PATH)[0]
        data: deareis.DataSet = deareis.parse_data(TEST_DATA_PATH)[0]
        cls.control_zhit: pyimpspec.ZHITResult = pyimpspec.perform_zhit(control_data)
        settings: deareis.ZHITSettings = deareis.ZHITSettings(
            smoothing=deareis.ZHITSmoothing.MODSINC,
            num_points=3,
            polynomial_order=2,
            num_iterations=3,
            interpolation=deareis.ZHITInterpolation.MAKIMA,
            window=deareis.ZHITWindow.AUTO,
            window_center=1.5,
            window_width=3.0,
            representation=deareis.ZHITRepresentation.IMPEDANCE,
        )
        cls.zhit: deareis.ZHITResult = deareis.perform_zhit(data, settings)

    def test_get_label(self):
        self.assertTrue(self.zhit.get_label().startswith("Z-HIT ("))

    def test_pseudo_chisqr(self):
        self.assertTrue(
            isclose(
                self.control_zhit.pseudo_chisqr,
                self.zhit.pseudo_chisqr,
            )
        )

    def test_modulus(self):
        self.assertTrue(
            allclose(
                abs(self.control_zhit.get_impedances()),
                abs(self.zhit.get_impedances()),
            )
        )

    def test_phase(self):
        self.assertTrue(
            allclose(
                -angle(self.control_zhit.get_impedances()),
                -angle(self.zhit.get_impedances()),
            )
        )

    def test_final_settings(self):
        self.assertEqual(self.control_zhit.smoothing, self.zhit.smoothing)
        self.assertEqual(self.control_zhit.interpolation, self.zhit.interpolation)
        self.assertEqual(self.control_zhit.window, self.zhit.window)

    def test_to_statistics_dataframe(self):
        df: DataFrame = self.zhit.to_statistics_dataframe()
        self.assertIsInstance(df, DataFrame)
        markdown: str = df.to_markdown()
        self.assertTrue("Log pseudo chi-squared" in markdown)
        self.assertTrue("Smoothing" in markdown)
        self.assertTrue("Interpolation" in markdown)
        self.assertTrue("Window" in markdown)
