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

from dataclasses import replace
from os.path import dirname
from traceback import format_exc
from typing import (
    List,
    Optional,
)
from warnings import (
    WarningMessage,
    catch_warnings,
    simplefilter,
)
from pyimpspec import (
    Circuit,
    parse_mdl,
)
from deareis.data import Project
from deareis.enums import Context
from deareis.gui import ProjectTab
from deareis.gui.file_dialog import FileDialog
from deareis.signals import Signal
from deareis.state import STATE
import deareis.signals as signals


def select_circuit_file(*args, **kwargs):
    project: Optional[Project] = STATE.get_active_project()
    project_tab: Optional[ProjectTab] = STATE.get_active_project_tab()
    if project is None or project_tab is None:
        return

    context: Optional[Context] = kwargs.get("context")
    if context not in (Context.FITTING_TAB, Context.SIMULATION_TAB):
        return

    FileDialog(
        cwd=STATE.latest_circuit_directory,
        label="Select circuit file",
        callback=lambda *a, **k: signals.emit(
            Signal.LOAD_CIRCUIT_FILE,
            *a,
            context=context,
            **k,
        ),
        extensions=[".mdl", ".*"],
        multiple=False,
    )


def load_circuit_file(*args, **kwargs):
    project: Optional[Project] = STATE.get_active_project()
    project_tab: Optional[ProjectTab] = STATE.get_active_project_tab()
    if project is None or project_tab is None:
        return

    context: Optional[Context] = kwargs.get("context")
    if context not in (Context.FITTING_TAB, Context.SIMULATION_TAB):
        return

    paths: List[str] = kwargs.get("paths")
    if not paths:
        return

    path: str = paths[0]
    STATE.latest_circuit_directory = dirname(path)

    circuit: Circuit
    caught_warnings: List[WarningMessage]
    try:
        with catch_warnings(record=True) as caught_warnings:
            simplefilter("always")
            circuit = parse_mdl(path)
    except Exception:
        signals.emit(
            Signal.SHOW_ERROR_MESSAGE,
            traceback=f"{format_exc()}\nThe exception above was encountered while parsing '{path}'.",
            message="""
Encountered an error while parsing the circuit file. The file might be malformed, corrupted, or contain elements that are not supported by this version of DearEIS.
            """.strip(),
        )
        return

    # The settings are applied via signals so that the circuit, including
    # its parameter values, limits, and labels, is retained as is.
    if context == Context.FITTING_TAB:
        signals.emit(
            Signal.APPLY_FIT_SETTINGS,
            settings=replace(project_tab.get_fit_settings(), cdc=circuit.serialize()),
        )
    else:
        signals.emit(
            Signal.APPLY_SIMULATION_SETTINGS,
            settings=replace(
                project_tab.get_simulation_settings(),
                cdc=circuit.serialize(),
            ),
        )

    if caught_warnings:
        signals.emit(
            Signal.SHOW_ERROR_MESSAGE,
            traceback="\n".join(map(lambda _: str(_.message), caught_warnings)),
            message=f"""
The circuit in '{path}' was loaded, but the following warning(s) were encountered.
            """.strip(),
        )
