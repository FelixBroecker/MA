#!/usr/bin/env python3

import sys
import numpy as np
import time
import yaml
import math
from pyscript import *  # requirement pyscript as python package https://github.com/Leonard-Reuter/pyscript
from csf import SelectedCI
from automation import Automation
from evaluation import Evaluation


def main():

    if len(sys.argv) == 1:
        sys.exit(
            """
        Script to generate wavefunctions for selected CI calculation and run CI calculations.

        usage: main.py <infile>

        with:
            <infile> being an .yaml file with all specification on molecule and demanded calculations.
    """
        )
    input_file = sys.argv[1]
    with open(input_file, "r") as reffile:
        input_data = yaml.safe_load(reffile)

    # default parameter
    data = {
        "MoleculeInformation": {
            "numberOfElectrons": 0,
            "numberOfOrbitals": 0,
            "orbitalSymmetries": [],
            "pointGroup": "",
            "quantumNumber_S": 0,
            "quantumNumber_Ms": 0,
        },
        "WavefunctionOptions": {
            "wavefunctionName": "sCI",
            "wavefunctionOperation": "initial",
            "sort": "excitations",
            "excitations": [],
            "frozenElectrons": [],
            "frozenMOs": [],
            "splitAt": 0,
            "thresholdCI": 1.0,
            "keepMin": 0,
        },
        "Output": {
            "plotCICoefficients": False,
            "plotly": False,
        },
        "Specifications": {
            "blocksize": 0,
            "initialAMI": "",
            "iterationAMI": "",
            "keepAllSingles": False,
        },
        "Hardware": {"partition": "p16", "nTasks": "144"},
    }

    # load input in data
    for key, value in input_data.items():
        for sub_key, sub_value in value.items():
            data[key][sub_key] = sub_value
    print(data)

    N = data["MoleculeInformation"]["numberOfElectrons"]
    n_MO = data["MoleculeInformation"]["numberOfOrbitals"]
    S = data["MoleculeInformation"]["quantumNumber_S"]
    M_s = data["MoleculeInformation"]["quantumNumber_Ms"]
    point_group = data["MoleculeInformation"]["pointGroup"]
    orbital_symmetry = data["MoleculeInformation"]["orbitalSymmetries"]

    wavefunction_name = data["WavefunctionOptions"]["wavefunctionName"]
    excitations = data["WavefunctionOptions"]["excitations"]
    frozen_electrons = data["WavefunctionOptions"]["frozenElectrons"]
    frozen_MOs = data["WavefunctionOptions"]["frozenMOs"]
    split_at = data["WavefunctionOptions"]["splitAt"]
    threshold_ci = data["WavefunctionOptions"]["thresholdCI"]
    n_min = data["WavefunctionOptions"]["keepMin"]
    sort = data["WavefunctionOptions"]["sort"]

    blocksize = data["Specifications"]["blocksize"]
    initial_ami = data["Specifications"]["initialAMI"]
    iteration_ami = data["Specifications"]["iterationAMI"]
    final_ami = data["Specifications"]["finalAMI"]
    keep_all_singles = data["Specifications"]["keepAllSingles"]

    partition = data["Hardware"]["partition"]
    n_tasks = data["Hardware"]["nTasks"]

    auto = Automation(
        wavefunction_name,
        N,
        S,
        M_s,
        n_MO,
        excitations,
        orbital_symmetry,
        point_group,
        frozen_electrons,
        frozen_MOs,
        partition,
        n_tasks,
        blocksize,
        sort,
        True,
        n_min,
        threshold_ci,
        keep_all_singles,
    )
    evaluation = Evaluation()
    # call demanded routine

    if data["WavefunctionOptions"]["wavefunctionOperation"] == "initial":
        initial_determinant = sCI.build_energy_lowest_detetminant(N)
        sCI.get_initial_wf(
            S,
            M_s,
            n_MO,
            initial_determinant,
            excitations,
            orbital_symmetry,
            point_group,
            frozen_electrons,
            frozen_MOs,
            wavefunction_name,
            split_at=split_at,
            sort_option=sort,
            verbose=True,
        )

    elif (
        data["WavefunctionOptions"]["wavefunctionOperation"]
        == "do_excitations"
    ):
        initial_determinant = sCI.build_energy_lowest_detetminant(N)
        sCI.select_and_do_excitations(
            N,
            n_MO,
            S,
            M_s,
            initial_determinant,
            excitations,
            orbital_symmetry,
            point_group,
            frozen_electrons,
            frozen_MOs,
            wavefunction_name,
            threshold_ci,
            split_at=split_at,
            verbose=True,
        )

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "block_final":
        auto.do_final_block("intermediate", wavefunction_name, final_ami)

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "blockwise":
        auto.blockwise_optimization(initial_ami, iteration_ami, final_ami)

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "cut":
        # read wf and cut by split_at
        csf_coefficients, csfs, CI_coefficients, wfpretext = (
            sCI.read_AMOLQC_csfs(f"{wavefunction_name}.wf", N)
        )
        csf_coefficients, csfs, CI_coefficients = sCI.sort_csfs_by_CI_coeff(
            csf_coefficients, csfs, CI_coefficients
        )
        sCI.write_AMOLQC(
            csf_coefficients[:split_at],
            csfs[:split_at],
            CI_coefficients[:split_at],
            pretext=wfpretext,
            file_name=f"{wavefunction_name}_out.wf",
        )

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "iterative":
        auto.do_iterative_construction(initial_ami)

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "test":
        indices, energies = auto.parse_csf_energies("amolqc-2", 223)

    if data["Output"]["plotCICoefficients"]:
        if data["Output"]["plotly"]:
            evaluation.plot_ci_coefficients_plotly(wavefunction_name, N, n_MO)
        else:
            evaluation.plot_ci_coefficients(wavefunction_name, N)


# program starts
sCI = SelectedCI()

main()
