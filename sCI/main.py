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
from utils import Utils
from cipsi_jas import AddSingles


def main():
    sCI = SelectedCI()

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
            "maxCsfs": 1500,
            "wfType": "csf",
        },
        "Output": {
            "plotCICoefficients": False,
            "plotly": False,
        },
        "Specifications": {
            "criterion": "",
            "threshold": 1.0,
            "thresholdType": "cut_at",
            "keepMin": 0,
            "blocksize": 0,
            "initialAMI": "",
            "iterationAMI": "",
            "finalAMI": "",
            "energyAMI": "",
            "keepAllSingles": False,
        },
        "Hardware": {"partition": "p16", "nTasks": "144"},
    }

    # load input in data
    for key, value in input_data.items():
        for sub_key, sub_value in value.items():
            data[key][sub_key] = sub_value
    # TODO print input mor readable
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
    sort = data["WavefunctionOptions"]["sort"]
    max_csfs = data["WavefunctionOptions"]["maxCsfs"]
    wftype = data["WavefunctionOptions"]["wfType"]

    criterion = data["Specifications"]["criterion"]
    threshold = data["Specifications"]["threshold"]
    threshold_type = data["Specifications"]["thresholdType"]
    n_min = data["Specifications"]["keepMin"]
    blocksize = data["Specifications"]["blocksize"]
    initial_ami = data["Specifications"]["initialAMI"]
    iteration_ami = data["Specifications"]["iterationAMI"]
    energy_ami = data["Specifications"]["energyAMI"]
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
        criterion,
        blocksize,
        sort,
        True,
        n_min,
        threshold,
        threshold_type,
        keep_all_singles,
        max_csfs,
    )
    evaluation = Evaluation()
    utils = Utils()
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

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "block_final":
        auto.do_final_block("intermediate", wavefunction_name, final_ami)

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "blockwise":
        auto.blockwise_optimization(
            initial_ami,
            iteration_ami,
            final_ami,
            energy_ami=energy_ami,
        )

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "cut":
        # read wf and cut by split_at
        csf_coefficients, csfs, CI_coefficients, wfpretext = (
            sCI.read_AMOLQC_csfs(f"{wavefunction_name}.wf", N)
        )
        csf_coefficients, csfs, CI_coefficients = sCI.sort_lists_by_list(
            [csf_coefficients, csfs, CI_coefficients],
            CI_coefficients,
            side=-1,
            absol=True,
        )
        if wftype == "csf":
            n_dets = len(csfs[:split_at])
             # form csfs of these determinants
            csf_coefficients, csfs = sCI.get_unique_csfs(
                csfs[:split_at], S, M_s
            )
            csf_coefficients, csfs = sCI.sort_determinants_in_csfs(
                csf_coefficients, csfs
            )
            CI_coefficients = [
                1 if n == 0 else 0 for n in range(len(csfs))
            ]
            print(f"number of csfs generated from {n_dets} determinants is \
{len(csfs)}.")
        sCI.write_AMOLQC(
            csf_coefficients[:split_at],
            csfs[:split_at],
            CI_coefficients[:split_at],
            pretext=wfpretext,
            file_name=f"{wavefunction_name}_out.wf",
            wftype=wftype,
        )

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "iterative":
        initial_determinant = sCI.build_energy_lowest_detetminant(N)
        auto.do_iterative_construction(
            initial_ami,
            iteration_ami,
            final_ami,
            initial_determinant,
            energy_ami=energy_ami,
        )

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "test":
        # indices, energies = sCI.parse_csf_energies(
        #   "block1/block_initial_dis_out.wf", 30, verbose=True
        # )
        # print(indices)
        # print(energies)
        # exit()
        # auto.do_block_iteration(
        #    4, "block_initial", iteration_ami, energy_ami=energy_ami
        # )
        # auto.do_final_iteration(
        #    "it_final", "it2", 500, final_ami, energy_ami=energy_ami
        # )
        # reference_determinant = sCI.build_energy_lowest_detetminant(N)
        # last_it = auto.do_selective_iteration(
        #    1,
        #    "it2",
        #    iteration_ami,
        #    energy_ami,
        #    reference_determinant,
        #    [1],
        #    excitations,
        # )
        auto.do_final_block("final", "block3", final_ami)

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "exc":
        # read wf and cut by split_at
        csf_coefficients, csfs, CI_coefficients, wfpretext = (
            sCI.read_AMOLQC_csfs(f"{wavefunction_name}.wf", N)
        )
        csf_coefficients, csfs, CI_coefficients = sCI.sort_lists_by_list(
            [csf_coefficients, csfs, CI_coefficients],
            CI_coefficients,
            side=-1,
            absol=True,
        )
        CI_coefficients = [n for n in range(len(csfs), 0, -1)]
        sCI.write_AMOLQC(
            csf_coefficients[:split_at],
            csfs[:split_at],
            CI_coefficients[:split_at],
            pretext=wfpretext,
            file_name=f"mod.wf",
            wftype=wftype,
        )
        reference_determinant = sCI.build_energy_lowest_detetminant(N)
        sCI.select_and_do_excitations(
            N,
            n_MO,
            S,
            M_s,
            reference_determinant,
            excitations,
            [1],
            orbital_symmetry,
            point_group,
            frozen_electrons,
            frozen_MOs,
            "mod",
            f"_",
            criterion,
            threshold,
            max_csfs,
            threshold_type=threshold_type,
            verbose=True,
        )

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "add_singles":
        aS = AddSingles()
        aS.add_singles_det(
            N,
            S,
            M_s,
            n_MO,
            orbital_symmetry,
            point_group,
            frozen_electrons,
            frozen_MOs,
            wavefunction_name,
            wftype,
        )
        exit()

        initial_determinant = sCI.build_energy_lowest_detetminant(N)
        excited_determinants = sCI.get_excitations(
            n_MO,
            [1],
            initial_determinant,
            orbital_symmetry=orbital_symmetry,
            tot_sym=point_group,
            core=frozen_electrons,
            frozen_MOs=frozen_MOs,
        )
        # sort determinants
        temp = []
        for det in excited_determinants:
            _, det_tmp = sCI.sort_determinant(1, det)
            temp.append(det_tmp)
        excited_determinants = temp.copy()
        csf_coefficients, csfs, CI_coefficients, wfpretext = (
            sCI.read_AMOLQC_csfs(f"{wavefunction_name}.wf", N)
        )
        _, _, det_basis = sCI.get_transformation_matrix(
            csf_coefficients, csfs, CI_coefficients
        )
        temp = []
        for det in det_basis:
            _, det_tmp = sCI.sort_determinant(1, det)
            temp.append(det_tmp)
        det_basis = temp.copy()
        all_determinants = det_basis + excited_determinants
        all_determinants = sCI.spinfuncs.remove_duplicates(all_determinants)
        CI_coefficients = [
            1 if n == 0 else 0 for n in range(len(all_determinants))
        ]

        sCI.write_AMOLQC(
            [],
            all_determinants,
            CI_coefficients,
            pretext=wfpretext,
            file_name=f"{wavefunction_name}_keep_sgls.wf",
            wftype="det",
        )

    elif data["WavefunctionOptions"]["wavefunctionOperation"] == "read_cipsi":
        wf_name_praefix = wavefunction_name.split(".")[0]
        # parse determinants and print them in AMOLQC format
        ci_coefficients, determinants = utils.parse_cipsi_dets(
            wavefunction_name
        )
        sCI.write_AMOLQC(
            [],
            determinants[:split_at],
            ci_coefficients[:split_at],
            pretext="",
            file_name=f"{wf_name_praefix}_dets.wf",
            wftype="det",
        )
        print(len(determinants))

        # get csfs from determinant basis and print wavefunction.
        # create guess for CI coefficients
        csf_coefficients, csfs = sCI.get_unique_csfs(determinants, S, M_s)
        csf_coefficients, csfs = sCI.sort_determinants_in_csfs(
            csf_coefficients, csfs
        )
        ci_csf_coefficients = [1 if n == 0 else 0 for n in range(len(csfs))]

        sCI.write_AMOLQC(
            csf_coefficients[:split_at],
            csfs[:split_at],
            ci_csf_coefficients[:split_at],
            pretext="",
            file_name=f"{wf_name_praefix}_csfs.wf",
        )
        print(f"len csfs: {len(csfs)}")

        # expand again in determinants to see how may
        # determinants have been added
        _, _, determinant_basis_csfs = sCI.get_transformation_matrix(
            csf_coefficients, csfs, range(len(csf_coefficients))
        )
        print(len(determinant_basis_csfs))

    if data["Output"]["plotCICoefficients"]:
        if data["Output"]["plotly"]:
            evaluation.plot_ci_coefficients_plotly(wavefunction_name, N, n_MO)
        else:
            evaluation.plot_ci_coefficients(wavefunction_name, N)


# program starts


main()
