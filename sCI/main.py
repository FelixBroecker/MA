#!/usr/bin/env python3

import sys
import numpy as np
import yaml
from pyscript import * # requirement pyscript as python package https://github.com/Leonard-Reuter/pyscript
from csf import SelectedCI

def blockwise_optimization():
    
    # initial block
    n_block = 1 
    mkdir(f"block{n_block}")
    with cd("block{n_block}"):
        initial_determinant = sCI.build_energy_lowest_detetminant(N)
        sCI.get_initial_wf(S, M_s, n_MO, initial_determinant, excitations, orbital_symmetry, point_group, frozen_electrons, frozen_MOs, wavefunction_name,split_at=blocksize,verbose = True)
    
    #sCI.select_and_do_next_package("discarded", wavefunction_name, "residual", threshold_ci, split_at=split_at, n_min=n_min, verbose=True)
    

def main():
    sCI = SelectedCI()

    if len(sys.argv) == 1:
        sys.exit('''
        Script to generate wavefunctions for selected CI calculation and run CI calculations.
        
        usage: main.py <infile>
    
        with:
            <infile> being an .yaml file with all specification on molecule and demanded calculations.
    ''')
    input_file = sys.argv[1]
    with open(input_file, "r") as reffile:
        input_data = yaml.safe_load(reffile)

    # default parameter
    data = {
        'MoleculeInformation': 
            {
                'numberOfElectrons': 0, 
                'numberOfOrbitals': 0, 
                'orbitalSymmetries': [],
                'pointGroup': '', 
                'quantumNumber_S': 0, 
                'quantumNumber_Ms': 0
            }, 
        'WavefunctionOptions': 
            {
                'wavefunctionName': 'sCI', 
                'wavefunctionOperation': 'initial', 
                'sort': 'excitations', 
                'excitations': [], 
                'frozenElectrons': [], 
                'frozenMOs': [], 
                'splitAt': 0,
                'thresholdCI': 1.,
                'keepMin': 0,
            }, 
        'Output': {
                'plotCICoefficients': False
        }, 
        'Specifications': 
            {
                'type': '', 
                'blocksize': 0
            }
            }
    

    # load input in data
    for key, value in input_data.items():
        for sub_key, sub_value in value.items():
            data[key][sub_key] = sub_value
    print(data)


    N = data['MoleculeInformation']['numberOfElectrons']
    n_MO = data['MoleculeInformation']['numberOfOrbitals']
    S = data['MoleculeInformation']['quantumNumber_S']
    M_s = data['MoleculeInformation']['quantumNumber_Ms']
    point_group = data['MoleculeInformation']['pointGroup']
    orbital_symmetry = data['MoleculeInformation']['orbitalSymmetries']
    
    wavefunction_name = data['WavefunctionOptions']['wavefunctionName']
    excitations = data['WavefunctionOptions']['excitations']
    frozen_electrons = data['WavefunctionOptions']['frozenElectrons']
    frozen_MOs = data['WavefunctionOptions']['frozenMOs']
    split_at = data['WavefunctionOptions']['splitAt']
    threshold_ci = data['WavefunctionOptions']['thresholdCI']
    n_min = data['WavefunctionOptions']['keepMin']

    blocksize = data['Specifications']['blocksize']
    do_blockwise = False

    # call demanded routine
    if data['WavefunctionOptions']['wavefunctionOperation'] == 'initial':
        initial_determinant = sCI.build_energy_lowest_detetminant(N)
        sCI.get_initial_wf(S, M_s, n_MO, initial_determinant, excitations, orbital_symmetry, point_group, frozen_electrons, frozen_MOs, wavefunction_name,split_at=split_at,verbose = True)
    elif data['WavefunctionOptions']['wavefunctionOperation'] == 'do_excitations':
        initial_determinant = sCI.build_energy_lowest_detetminant(N)
        sCI.select_and_do_excitations(N,n_MO,S,M_s,initial_determinant, excitations ,orbital_symmetry, point_group,
                                      frozen_electrons, frozen_MOs, wavefunction_name, threshold_ci, split_at=split_at, verbose=True)
        # TODO implement in csf.py the option for n_min not only by CI cofficients
    elif data['WavefunctionOptions']['wavefunctionOperation'] == 'blockwise':
        #blockwise_optimization()
        # initial block
        n_block = 1 
        mkdir(f"block{n_block}")
        with cd(f"block{n_block}"):
            initial_determinant = sCI.build_energy_lowest_detetminant(N)
            sCI.get_initial_wf(S, M_s, n_MO, initial_determinant, excitations, orbital_symmetry, point_group, frozen_electrons, frozen_MOs, wavefunction_name,split_at=blocksize,verbose = True)
    if data['Output']['plotCICoefficients']: 
        sCI.plot_ci_coefficients(wavefunction_name,N)
        
main()


