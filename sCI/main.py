#!/usr/bin/env python3

import sys
import numpy as np
import yaml
from csf import SelectedCI

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
                'splitAt': 0
            }, 
        'Output': {
                'plotCICoefficients': False
        }, 
        'Calculation': 
            {
                'type': '', 
                'blocksize': 0
            }
            }
    

    # load input in data
    for key, value in input_data.items():
        data[key] = value


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
    print(data)

    # call demanded routine
    if data['WavefunctionOptions']['wavefunctionOperation'] == 'initial':
        initial_determinant = sCI.build_energy_lowest_detetminant(N)
        sCI.get_initial_wf(S, M_s, n_MO, initial_determinant, excitations, orbital_symmetry, point_group, frozen_electrons, frozen_MOs, wavefunction_name,split_at=split_at,verbose = True)
 
        
main()


