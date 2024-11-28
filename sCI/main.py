#!/usr/bin/env python3

import sys
import numpy as np
import time 
import yaml
import math
from pyscript import * # requirement pyscript as python package https://github.com/Leonard-Reuter/pyscript
from csf import SelectedCI

def blockwise_optimization(N,S, M_s, n_MO, excitations, orbital_symmetry, point_group, frozen_electrons, frozen_MOs, wavefunction_name,blocksize,initial_ami, iteration_ami,n_min):
    # 
    # initial block with adding jastrow and optimizing jastrow
    #
    n_block = 1 
    dir_name = f"block{n_block}"
    mkdir(dir_name)
    
    with cd(dir_name):
        cp(f"../{wavefunction_name}.wf",".")
        initial_determinant = sCI.build_energy_lowest_detetminant(N)
        sCI.get_initial_wf(S, M_s, n_MO, initial_determinant, excitations, orbital_symmetry, point_group, frozen_electrons, frozen_MOs, wavefunction_name,split_at=blocksize,verbose = True)
        # extract total number of csfs
        rm(f"{wavefunction_name}.wf")
        mv(f"{wavefunction_name}_out.wf",f"{wavefunction_name}.wf")
        cp(f"../{initial_ami}.ami", ".")

        # get number of csfs
        _, csfs, _, _ = sCI.read_AMOLQC_csfs(f"{wavefunction_name}.wf",N)
        _, csfs_dis, _, _ = sCI.read_AMOLQC_csfs(f"{wavefunction_name}_res.wf",N)
        n_all_csfs = len(csfs) + len(csfs_dis)
        # submit job
        with open("amolqc_job", "w") as printfile:
            printfile.write(f"""#!/bin/bash
#SBATCH --partition=p16      
#SBATCH --job-name=block1 
#SBATCH --output=o.%j       
#SBATCH --ntasks=144        
#SBATCH --ntasks-per-core=1 

# Befehle die ausgeführt werden sollen:
mpiexec -np 144 $AMOLQC/build/bin/amolqc {initial_ami}.ami
""")
        run("sbatch amolqc_job")
        job_done = False
        while not job_done:
            try:
                with open(f"{initial_ami}.amo", "r") as reffile:
                    for line in reffile:
                        if "Amolqc run finished" in line:
                            job_done = True
                            print("Job done")
            except:
                FileNotFoundError
            if not job_done:
                print("Job not done yet.")
                time.sleep(60)
        # get last wavefunction 
        wf_number = 0
        for file_name in ls():
            names = file_name.split(".")
            if names[-1] == "wf" and initial_ami in names[0]:
                temp = names[0]
                number = int(temp.split("-")[-1])
                if number > wf_number:
                    wf_number = number
                    last_wavefunction = file_name
        print(last_wavefunction)
        cp(last_wavefunction, f"../{dir_name}.wf")
        cp(f"{wavefunction_name}_res.wf", f"../{dir_name}_res.wf")
    #
    # perform blockwise iterations
    #
    # number of remaining blocks 
    n_blocks = math.ceil((n_all_csfs - blocksize)/(blocksize - n_min))
    print(f"number of total blocks (without initial block) {n_blocks}")
    
    last_wavefunction = dir_name
    n_block += 1 
    dir_name = f"block{n_block}"
    mkdir(dir_name)
    with cd(dir_name):
        cp(f"../{last_wavefunction}.wf",".")
        mv(f"../{last_wavefunction}_res.wf",".")
        sCI.select_and_do_next_package(N,f"{last_wavefunction}_dis", f"{last_wavefunction}", f"{last_wavefunction}_res", 1.000, split_at=blocksize,n_min=20, verbose=True)
        mv(f"{last_wavefunction}_out.wf",f"{wavefunction_name}.wf")
        mv(f"{last_wavefunction}_dis_out.wf",f"{wavefunction_name}_dis.wf")
        mv(f"{last_wavefunction}_res_out.wf",f"{wavefunction_name}_res.wf")
        cp(f"../{iteration_ami}.ami", ".")
        with open("amolqc_job", "w") as printfile:
            printfile.write(f"""#!/bin/bash
#SBATCH --partition=p16      
#SBATCH --job-name={dir_name} 
#SBATCH --output=o.%j       
#SBATCH --ntasks=144        
#SBATCH --ntasks-per-core=1 

# Befehle die ausgeführt werden sollen:
mpiexec -np 144 $AMOLQC/build/bin/amolqc {iteration_ami}.ami
""")

        run("sbatch amolqc_job")
        job_done = False
        while not job_done:
            try:
                with open(f"{iteration_ami}.amo", "r") as reffile:
                    print("amo exists")
                    for line in reffile:
                        if "Amolqc run finished" in line:
                            print("FOUND")
                            job_done = True
            except:
                FileNotFoundError
                print("file not found")
            if not job_done:
                print("Job not done")
                time.sleep(60)
        
                
    #sCI.select_and_do_next_package("discarded", wavefunction_name, "residual", threshold_ci, split_at=split_at, n_min=n_min, verbose=True)
    

def main():

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
                'blocksize': 0,
                'initialAMI': "",
                'iterationAMI': "",
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
    initial_ami = data['Specifications']['initialAMI']
    iteration_ami = data['Specifications']['iterationAMI']
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
        blockwise_optimization(N,S, M_s, n_MO, excitations, orbital_symmetry, point_group, frozen_electrons, frozen_MOs, wavefunction_name,blocksize, initial_ami, iteration_ami,n_min)

    if data['Output']['plotCICoefficients']: 
        sCI.plot_ci_coefficients(wavefunction_name,N)
        
# program starts
sCI = SelectedCI()
main()


