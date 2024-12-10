import time
import math
import numpy as np
from pyscript import *  # requirement pyscript as python package https://github.com/Leonard-Reuter/pyscript
from csf import SelectedCI


class Automation:
    def __init__(
        self,
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
        sort_option,
        verbose,
        n_min,
        ci_threshold,
        keep_all_singles,
    ):
        self.sCI = SelectedCI()
        self.wavefunction_name = wavefunction_name
        self.N = N
        self.S = S
        self.M_s = M_s
        self.n_MO = n_MO
        self.excitations = excitations
        self.orbital_symmetry = orbital_symmetry
        self.point_group = point_group
        self.frozen_electrons = frozen_electrons
        self.frozen_MOs = frozen_MOs
        self.blocksize = blocksize
        self.sort_option = sort_option
        self.verbose = verbose
        self.partition = partition
        self.n_tasks = n_tasks
        self.n_min = n_min
        self.ci_threshold = ci_threshold
        self.keep_all_singles = keep_all_singles
        self.n_all_csfs = 0

    def print_job_file(
        self,
        partition,
        job_name,
        n_tasks,
        ami_name,
        jobfile_name="amolqc_job",
    ):
        with open(f"{jobfile_name}", "w") as printfile:
            printfile.write(
                f"""#!/bin/bash
#SBATCH --partition={partition}
#SBATCH --job-name={job_name}
#SBATCH --output=o.%j
#SBATCH --ntasks={n_tasks}
#SBATCH --ntasks-per-core=1
# Befehle die ausgeführt werden sollen:
mpiexec -np {n_tasks} /home/theochem/Amolqc/build-heap/bin/amolqc {ami_name}.ami
"""
            )

    def check_job_done(self, amo_name, verbose=True):
        job_done = False
        try:
            with open(f"{amo_name}.amo", "r") as reffile:
                for line in reffile:
                    if "Amolqc run finished" in line:
                        job_done = True
                        if verbose:
                            print("job done.")
        except FileNotFoundError:
            pass
        return job_done

    def get_final_wavefunction(self, ami_name):
        last_wavefunction = ""
        wf_number = 0
        for file_name in ls():
            names = file_name.split(".")
            if names[-1] == "wf" and ami_name in names[0]:
                temp = names[0]
                number = int(temp.split("-")[-1])
                if number > wf_number:
                    wf_number = number
                    last_wavefunction = file_name
        return last_wavefunction.split(".")[0]

    def get_n_all_csfs(self, input_wf):
        """"""
        n_csfs = 0
        _, csfs, _, _ = self.sCI.read_AMOLQC_csfs(f"{input_wf}.wf", self.N)
        n_csfs += len(csfs)
        try:
            _, csfs_res, _, _ = self.sCI.read_AMOLQC_csfs(
                f"{input_wf}_res.wf", self.N
            )
            n_csfs += len(csfs_res)
        except FileNotFoundError:
            if self.verbose:
                print(f"{input_wf}_res.wf not found.")
        try:
            _, csfs_dis, _, _ = self.sCI.read_AMOLQC_csfs(
                f"{input_wf}_dis.wf", self.N
            )
            n_csfs += len(csfs_dis)
        except FileNotFoundError:
            print(f"{input_wf}_dis.wf not found.")
        return n_csfs

    def do_initial_block(self, block_label, initial_ami: str):
        dir_name = f"block{block_label}"
        mkdir(dir_name)

        with cd(dir_name):
            cp(f"../{self.wavefunction_name}.wf", ".")
            initial_determinant = self.sCI.build_energy_lowest_detetminant(
                self.N
            )
            self.sCI.get_initial_wf(
                self.S,
                self.M_s,
                self.n_MO,
                initial_determinant,
                self.excitations,
                self.orbital_symmetry,
                self.point_group,
                self.frozen_electrons,
                self.frozen_MOs,
                self.wavefunction_name,
                split_at=self.blocksize,
                sort_option=self.sort_option,
                verbose=self.verbose,
            )
            # extract total number of csfs
            rm(f"{self.wavefunction_name}.wf")
            mv(
                f"{self.wavefunction_name}_out.wf",
                f"{self.wavefunction_name}.wf",
            )
            cp(f"../{initial_ami}.ami", ".")
            # submit job
            self.print_job_file(
                self.partition,
                dir_name,
                self.n_tasks,
                initial_ami,
            )
            run("sbatch amolqc_job")
            # check if job done
            job_done = False
            while not job_done:
                job_done = self.check_job_done(initial_ami)
                if not job_done:
                    if self.verbose:
                        print("job not done yet.")
                    time.sleep(20)
            # get last wavefunction and copy to folder with all blocks
            last_wavefunction = self.get_final_wavefunction(initial_ami)
            cp(f"{last_wavefunction}.wf", f"../{dir_name}.wf")
            cp(f"{self.wavefunction_name}_res.wf", f"../{dir_name}_res.wf")
        if self.verbose:
            print("finish initial block.")

    def parse_csf_energies(
        self, input_amo: str, n_csfs: int, sort_by_idx=True
    ):
        """"""
        energies = []
        indices = []
        with open(f"{input_amo}.amo", "r") as reffile:
            found = False
            counter = 0
            for line in reffile:
                if counter == n_csfs:
                    break
                if found:
                    counter += 1
                    items = line.split()
                    indices.append(int(items[0]))
                    energies.append(float(items[1]))
                if "  Index  Energy difference" in line:
                    found = True
        if sort_by_idx:
            idx = np.array(indices).argsort()
            indices = [indices[i] for i in idx]
            energies = [energies[i] for i in idx]

        return indices, energies

    def do_block_iteration(
        self, n_blocks: int, input_wf: str, blockwise_ami: str
    ):
        """"""
        if self.verbose:
            print(f"number of total blocks (without initial block) {n_blocks}")
        n_block = 0
        last_wavefunction = input_wf
        for i in range(n_blocks):
            print()
            print(f"Block iteration: {i+1}/{n_blocks}")
            print()
            n_block += 1
            dir_name = f"block{n_block}"
            mkdir(dir_name)
            # get wavefunctions from previous iterations
            with cd(dir_name):
                try:
                    mv(f"../{last_wavefunction}_dis.wf", ".")
                except FileNotFoundError:
                    pass
                cp(f"../{last_wavefunction}.wf", ".")
                mv(f"../{last_wavefunction}_res.wf", ".")
                self.sCI.select_and_do_next_package(
                    self.N,
                    f"{last_wavefunction}_dis",
                    f"{last_wavefunction}",
                    f"{last_wavefunction}_res",
                    self.ci_threshold,
                    split_at=self.blocksize,
                    n_min=self.n_min,
                    verbose=self.verbose,
                )
                mv(
                    f"{last_wavefunction}_out.wf",
                    f"{self.wavefunction_name}.wf",
                )
                mv(
                    f"{last_wavefunction}_dis_out.wf",
                    f"{self.wavefunction_name}_dis.wf",
                )
                mv(
                    f"{last_wavefunction}_res_out.wf",
                    f"{self.wavefunction_name}_res.wf",
                )
                cp(f"../{blockwise_ami}.ami", ".")
                # submit job
                self.print_job_file(
                    self.partition,
                    dir_name,
                    self.n_tasks,
                    blockwise_ami,
                )
                run("sbatch amolqc_job")
                job_done = False
                while not job_done:
                    job_done = self.check_job_done(blockwise_ami)
                    if not job_done:
                        if self.verbose:
                            print("job not done yet.")
                        time.sleep(20)
                    # get last wavefunction and copy to folder with all blocks
                optimized_wf = self.get_final_wavefunction(blockwise_ami)
                cp(f"{optimized_wf}.wf", f"../{dir_name}.wf")
                cp(
                    f"{self.wavefunction_name}_res.wf",
                    f"../{dir_name}_res.wf",
                )
                cp(
                    f"{self.wavefunction_name}_dis.wf",
                    f"../{dir_name}_dis.wf",
                )
                last_wavefunction = dir_name
        if self.verbose:
            print("finish blockwise optimization")

    def do_final_block(self, block_label: str, input_wf: str, final_ami: str):
        #
        # do finial selection from all CI coefficients
        #
        dir_name = f"block_{block_label}"
        mkdir(dir_name)
        with cd(dir_name):
            cp(f"../{input_wf}.wf", ".")
            mv(f"../{input_wf}_res.wf", ".")
            mv(f"../{input_wf}_dis.wf", ".")
            #
            csf_coefficients, csfs, CI_coefficients, wfpretext = (
                self.sCI.read_AMOLQC_csfs(f"{input_wf}.wf", self.N)
            )
            csf_coefficients_dis, csfs_dis, CI_coefficients_dis, _ = (
                self.sCI.read_AMOLQC_csfs(f"{input_wf}_dis.wf", self.N)
            )
            csf_coefficients += csf_coefficients_dis
            csfs += csfs_dis
            CI_coefficients += CI_coefficients_dis
            # keep all singe excitations
            idx = 0
            if self.keep_all_singles:
                initial_determinant = self.sCI.build_energy_lowest_detetminant(
                    self.N
                )
                csf_coefficients, csfs, CI_coefficients = (
                    self.sCI.sort_order_of_csfs(
                        csf_coefficients,
                        csfs,
                        CI_coefficients,
                        reference_determinant=initial_determinant,
                        option="by_excitation",
                    )
                )
                n_excitation = self.sCI.determine_excitations(
                    csfs, initial_determinant
                )
                idx = 0
                for i, n_excitation in enumerate(n_excitation):
                    if n_excitation > 1:
                        idx = i
                        break

            # sort csfs by CI coeffs
            csf_coefficients[idx:], csfs[idx:], CI_coefficients[idx:] = (
                self.sCI.sort_csfs_by_CI_coeff(
                    csf_coefficients[idx:], csfs[idx:], CI_coefficients[idx:]
                )
            )
            #
            self.sCI.write_AMOLQC(
                csf_coefficients[: self.blocksize],
                csfs[: self.blocksize],
                CI_coefficients[: self.blocksize],
                pretext=wfpretext,
                file_name=f"{self.wavefunction_name}.wf",
            )
            self.sCI.write_AMOLQC(
                csf_coefficients[self.blocksize :],
                csfs[self.blocksize :],
                CI_coefficients[self.blocksize :],
                file_name=f"{self.wavefunction_name}_dis.wf",
            )

            cp(f"../{final_ami}.ami", ".")
            # submit job
            self.print_job_file(
                self.partition,
                dir_name,
                self.n_tasks,
                final_ami,
            )
            run("sbatch amolqc_job")
            # check if job done
            job_done = False
            while not job_done:
                job_done = self.check_job_done(final_ami)
                if not job_done:
                    if self.verbose:
                        print("job not done yet.")
                    time.sleep(20)

            # get last wavefunction and copy to folder with all blocks
            last_wavefunction = self.get_final_wavefunction(final_ami)
            cp(f"{last_wavefunction}.wf", f"../{dir_name}.wf")
            if self.verbose:
                print()
                print("finish final block.")

    def do_initial_iteration(self, block_label, initial_ami):
        """"""
        dir_name = f"block{block_label}"
        mkdir(dir_name)

        with cd(dir_name):
            cp(f"../{self.wavefunction_name}.wf", ".")
            initial_determinant = self.sCI.build_energy_lowest_detetminant(
                self.N
            )
            self.sCI.get_initial_wf(
                self.S,
                self.M_s,
                self.n_MO,
                initial_determinant,
                self.excitations,
                self.orbital_symmetry,
                self.point_group,
                self.frozen_electrons,
                self.frozen_MOs,
                self.wavefunction_name,
                sort_option=self.sort_option,
                verbose=self.verbose,
            )
            # extract total number of csfs
            rm(f"{self.wavefunction_name}.wf")
            mv(
                f"{self.wavefunction_name}_out.wf",
                f"{self.wavefunction_name}.wf",
            )
            cp(f"../{initial_ami}.ami", ".")
            # submit job
            self.print_job_file(
                self.partition,
                dir_name,
                self.n_tasks,
                initial_ami,
            )
            run("sbatch amolqc_job")
            # check if job done
            job_done = False
            while not job_done:
                job_done = self.check_job_done(initial_ami)
                if not job_done:
                    if self.verbose:
                        print("job not done yet.")
                    time.sleep(20)
            # get last wavefunction and copy to folder with all blocks
            last_wavefunction = self.get_final_wavefunction(initial_ami)
            cp(f"{last_wavefunction}.wf", f"../{dir_name}.wf")
            cp(f"{self.wavefunction_name}_res.wf", f"../{dir_name}_res.wf")
        if self.verbose:
            print("finish initial block.")

    def blockwise_optimization(self, initial_ami, blockwise_ami, final_ami):
        #
        # initial block with adding jastrow and optimizing jastrow
        #
        n_block = "_initial"
        self.do_initial_block(n_block, initial_ami)
        #
        # perform blockwise iterations
        #
        # TODO get n_blocks from somewhere. fill here n_csfs
        # get number of csfs

        self.n_all_csfs = self.get_n_all_csfs("block_initial")
        n_blocks = math.ceil(
            (self.n_all_csfs - self.blocksize) / (self.blocksize - self.n_min)
        )
        # blockwise iteration
        self.do_block_iteration(n_blocks, "block_initial", blockwise_ami)
        # final block
        self.do_final_block("final", f"block{n_blocks}", final_ami)

        # sCI.select_and_do_next_package("discarded", wavefunction_name, "residual", threshold_ci, split_at=split_at, n_min=n_min, verbose=True)

    def iterative_construction(self, initial_ami):
        """"""
        n_block = 1
