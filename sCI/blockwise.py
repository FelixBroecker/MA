import time
import math
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
        initial_ami,
        blockwise_ami,
        n_min,
        ci_threshold,
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
        self.initial_ami = initial_ami
        self.blockwise_ami = blockwise_ami
        self.partition = partition
        self.n_tasks = n_tasks
        self.n_min = (n_min,)
        self.ci_threshild = ci_threshold

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
    mpiexec -np {n_tasks} $AMOLQC/build/bin/amolqc {ami_name}.ami
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
        wf_number = 0
        for file_name in ls():
            names = file_name.split(".")
            if names[-1] == "wf" and ami_name in names[0]:
                temp = names[0]
                number = int(temp.split("-")[-1])
                if number > wf_number:
                    wf_number = number
                    last_wavefunction = file_name
        return last_wavefunction

    def do_initial_block(self, block_label):
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
            cp(f"../{self.initial_ami}.ami", ".")
            # submit job
            self.print_job_file(
                self.partition,
                dir_name,
                self.n_tasks,
                self.initial_ami,
            )
            run("sbatch amolqc_job")
            # check if job done
            job_done = False
            while not job_done:
                job_done = self.check_job_done(self.initial_ami)
                if not job_done:
                    print("job not done yet.")
                    time.sleep(20)
            # get last wavefunction and copy to folder with all blocks
            last_wavefunction = self.get_final_wavefunction(self.initial_ami)
            cp(last_wavefunction, f"../{dir_name}.wf")
            cp(f"{self.wavefunction_name}_res.wf", f"../{dir_name}_res.wf")

    def do_block_iteration(self, n_blocks, initial_dir):
        """"""
        print(f"number of total blocks (without initial block) {n_blocks}")
        n_block = 0
        for i in range(n_blocks):
            print()
            print(f"Block iteration: {i+1}")
            print()
            last_wavefunction = initial_dir
            n_block += 1
            dir_name = f"block{n_block}"
            mkdir(dir_name)
            # get wavefunctions from previous iterations
            with cd(dir_name):
                try:
                    mv(f"../{last_wavefunction}_dis.wf", ".")
                except FileNotFoundError():
                    pass
                cp(f"../{last_wavefunction}.wf", ".")
                mv(f"../{last_wavefunction}_res.wf", ".")
                # TODO continue after here
                self.sCI.select_and_do_next_package(
                    self.N,
                    f"{last_wavefunction}_dis",
                    f"{last_wavefunction}",
                    f"{last_wavefunction}_res",
                    self.ci_threshold,
                    split_at=self.blocksize,self.ci_threshold
                    n_min=self.n_min,
                    verbose=self.verbose,
                )
                mv(f"{last_wavefunction}_out.wf", f"{wavefunction_name}.wf")
                mv(
                    f"{last_wavefunction}_dis_out.wf",
                    f"{wavefunction_name}_dis.wf",
                )
                mv(
                    f"{last_wavefunction}_res_out.wf",
                    f"{wavefunction_name}_res.wf",
                )
                cp(f"../{iteration_ami}.ami", ".")
                with open("amolqc_job", "w") as printfile:
                    printfile.write(
                        f"""#!/bin/bash
    #SBATCH --partition=p64
    #SBATCH --job-name={dir_name}
    #SBATCH --output=o.%j
    #SBATCH --ntasks=192
    #SBATCH --ntasks-per-core=1

    # Befehle die ausgeführt werden sollen:
    mpiexec -np 192 $AMOLQC/build/bin/amolqc {iteration_ami}.ami
    """
                    )

                run("sbatch amolqc_job")
                job_done = False
                while not job_done:
                    try:
                        with open(f"{self.iteration_ami}.amo", "r") as reffile:
                            for line in reffile:
                                if "Amolqc run finished" in line:
                                    job_done = True
                                    print("job done.")
                    except:
                        FileNotFoundError
                    if not job_done:
                        print("job not done yet.")
                        time.sleep(20)
                    # get last wavefunction
                    wf_number = 0
                    for file_name in ls():
                        names = file_name.split(".")
                        if (
                            names[-1] == "wf"
                            and self.iteration_ami in names[0]
                        ):
                            temp = names[0]
                            number = int(temp.split("-")[-1])
                            if number > wf_number:
                                wf_number = number
                                last_wavefunction = file_name
                    cp(last_wavefunction, f"../{dir_name}.wf")
                    cp(
                        f"{self.wavefunction_name}_res.wf",
                        f"../{dir_name}_res.wf",
                    )
                    cp(
                        f"{self.wavefunction_name}_dis.wf",
                        f"../{dir_name}_dis.wf",
                    )

    def blockwise_optimization(
        self,
        N,
        S,
        M_s,
        n_MO,
        excitations,
        orbital_symmetry,
        point_group,
        frozen_electrons,
        frozen_MOs,
        wavefunction_name,
        blocksize,
        initial_ami,
        iteration_ami,
        n_min,
        partition,
        n_tasks,
        ci_threshold,
        sort="",
        keep_all_singles=False,
    ):
        #
        # initial block with adding jastrow and optimizing jastrow
        #
        n_block = "initial"
        self.do_initial_block(n_block)
        #
        # perform blockwise iterations
        #
        # TODO get n_blocks from somewhere
        # number of remaining blocks
        # get number of csfs
        _, csfs, _, _ = self.sCI.read_AMOLQC_csfs(
            f"{self.wavefunction_name}.wf", self.N
        )
        _, csfs_dis, _, _ = self.sCI.read_AMOLQC_csfs(
            f"{self.wavefunction_name}_res.wf", self.N
        )
        n_all_csfs = len(csfs) + len(csfs_dis)
        n_blocks = math.ceil(
            (n_all_csfs - self.blocksize) / (self.blocksize - self.n_min)
        )

        #
        # do finial selection from all CI coefficients
        #
        mkdir(dir_name)
        with cd(dir_name):
            cp(f"../{last_wavefunction}.wf", ".")
            mv(f"../{last_wavefunction}_res.wf", ".")
            mv(f"../{last_wavefunction}_dis.wf", ".")
            #
            csf_coefficients, csfs, CI_coefficients, wfpretext = (
                sCI.read_AMOLQC_csfs(f"{last_wavefunction}.wf", N)
            )
            csf_coefficients_dis, csfs_dis, CI_coefficients_dis, _ = (
                sCI.read_AMOLQC_csfs(f"{last_wavefunction}_dis.wf", N)
            )
            csf_coefficients += csf_coefficients_dis
            csfs += csfs_dis
            CI_coefficients += CI_coefficients_dis
            # keep all sinlge excitations
            idx = 0
            keep_all_singles = False
            if keep_all_singles:
                csf_coefficients, csfs, CI_coefficients = (
                    sCI.sort_order_of_csfs(
                        csf_coefficients,
                        csfs,
                        CI_coefficients,
                        reference_determinant=initial_determinant,
                        option="by_excitation",
                    )
                )
                n_excitation = sCI.determine_excitations(
                    csfs, initial_determinant
                )
                idx = 0
                for i, n_excitation in enumerate(n_excitation):
                    if n_excitation > 1:
                        idx = i
                        break

            # sort csfs by CI coeffs
            csf_coefficients[idx:], csfs[idx:], CI_coefficients[idx:] = (
                sCI.sort_csfs_by_CI_coeff(
                    csf_coefficients[idx:], csfs[idx:], CI_coefficients[idx:]
                )
            )
            #
            sCI.write_AMOLQC(
                csf_coefficients[:blocksize],
                csfs[:blocksize],
                CI_coefficients[:blocksize],
                pretext=wfpretext,
                file_name=f"{wavefunction_name}.wf",
            )
            sCI.write_AMOLQC(
                csf_coefficients[blocksize:],
                csfs[blocksize:],
                CI_coefficients[blocksize:],
                file_name=f"{wavefunction_name}_dis.wf",
            )

            cp(f"../{iteration_ami}.ami", ".")
            with open("amolqc_job", "w") as printfile:
                printfile.write(
                    f"""#!/bin/bash
    #SBATCH --partition=p64
    #SBATCH --job-name={dir_name}
    #SBATCH --output=o.%j
    #SBATCH --ntasks=192
    #SBATCH --ntasks-per-core=1

    # Befehle die ausgeführt werden sollen:
    mpiexec -np 192 $AMOLQC/build/bin/amolqc {iteration_ami}.ami
    """
                )

            run("sbatch amolqc_job")
            job_done = False
            while not job_done:
                try:
                    with open(f"{iteration_ami}.amo", "r") as reffile:
                        for line in reffile:
                            if "Amolqc run finished" in line:
                                job_done = True
                                print("job done.")
                except:
                    FileNotFoundError
                if not job_done:
                    print("job not done yet.")
                    time.sleep(20)

            # get last wavefunction
            wf_number = 0
            for file_name in ls():
                names = file_name.split(".")
                if names[-1] == "wf" and iteration_ami in names[0]:
                    temp = names[0]
                    number = int(temp.split("-")[-1])
                    if number > wf_number:
                        wf_number = number
                        last_wavefunction = file_name
            cp(last_wavefunction, f"../{dir_name}.wf")
            print()
            print("finish blockwise optimization.")

        # sCI.select_and_do_next_package("discarded", wavefunction_name, "residual", threshold_ci, split_at=split_at, n_min=n_min, verbose=True)
