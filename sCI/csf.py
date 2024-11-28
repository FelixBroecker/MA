#!/usr/bin/env python3

import numpy as np
import random
from fractions import Fraction
from charactertables import CharacterTable
from spincoupling import SpinCoupling


class SelectedCI():
    """generate excited determinants"""
    def __init__(self):
        self.spinfuncs = SpinCoupling()
    
    def custom_sort(self,x):
        return (abs(x), x < 0)
    
    def write_AMOLQC(self, csf_coefficients, csfs, CI_coefficients, pretext="", file_name="sCI/csfs.out", write_file = True, verbose=False):
        """determinant representation in csfs needs to be sorted for alpha spins first
        and then beta spins"""
        out="$csfs\n"
        out += f"{int(len(csfs)): >7}\n"
        for i, csf in enumerate(csfs):
            out += f"{CI_coefficients[i]: >10.7f}       {len(csf)}\n"
            for j, determinant in enumerate(csf):
                out += f" {csf_coefficients[i][j]: 9.7f}"
                for electron in determinant:
                    out += f"  {abs(electron)}"
                out += "\n"
        out += "$end"
        out = pretext + out
        if verbose:
            print(out)
        if write_file:
            with open(file_name, "w") as printfile:
                printfile.write(out)

    def read_AMOLQC_csfs(self, filename, n_elec, wftype="csf"):
        """read in csfs of AMOLQC format with CI coefficients"""
        csf_coefficients = []
        csfs = []
        CI_coefficients = []
        csf_tmp = []
        csf_coefficient_tmp = []
        pretext = ""
        # read csfs
        if wftype == "csf":
            with open(f"{filename}", "r") as f:
                found_csf = False
                found_det = False
                for line in f:
                    if "$det" in line:
                        found_det = True
                    if "$csfs" in line:
                        found_csf = True
                        new_csf = True
                        # extract number of csfs
                        line = f.readline()
                        n_csfs = int(line)
                        line = f.readline()
                        # initialize counter to iterate over csfs
                        csf_counter = 0
                    if not found_csf and not found_det:
                        pretext += line
                    if found_csf:
                        entries = line.split()
                        if new_csf:
                            CI_coefficients.append(float(entries[0]))
                            n_summands = int(entries[1])
                            summand_counter = 0
                            new_csf = False
                            csf_counter +=1
                        else:
                            det = []
                            for i in range(1,len(entries)):
                                if i <= n_elec/2:
                                    det.append(1 * int(entries[i]))
                                else:
                                    det.append(-1 * int(entries[i]))
                            csf_coefficient_tmp.append(float(entries[0]))
                            csf_tmp.append(det.copy())
                            summand_counter += 1
                            if summand_counter == n_summands:
                                new_csf = True
                                csf_coefficients.append(csf_coefficient_tmp)
                                csfs.append(csf_tmp)
                                csf_coefficient_tmp = []
                                csf_tmp = []
                                if csf_counter == n_csfs:
                                    break
        return csf_coefficients, csfs, CI_coefficients, pretext
    

    def get_transformation_matrix(self, csf_coefficients: list, csfs: list, CI_coefficients: list):
        """convert csfs and MO coefficients in CI coefficient matrix, csf coefficient matrix, and
        resprective determinant basis

        Parameters
        ----------
        csf_coefficients : list
            list of csf coefficients.
        csfs : list
            list of determinants that builds csf with coefficient from coefficient list.
        CI_coefficients : list
            list of csf CI coefficients.

        Returns
        -------
        CI_coefficient_matrix : list
            Square matrix (n_csf x n_csf) with CI coefficients on the diagonal.
        transformation_matrix : numpy array
            Transformation matrix that stores row-wise the coupling coefficients of respective determinants
            in determinant basis to form csf.
        det_basis : numpy array
            All unique determinants that are basis to form csfs.
        """
        det_basis = []
        n_csfs = len(csf_coefficients)
        # expand determinants from csfs in determinant basis
        for csf in csfs:
            for det in csf:
                if det not in det_basis:
                    det_basis.append(det)
        # get transformation matrix and CI coefficient matrix
        n_dets = len(det_basis)
        transformation_matrix = np.zeros((n_csfs,n_dets))
        CI_coefficient_matrix = np.zeros((n_csfs,n_csfs))
        for i in range(len(csfs)):
            CI_coefficient_matrix[i,i] = CI_coefficients[i]
            for j in range(len(csfs[i])):
                det_idx = det_basis.index(csfs[i][j])
                transformation_matrix[i][det_idx] = csf_coefficients[i][j]

        return CI_coefficient_matrix, transformation_matrix, det_basis

        
    def get_determinant_symmetry(self, determinant, orbital_symmetry, molecule_symmetry):
        """determine symmetry of total determinant based on occupation of molecular orbitals 
        with certain symmetry"""
        # TODO write test for get_determinant symmetry
        # get characters
        symmetry = CharacterTable(molecule_symmetry)
        character = symmetry.characters
        # initialize product with symmetry of first electron
        prod = character[orbital_symmetry[abs(determinant[0])-1]]
         # multiply up for each electron the orbital symmetry. 
        for i in range(1,len(determinant)):
            prod = symmetry.multiply(prod,character[orbital_symmetry[abs(determinant[i])-1]])
        symm = symmetry.character2label(prod)
        return symm

    def sort_determinant(self, coefficient, determinant):
        # replace alpha spin by inverse as fraction 
        for i in range(len(determinant)):
            if determinant[i] > 0:
                determinant[i] = Fraction(1, determinant[i])
        # bubble sort from large to small
        n_swap = 0
        for i in range(len(determinant) - 1, 0, -1):
            for j in range(i):
                if determinant[j] < determinant[j + 1]:
                    determinant[j], determinant[j + 1] = determinant[j + 1], determinant[j]
                    coefficient = -1 * coefficient
                    n_swap +=1
        # replacing inverse alpha spins again by their inverse
        for i in range(len(determinant)):
            if determinant[i] > 0:
                determinant[i] = int(Fraction(1, determinant[i]))
        return coefficient, determinant
    
    def sort_determinants_in_csfs(self, csf_coefficients, csfs):
        "sort each determinant in determinant list to obtain correct AMOLQC format"
        for i, determinants in enumerate(csfs):
            for j in range(len(determinants)):
                csf_coefficients[i][j], csfs[i][j] = self.sort_determinant(csf_coefficients[i][j], csfs[i][j])
        return csf_coefficients, csfs

    def is_singulett(self, determinant):
        """check if single slaterdeterminant is a singulett spineigenfunction"""
        # Check if all values in the counter (i.e., occurrences) are exactly 2
        return all(determinant.count(x) == 2 for x in set(determinant))
    
    def cut_csfs(self, csf_coefficients, csfs, CI_coefficients, CI_coefficient_thresh):
        """cut off csf coefficients, csfs, and CI coefficients by the size of the CI coefficients
        Parameters
        ----------
        csf_coefficients : list
            list of csf coefficients.
        csfs : list
            list of determinants that builds csf with coefficient from coefficient list.
        CI_coefficients : list
            list of csf CI coefficients.
        CI_coefficient_thresh: float
            cut-off value below which the csfs are discarded.

        Returns
        -------
        csf_coefficients : list
            list of csf coefficients.
        csfs : list
            list of determinants that builds csf with coefficient from coefficient list.
        CI_coefficients : list
            list of csf CI coefficients.
        cut_csf_coefficients : list
            list of discarded csf coefficients.
        cut_csfs : list
            list of discarded determinants that build csf with coefficient from coefficient list.
        cut_CI_coefficients : list
            list of discarded csf CI coefficients.
        """
        # sort CI coefficients from largest to smallest absolut value and respectively csfs and csf_coefficients
        CI_coefficients_abs = -np.abs(np.array(CI_coefficients))
        idx = CI_coefficients_abs.argsort()

        CI_coefficients = [CI_coefficients[i] for i in idx]
        csf_coefficients = [csf_coefficients[i] for i in idx]
        csfs = [csfs[i] for i in idx]

        # cut off csfs below CI coefficient threshold
        cut_CI_coefficients = []
        cut_csf_coefficients = []
        cut_csfs = []
        cut_off = False
        for i,coeff in enumerate(CI_coefficients):
            if abs(coeff)<CI_coefficient_thresh:
                cut_off = True
                i_cut = i
                break

        if cut_off:
            cut_CI_coefficients += CI_coefficients[i_cut:]
            cut_csf_coefficients += csf_coefficients[i_cut:]
            cut_csfs += csfs[i_cut:]

            CI_coefficients = CI_coefficients[:i_cut]
            csf_coefficients = csf_coefficients[:i_cut]
            csfs = csfs[:i_cut]
        return csf_coefficients, csfs, CI_coefficients, cut_csf_coefficients, cut_csfs, cut_CI_coefficients
    

    def build_energy_lowest_detetminant(self, n_elecs):
        # create HF determinant, if no initial determinant is passed
        det = []
        n_doubly_occ = n_elecs // 2
        orbital = 1
        if n_elecs > 1:
            for _ in range(n_doubly_occ):
                det.append((orbital))
                det.append(-(orbital))
                orbital += 1
        if n_elecs%2:
            det.append((orbital))
        return det
    

    def get_excitations(self, n_orbitals, excitations, det_ini, orbital_symmetry=[], tot_sym="",det_reference=[], core=[],frozen_MOs=[]):
        """create all excitation determinants"""

        # all unoccupied MOs are virtual orbitals
        virtuals = [i for i in range(-n_orbitals,n_orbitals+1) if i not in det_ini and i != 0]

        # consider symmetry if symmetry is specified in input
        consider_symmetry = bool(orbital_symmetry)

        # determine symmetry of input determinant
        if consider_symmetry:
            symm_of_det_ini = self.get_determinant_symmetry(det_ini, orbital_symmetry, tot_sym)

        # initialize list to mask excitations
        n_elec = len(det_ini)
        n_virt = len(virtuals)
        occ_mask = [True for _ in range(n_elec)]
        virt_mask = [True for _ in range(n_virt)]

        # do excitations from electrons that correspond to not excited electrons in reference determinant
        if det_reference: 
            #assert bool(det_reference), "reference_excitation is True but no reference state has been passed"
            virtuals_reference = [i for i in range(-n_orbitals,n_orbitals+1) if i not in det_reference and i != 0]
            for idx, i in enumerate(det_ini):
                if i not in det_reference:
                    occ_mask[idx] = False
            for idx, a in enumerate(virtuals):
                if a not in virtuals_reference:
                    virt_mask[idx] = False
        
        # if core electrons are passed, no excitations shall be performed with these electrons
        if core:
            for idx, i in enumerate(det_ini):
                if i in core:
                    occ_mask[idx] = False
        if frozen_MOs:
            for idx, i in enumerate(virtuals):
                if i in frozen_MOs:
                    virt_mask[idx] = False

        # generate all required excitations on initial determinant
        excited_determinants = []
        def get_n_fold_excitation(occupied, virtual, n_fold_excitation, occ_mask=[], virt_mask=[]):
            """perform single excitation for all electrons in all virtual orbitals that
            are not masked"""
            n_elec = len(occupied)
            n_virt = len(virtual)
            
            # initialize mask lists if lists are empty
            if not occ_mask:
                occ_mask = [True for _ in range(n_elec)]
            if not virt_mask:
                virt_mask = [True for _ in range(n_virt)]
            virt_mask_save = virt_mask.copy()
            occ_mask_save = occ_mask.copy()

            for i in range(n_elec):
                for a in range(n_virt):
                    # check if i and a have the same sign (not spin forbidden)
                    #       if occupied lower than virtual
                    #       if electron and virtual orbital have already been changed by  
                    #           previous excitation
                    is_spin_allowed = occupied[i] * virtual[a] > 0
                    is_excitation = abs(occupied[i]) < abs(virtual[a])
                    not_touched = all([occ_mask[i], virt_mask[a]])
                    #
                    if is_spin_allowed and is_excitation and not_touched:
                        occupied_tmp = occupied.copy()
                        virtual_tmp = virtual.copy() 
                        occupied_tmp[i], virtual_tmp[a] = virtual[a], occupied[i]
                        if n_fold_excitation == 1:
                            occupied_tmp = sorted(occupied_tmp,key=self.custom_sort)
                            excited_determinants.append(occupied_tmp)
                        else:
                            # mask already excited electron in occupied and already occupied orbital in virtual
                            occ_mask[i] = False
                            virt_mask[a] = False
                            get_n_fold_excitation(occupied_tmp, virtual_tmp, n_fold_excitation-1, occ_mask, virt_mask)
                            occ_mask = occ_mask_save.copy()
                            virt_mask = virt_mask_save.copy()
                    else:
                        continue
        
        
        # get all demanded excited determinants recursivly
        for excitation in excitations:
            occ_mask_ini = occ_mask.copy()
            virt_mask_ini = virt_mask.copy()
            get_n_fold_excitation(det_ini, virtuals, excitation, occ_mask=occ_mask_ini, virt_mask=virt_mask_ini)
        # remove duplicates
        excited_determinants = self.spinfuncs.remove_duplicates(excited_determinants)
        # remove spin forbidden ones
        if consider_symmetry:
            temp = []
            for determinant in excited_determinants:
                symm = self.get_determinant_symmetry(determinant, orbital_symmetry, tot_sym)
                if symm == symm_of_det_ini:
                    temp.append(determinant)
            excited_determinants = temp

        # add initial determinant, of which excitations have been performed
        res = excited_determinants
        return res
        
        
    def get_unique_csfs(self, determinant_basis, S, M_s):
        """clean determinant basis to obtain unique determinants to construct same csf only once"""
        csf_determinants = []
        csf_coefficients = []
        N = len(determinant_basis[0])
        # TODO sort determinants in determinant basis
        for i in range(len(determinant_basis)):
            determinant_basis[i] = sorted(determinant_basis[i],key=self.custom_sort)
        # remove spin information and consider only occupation
        for i, det in enumerate(determinant_basis):
            for j, orbital in enumerate(det):
                determinant_basis[i][j] = abs(orbital)
        # keep only unique determinants
        # det_basis_temp = [list(t) for t in set(tuple(x) for x in determinant_basis)]
        seen = set()
        det_basis_temp = []
        for sublist in determinant_basis:
            tuple_sublist = tuple(sublist)
            if tuple_sublist not in seen:
                seen.add(tuple_sublist)
                det_basis_temp.append(sublist)

        det_basis = []
        # move single SD's that are singulett spin eigenfunctions directly in list csfs
        # and move all other determinants in det_basis
        for det in det_basis_temp:
            if self.is_singulett(det):
                # add spin again 
                det = [electron*(-1)**n for n, electron in enumerate(det)]
                csf_determinants.append([det])
                csf_coefficients.append([1.])
            else:
                det_basis.append(det)

        # add spin for singletts
        # check which electrons are in a double occupied orbital and mask them
        masked_electrons = []
        for determinant in  det_basis:
            mask = []
            for electron in determinant:   
                if determinant.count(electron)== 2:
                    mask.append(False)
                else:
                    mask.append(True)
            masked_electrons.append(mask)

        # generate csf from unique determinant
        for determinant, mask in  zip(det_basis, masked_electrons):
            n_uncoupled = sum(mask)
            # get spin eigenfunctions for corresponding determinant
            geneological_path, primitive_spin_summands, coupling_coefficients = self.spinfuncs.get_all_csfs(n_uncoupled,S,M_s)
            # form correct determinants 
            # assign psimitive spin to orbitals by element wise multiplication 
            for i, lin_combination in enumerate(primitive_spin_summands):
                csf_tmp = []
                for primitive in lin_combination:
                    idx_primitive = 0
                    idx_singlet_electron = 0
                    det_tmp = []
                    for j, electron in enumerate(determinant):
                        if mask[j]:
                            det_tmp.append(electron*primitive[idx_primitive])
                            idx_primitive +=1
                        else:
                            det_tmp.append(electron*(-1)**idx_singlet_electron)
                            idx_singlet_electron += 1
                    csf_tmp.append(det_tmp)
                csf_determinants.append(csf_tmp)
                csf_coefficients.append(coupling_coefficients[i])
        return csf_coefficients, csf_determinants
    
    def get_initial_wf(self, S, M_s, n_MO, initial_determinant, excitations, orbital_symmetry, total_symmetry, frozen_elecs, frozen_MOs,filename,split_at=0,verbose=False):
        """get initial wave function for selected Configuration Interaction in Amolqc format."""
        N = len(initial_determinant)
        determinant_basis = []
        
        # get excitation determinants from ground state HF determinant
        excited_determinants = self.get_excitations(n_MO, excitations, initial_determinant, orbital_symmetry=orbital_symmetry, tot_sym=total_symmetry,core=frozen_elecs,frozen_MOs=frozen_MOs)
        determinant_basis += [initial_determinant]
        determinant_basis += excited_determinants
        if verbose:
            print(f"number of determinant basis: {len(determinant_basis)}")
            print()

        # form csfs from determinants in determinant basis
        csf_coefficients, csfs = self.get_unique_csfs(determinant_basis, S, M_s) 
        if verbose:
            print(f"number of csfs {len(csf_coefficients)}")
            print()

        # sort determinants to obtain AMOLQC format
        csf_coefficients, csfs = self.sort_determinants_in_csfs(csf_coefficients, csfs) 
        
        # generate MO initial list
        CI_coefficients = [1 if n == 0 else 0 for n in range(len(csfs))]
       
        # read wave function pretext from already generated wavefunction
        wfpretext = ""
        try:
            _, _, _, wfpretext = self.read_AMOLQC_csfs(f"{filename}.wf", N) 
        except:
            FileNotFoundError
        if split_at>0:
            # prints csfs inlcusive the indice of split at in first wf and residual in second
            self.write_AMOLQC(csf_coefficients[:split_at], csfs[:split_at], CI_coefficients[:split_at:],pretext=wfpretext,file_name=f"{filename}_out.wf")   
            self.write_AMOLQC(csf_coefficients[split_at:], csfs[split_at:], CI_coefficients[split_at:],file_name=f"{filename}_res.wf")   
            if verbose:
                print(f"number of csfs in wf 1: {len(csf_coefficients[:split_at])}")
                print(f"number of csfs in wf 2: {len(csf_coefficients[split_at:])}")
                print()
        else:
            # write wavefunction in AMOLQC format
            self.write_AMOLQC(csf_coefficients, csfs, CI_coefficients, pretext=wfpretext, file_name=f"{filename}_out.wf") 
        

    def select_and_do_excitations(
            self, N: int, n_MO: int, S: float, M_s: float, reference_determinant: list, excitations: list, orbital_symmetry: list, 
            total_symmetry: str, frozen_elecs: list, frozen_MOs: list, input_wf: str, 
            CI_coefficient_thresh: float, split_at=0, use_optimized_CI_coeffs = True, verbose=False
                                ):
        """select csfs by size of their coefficients and do n-fold excitations of determinants in selected csfs."""
        # TODO add n_min as second criterion beside the selection on threshold size
        # read wavefunction with optimized CI coefficients
        csf_coefficients, csfs, CI_coefficients, wfpretext = self.read_AMOLQC_csfs(f"{input_wf}.wf", N) 
        print(f"number of initial csfs from input wavefunction {len(csfs)}")

        # cut of csfs by CI coefficient threshold
        csf_coefficients_selected, csfs_selected, CI_coefficients_selected, csf_coefficients_discarded, csfs_discarded, CI_coefficients_discarded \
        = self.cut_csfs(csf_coefficients, csfs, CI_coefficients, CI_coefficient_thresh) 
        # write file with selected csfs
        self.write_AMOLQC(csf_coefficients_selected, csfs_selected, CI_coefficients_selected, file_name=f"{input_wf}_selected.wf") 

        # # expand cut csfs in determinants 
        _, _, determinant_basis_discarded = self.get_transformation_matrix(csf_coefficients_discarded, csfs_discarded, CI_coefficients_discarded)

        # expand selected csfs in determinants 
        _, _, determinant_basis_selected = self.get_transformation_matrix(csf_coefficients_selected, csfs_selected, CI_coefficients_selected)
        determinants_already_visited = determinant_basis_selected + determinant_basis_discarded

        # do exitations from selected determinants. only excite electrons that have not yet been excited
        # in respect to the reference determinant (initial input determinant)
        excited_determinants = []
        for det in determinant_basis_selected:
            determinants = self.get_excitations(n_MO,excitations,det, \
                    det_reference=reference_determinant,orbital_symmetry=orbital_symmetry, tot_sym=total_symmetry,core=frozen_elecs, frozen_MOs=frozen_MOs)
            excited_determinants += determinants
        excited_determinants = self.spinfuncs.remove_duplicates(excited_determinants)

        # remove determinants that have already been visited and are found in the input wave function
        seen = set()
        for det in determinants_already_visited:
            det = sorted(det,key=self.custom_sort)
            seen.add(tuple(det))
        res = []
        for det in excited_determinants:
            # Convert sublist to tuple 
            det = sorted(det,key=self.custom_sort)
            det_tuple = tuple(det)
            #if 1 in det and 2 in det and 4 in det and 5 in det and 8 in det and -1 in det and -2 in det and -4 in det and -5 in det and -8 in det:
            #    print(det)
            if det_tuple not in seen:
                res.append(det)  
                seen.add(det_tuple) 
        excited_determinants = res
        if verbose:
            print(f"number determinants to form csfs: {len(excited_determinants)}")
        # form csfs of these determinants
        csf_coefficients, csfs = self.get_unique_csfs(excited_determinants, S, M_s) 
        csf_coefficients, csfs = self.sort_determinants_in_csfs(csf_coefficients, csfs)
        if verbose:
            print(f"number of newly generated csfs: {len(csf_coefficients)}")
        # generate MO initial list for new csfs and optional for selected csfs 
        if not use_optimized_CI_coeffs:
            CI_coefficients_selected = [1 if n == 0 else 0 for n in range(len(csfs_selected))]
        CI_coefficients = [0 for _ in range(len(csfs))]
        # combine new csfs with selected csfs
        csfs = csfs_selected + csfs
        csf_coefficients = csf_coefficients_selected + csf_coefficients
        CI_coefficients = CI_coefficients_selected + CI_coefficients
        if verbose:
            print(f"total number of csfs after adding selected ones {len(csf_coefficients)}")
        # write wavefunction in AMOLQC format
        if split_at>0:
            # prints csfs inlcusive the indice of split at in first wf and residual in second
            self.write_AMOLQC(csf_coefficients[:split_at], csfs[:split_at], CI_coefficients[:split_at:],pretext=wfpretext,file_name=f"{input_wf}_out.wf")   
            self.write_AMOLQC(csf_coefficients[split_at:], csfs[split_at:], CI_coefficients[split_at:],file_name=f"{nput_wf}_residual.wf")   
            if verbose:
                print(f"number of csfs in next iteration wf: {len(csf_coefficients[:split_at])}")
                print(f"number of csfs in residual wf: {len(csf_coefficients[split_at:])}")
                print()
        else:
            # write wavefunction in AMOLQC format
            self.write_AMOLQC(csf_coefficients, csfs, CI_coefficients, pretext=wfpretext, file_name=f"{input_wf}_out.wf") 
    
    def select_and_do_next_package(self, N, filename_discarded_all, filename_optimized, filename_residual, CI_coefficient_thresh,split_at=0, n_min=0, verbose=False):
        """select csfs by size of their coefficients and add next package of already generated csfs."""
        # read in all three csf files with already discarded csfs, not-yet-selected csfs and not-yet-optimized csfs
        no_file_for_discarded = False

        try:
            csf_coefficients_discarded_all, csfs_discarded_all, CI_coefficients_discarded_all, _ = self.read_AMOLQC_csfs(f"{filename_discarded_all}.wf", N)    
            csf_coefficients_discarded_all, csfs_discarded_all, CI_coefficients_discarded_all , _, _, _ \
        = self.cut_csfs(csf_coefficients_discarded_all, csfs_discarded_all, CI_coefficients_discarded_all, 0.0) 
        except:
            FileNotFoundError
            csf_coefficients_discarded_all, csfs_discarded_all, CI_coefficients_discarded_all = [],[],[]
            print(f"no file {filename_discarded_all}.wf . Thus it is going to be generated for this selection.")
            #no_file_for_discarded = True

        csf_coefficients_optimized, csfs_optimized, CI_coefficients_optimized, wfpretext = self.read_AMOLQC_csfs(f"{filename_optimized}.wf", N) 
        try:
            csf_coefficients_residual, csfs_residual, CI_coefficients_residual, _ = self.read_AMOLQC_csfs(f"{filename_residual}.wf", N) 
        except:
            FileNotFoundError
            csf_coefficients_residual, csfs_residual, CI_coefficients_residual = [],[],[]
            print(f"no file {filename_residual}.wf . Thus it is going to ignored.")
        #
        csf_coefficients_selected, csfs_selected, CI_coefficients_selected, csf_coefficients_discarded, csfs_discarded, CI_coefficients_discarded \
        = self.cut_csfs(csf_coefficients_optimized, csfs_optimized, CI_coefficients_optimized, CI_coefficient_thresh) 

        # take n_min number of csfs by largest CI coefficients
        if len(csfs_selected) < n_min:
            csf_coefficients_selected, csfs_selected, CI_coefficients_selected, _, _, _ \
        = self.cut_csfs(csf_coefficients_optimized, csfs_optimized, CI_coefficients_optimized, 0.0) 
            csf_coefficients_discarded, csfs_discarded, CI_coefficients_discarded = csf_coefficients_selected[n_min:], csfs_selected[n_min:], CI_coefficients_selected[n_min:]
            csf_coefficients_selected, csfs_selected, CI_coefficients_selected = csf_coefficients_selected[:n_min], csfs_selected[:n_min], CI_coefficients_selected[:n_min]
        print(f"number of selected csfs {len(csfs_selected)}")
        # full wavefunction without already discarded csfs
        csf_coefficients = csf_coefficients_selected + csf_coefficients_residual
        csfs = csfs_selected + csfs_residual
        CI_coefficients = CI_coefficients_selected + CI_coefficients_residual
        csf_coefficients, csfs = self.sort_determinants_in_csfs(csf_coefficients, csfs)

        # all discarded csfs
        csf_coefficients_discarded_all += csf_coefficients_discarded
        csfs_discarded_all += csfs_discarded
        CI_coefficients_discarded_all += CI_coefficients_discarded

        #print wavefunctions
        if split_at>0:
            # prints csfs inlcusive the indice of split at in first wf and residual in second
            self.write_AMOLQC(csf_coefficients[:split_at], csfs[:split_at], CI_coefficients[:split_at:],pretext=wfpretext,file_name=f"{filename_optimized}_out.wf")   
            self.write_AMOLQC(csf_coefficients[split_at:], csfs[split_at:], CI_coefficients[split_at:],file_name=f"{filename_residual}_out.wf")   
            if verbose:
                print(f"number of csfs in next iteration wf: {len(csf_coefficients[:split_at])}")
                print(f"number of csfs in residual wf: {len(csf_coefficients[split_at:])}")
                print()
        else:
            # write wavefunction in AMOLQC format
            self.write_AMOLQC(csf_coefficients, csfs, CI_coefficients, pretext=wfpretext, file_name=f"{filename_optimized}_out.wf")  
        self.write_AMOLQC(csf_coefficients_discarded_all, csfs_discarded_all, CI_coefficients_discarded_all, file_name=f"{filename_discarded_all}_out.wf")  
        
        # print info file
        with open("info.txt", 'w') as reffile:
            reffile.write(f"""selected csfs:\t{len(csfs_selected)}
threshold:\t{CI_coefficient_thresh}
number of csfs in next iteration wf:\t{len(csf_coefficients[:split_at])}
number of csfs in residual wf:\t{len(csf_coefficients[split_at:])}
 """)

    def plot_ci_coefficients(self,wavefunction_name, n_elec):
        """plot ci coefficients of wavefunction"""
        import matplotlib.pyplot as plt

        csf_coefficients, csfs, CI_coefficients, _ = self.read_AMOLQC_csfs(f"{wavefunction_name}.wf", n_elec) 
        # sort CI coefficients from largest to smallest absolut value and respectively csfs and csf_coefficients
        CI_coefficients_abs = -np.abs(np.array(CI_coefficients))
        idx = CI_coefficients_abs.argsort()

        CI_coefficients = [abs(CI_coefficients[i]) for i in idx]
        csf_coefficients = [csf_coefficients[i] for i in idx]
        csfs = [csfs[i] for i in idx]

        #plot ci coefficients
        plt.rcParams['figure.figsize'] = (10, 6.5)
        plt.rcParams['font.size'] = 16
        plt.rcParams['lines.linewidth'] = 3
        #plt.rcParams['font.family'] = 'Avenir'
        plt.rcParams['mathtext.fontset'] =  'dejavuserif'
        plt.rc ('axes', titlesize = 16)
        plt.rc ('axes', labelsize = 16)
        plt.rc('axes', linewidth=2)
        color = []
        ref_state = self.build_energy_lowest_detetminant(n_elec)
        # color ci points by respective excitation
        for csf in csfs:
            difference = 0
            for electron in csf[0]:
                if electron not in ref_state:
                    difference +=1
            if difference == 1:
                color.append('darkred')
            elif difference == 2:
                color.append('grey')
            elif difference == 3:
                color.append('teal')
            elif difference == 4:
                color.append('orange')
            elif difference == 5:
                color.append('blue')
            elif difference == 6:
                color.append('lime')
            else:
                color.append('fuchsia')

        plt.xlabel('determinant idx [ ]', labelpad = 15)
        plt.ylabel('CI coefficients [ ]', labelpad = 15)
        plt.scatter(
            range(len(CI_coefficients)-1),CI_coefficients[1:], 
            s=15, marker=".", color=color[1:], label="CI coefficients"
            )
        plt.legend( loc='upper right',)
        plt.savefig(f'sCI_{wavefunction_name}.png', format = 'png', bbox_inches='tight', dpi = 450)

if __name__ == "__main__":

    sCI = SelectedCI()
    spinfuncs = SpinCoupling()

    #csf_coefficients, csfs, CI_coefficients = sCI.read_AMOLQC_csfs(f"block_final.wf", 10) 
    #csf_coefficients, csfs, CI_coefficients, _, _, _ \
    #= sCI.cut_csfs(csf_coefficients, csfs, CI_coefficients, 0.01) 
    #sCI.write_AMOLQC(csf_coefficients, csfs, CI_coefficients, file_name=f"all_discarded_sort.wf")  
    #exit(1)

    print("Script to obtain selected configuration interaction (sCI) wave functions in Quantum Monte Carlo.")
    print("Select one from the following options by entering the respective integer:")
    print(" 1\tgenerate initial wave function")
    print(" 2\tgenerate next sCI iteration by excitations on determinants in selected csfs.")
    print(" 3\tgenerate next blocked wavefunction within one iteration.")
    print(" 4\tplot wave function coefficients.")
    print(" 5\trandomize order of csfs in wavefunction.")
    print()
    wavefunction_choice = int(input())

    if wavefunction_choice == 4:
        wavefunction_name = input("enter wave function name (without .wf).\n")
        n_elec = int(input("number of electrons\n"))
        sCI.plot_ci_coefficients(wavefunction_name,n_elec)
        exit(1)
    elif wavefunction_choice == 5:
        wavefunction_name = input("enter wave function name (without .wf).\n")
        n_elec = int(input("number of electrons\n"))
        csf_coefficients, csfs, CI_coefficients, _ = sCI.read_AMOLQC_csfs(f"{wavefunction_name}.wf", n_elec) 
        indices = list(range(1,len(csf_coefficients)))
        random.shuffle(indices)
        indices = [0] + indices
        # resort 
        csf_coefficients_shuffel = [csf_coefficients[i] for i in indices]
        csfs_shuffel = [csfs[i] for i in indices]
        CI_coefficients_shuffel = [CI_coefficients[i] for i in indices]
        # write shuffled wf
        sCI.write_AMOLQC(csf_coefficients_shuffel, csfs_shuffel, CI_coefficients_shuffel, file_name=f"{wavefunction_name}_shuffel.wf")  

        #print(indices)
        #print(len(indices))
        exit(1)
        
        
        
    
    print("Choose molecule in DZae basis that are currently available.")
    print(" 1\twater")
    print(" 2\tethene")
    molecule_choice = int(input())

    print("split wavefunction in two parts. size of first wave function part: (enter 0 to take full wave function)")
    split_at = int(input())

    # set molecule quantities
    if molecule_choice == 1:
        # water
        N = 10
        n_MO = 14
        S = 0
        M_s = 0
        frozen_elecs = [1,-1]
        orbital_symmetry = ['A1', 'A1', 'B2', 'A1', 'B1', 'A1', 'B2', 'B2', 'A1', 'B1', 'A1', 'B2', 'A1', 'A1'] #H2O DZAE
        total_symmetry = "c2v"

    elif molecule_choice == 2:
        # ethene
        N = 16
        n_MO = 28
        S = 0
        M_s = 0
        frozen_elecs = [1,-1,2,-2]
        orbital_symmetry = [
            'Ag','B1u','Ag','B1u','B2u','Ag','B3g','B3u','B2g','Ag','B1u','B2u','B3g',
            'B1u','B2u','B3u','Ag','Ag','B1u','B2g','Ag','B3g','B2u','B1u','B1u','B3g',
            'Ag','B1u',
        ] # ethene DZAE
        total_symmetry = "d2h"

    #orbital_symmetry = ['Ag', 'B1u', 'Ag', 'B2u', 'B3u', 'B1u', 'Ag', 'B2g', 'B3g', 'Ag', 'B1u', 'B1u'] # H2 TZPAE
    #orbital_symmetry =['Ag', 'B1u', 'Ag', 'B1u', 'B3u', 'B2u', 'Ag', 'B3g', 'B2g', 'B1u', 'B1u', 'B2u', 'Ag', 'B3g', 'B2g', 'Ag', 'B1u', 'B1g'] # N2 PBE0 TZPAE
   # orbital_symmetry = ['A1', 'A1', 'B2', 'A1', 'B1', 'A1', 'B2', 'A1', 'B2', 'A1', 'B1', 'A1', 'B2', 'A2', 
   #                     'B1', 'A1', 'B2', 'B2', 'A1', 'B1', 'A2', 'A1', 'A1', 'A1', 'B2', 'B2', 'B1', 'A1',
   #                     'B2', 'A1', 'A1'
   #                     ] #H2O TZPAE
   # orbital_symmetry = ['A1', 'A1', 'B2', 'A1', 'B1', 'A1', 'B2', 'B2', 'A1', 'B1', 'A1', 'B2', 'A1', 'A1'] #H2O DZAE
   # orbital_symmetry = []

    # get HF determinant (energy lowest determinant)
    initial_determinant = sCI.build_energy_lowest_detetminant(N)


    if wavefunction_choice == 1:
        wavefunction_name = input("enter wave function name (without .wf).\n")
        sCI.get_initial_wf(S, n_MO, initial_determinant,[1,2], orbital_symmetry, total_symmetry,frozen_elecs,[],wavefunction_name,split_at=split_at,verbose = True)
    elif wavefunction_choice == 2:
        wavefunction_name = input("enter wave function name (without .wf).\n")
        CI_coefficient_thresh = float(input("enter CI coefficient threshold to select csfs."))
        sCI.select_and_do_excitations(N,n_MO,S,M_s,initial_determinant,[1,2],orbital_symmetry,total_symmetry,
                                      frozen_elecs,[],wavefunction_name,CI_coefficient_thresh,split_at=split_at,verbose=True)
    elif wavefunction_choice == 3:
        wavefunction_name = input("enter wave function name (without .wf).\n")
        CI_coefficient_thresh = float(input("enter CI coefficient threshold to select csfs.\n"))
        n_min = int(input("enter minimum number of csfs that shall be selected.\n"))
        sCI.select_and_do_next_package("discarded", wavefunction_name, "residual", CI_coefficient_thresh, split_at=split_at,n_min=n_min, verbose=True)


    ######
    # NITROGEN 2
    ######
    #orbital_symmetry= ['A1g', 'A2u', 'A1g', 'A2u',  'Eu', 'Eu', 'A1g', 'Eg', 'Eg', 'A2u', 'Eu', 'Eu', 'A1g', 'A1g', 'Eg', 'Eg', 'A2u', 'A2u', 'A1g', 'A2u' ] # N2 PBE0 DZAE d4h 
    #sCI.get_initial_wf(S, n_MO, initial_determinant,[1,2], orbital_symmetry, "d4h",[1,-1,2,-2],[15,-15,16,-16,17,-17,18,-18,19,-19,20,-20],verbose = True)
    
    
    #print(csfs)
    ########################################
    # Section to prepare next iteration 
    ########################################
    if False:
        N = 10
        n_MO = 14
        S = 0
        M_s = 0
        reference_determinant = [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]
        orbital_symmetry = ['A1', 'A1', 'B2', 'A1', 'B1', 'A1', 'B2', 'B2', 'A1', 'B1', 'A1', 'B2', 'A1', 'A1'] #H2O DZAE
        total_symmetry = "c2v"
        split_at = 250
        sCI.select_and_do_excitations(N,n_MO,S,M_s,reference_determinant,[1,2],orbital_symmetry,total_symmetry,
                                      [1,-1],[],"sCI/fin_1-1_000",0.01,split_at=150,verbose=True)
    
    ########################################
    # Section to obtain next package of csfs in one iteration 
    ########################################
    if False:
        N = 10 #
        n_MO = 14
        S = 0
        M_s = 0
        reference_determinant = [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]
        orbital_symmetry = ['A1', 'A1', 'B2', 'A1', 'B1', 'A1', 'B2', 'B2', 'A1', 'B1', 'A1', 'B2', 'A1', 'A1'] #H2O DZAE
        total_symmetry = "c2v"
        path = "evaluation/h2o/it_2"
        #optimized = input("optimized wavefunction name.")
        optimized = "amolqc-12"
        sCI.select_and_do_next_package(f"discarded", f"{optimized}", f"residual", 0.005, split_at=150,n_min=30, verbose=True)
