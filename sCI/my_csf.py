import numpy as np
from fractions import Fraction
from charactertables import CharacterTable
from spincoupling import SpinCoupling


class SelectedCI():
    """generate excited determinants"""
    def __init__(self):
        self.spinfuncs = SpinCoupling()
    
    def custom_sort(self,x):
        return (abs(x), x < 0)
    
    def write_AMOLQC(self, csf_coefficients, csfs, CI_coefficients, write_file = True, verbose=False):
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
        if verbose:
            print(out)
        if write_file:
            with open("sCI/csfs.out", "w") as printfile:
                printfile.write(out)

    def read_AMOLQC_csfs(self, filename, n_elec, type="csf"):
        """read in csfs of AMOLQC format with CI coefficients"""
        csf_coefficients = []
        csfs = []
        CI_coefficients = []
        csf_tmp = []
        csf_coefficient_tmp = []
        # read csfs
        if type == "csf":
            with open(f"sCI/{filename}", "r") as f:
                found_csf = False
                for line in f:
                    if "$csfs" in line:
                        found_csf = True
                        new_csf = True
                        # extract number of csfs
                        line = f.readline()
                        n_csfs = int(line)
                        line = f.readline()
                        # initialize counter to iterate over csfs
                        csf_counter = 0
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
        return csf_coefficients, csfs, CI_coefficients
    

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
    

    def get_excitations(self, n_orbitals, excitations, det_ini, orbital_symmetry=[], tot_sym="",det_reference=[], core=[]):
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
    
    def get_initial_wf(self, S, n_MO, initial_determinant, excitations, orbital_symmetry, total_symmetry, frozen_elecs, verbose=False):
        """get initial wave function for selected Configuration Interaction in Amolqc format."""
        determinant_basis = []
        
        # get excitation determinants from ground state HF determinant
        excited_determinants = self.get_excitations(n_MO, excitations, initial_determinant, orbital_symmetry=orbital_symmetry, tot_sym=total_symmetry,core=frozen_elecs)
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

        # write wavefunction in AMOLQC format
        self.write_AMOLQC(csf_coefficients, csfs, CI_coefficients) 

if __name__ == "__main__":
    # set quantities
    N = 10
    n_MO = 14
    S = 0
    M_s = 0
    CI_coefficient_thresh = 1e-2

    #orbital_symmetry = ['Ag', 'B1u', 'Ag', 'B2u', 'B3u', 'B1u', 'Ag', 'B2g', 'B3g', 'Ag', 'B1u', 'B1u'] # H2 TZPAE
    #orbital_symmetry =['Ag', 'B1u', 'Ag', 'B1u', 'B3u', 'B2u', 'Ag', 'B3g', 'B2g', 'B1u', 'B1u', 'B2u', 'Ag', 'B3g', 'B2g', 'Ag', 'B1u', 'B1g'] # N2 PBE0 TZPAE
    orbital_symmetry = ['A1', 'A1', 'B2', 'A1', 'B1', 'A1', 'B2', 'B2', 'A1', 'B1', 'A1', 'B2', 'A1', 'A1'] #H2O DZAE
    total_symmetry = "c2v"
    #orbital_symmetry =[]
    frozen_elecs = [1,-1]

    # call own implementation
    
    sCI = SelectedCI()
    spinfuncs = SpinCoupling()

    # get HF determinant (energy lowest determinant)
    initial_determinant = sCI.build_energy_lowest_detetminant(N)

    # get initial wavefunction 
    # TODO can be switched on sCI.get_initial_wf(S, n_MO, initial_determinant,[1,2], orbital_symmetry, total_symmetry,frozen_elecs,verbose = True)

    #print(csfs)
    ########################################
    # perform CI and Jas optimization
    ########################################

    # read wavefunction from 1. optimization
    csf_coefficients, csfs, CI_coefficients = sCI.read_AMOLQC_csfs("fin_1-1.wf", N) 
    print(f"number of initial csfs from 1. iteration {len(csfs)}")
    # cut of csfs
    cut_CI_coefficients = []
    cut_csf_coefficients = []
    cut_csfs = []
    csf_coefficients_old, csfs_old, CI_coefficients_old, cut_csf_coeffs_tmp, cut_csfs_tmp, cut_CI_coeffs_tmp \
    = sCI.cut_csfs(csf_coefficients, csfs, CI_coefficients, CI_coefficient_thresh) 
    
    #csf_coefficients, csfs = sCI.sort_determinants_in_csfs(csf_coefficients_old, csfs_old)
    #CI_coefficients = [0 for _ in range(len(csfs))]
    #sCI.write_AMOLQC(csf_coefficients, csfs, CI_coefficients) 
    #exit()
    # expand csfs in determinants 
    CI_mat, trans_mat, determinant_basis = sCI.get_transformation_matrix(csf_coefficients_old, csfs_old, CI_coefficients_old)
    excitations = []
    det_reference = [1, 2, 3, 4, 5, -1, -2, -3, -4, -5]
    for det in determinant_basis:
        determinants = sCI.get_excitations(n_MO,[1,2],det, \
                det_reference=det_reference,orbital_symmetry=orbital_symmetry, tot_sym=total_symmetry,core=frozen_elecs)
        excitations += determinants

   
    excitations = spinfuncs.remove_duplicates(excitations)

    print(f"all determinants for second step {len(excitations)}")


    # form csfs of this determinants
    csf_coefficients, csfs = sCI.get_unique_csfs(excitations, S, M_s) 
    csf_coefficients, csfs = sCI.sort_determinants_in_csfs(csf_coefficients, csfs)
    
    print(f"number of csfs {len(csf_coefficients)}")
    # generate MO initial list
    CI_coefficients = [0 for _ in range(len(csfs))]
    csfs = csfs_old + csfs
    csf_coefficients = csf_coefficients_old + csf_coefficients
    CI_coefficients = CI_coefficients_old + CI_coefficients
    print(f"number of csfs after adding old ones {len(csf_coefficients)}")
    # keep old coefficients as guess
    
    # write wavefunction in AMOLQC format
    sCI.write_AMOLQC(csf_coefficients, csfs, CI_coefficients) 
    exit()
    #print("After cutting")
    #print(determinants)

    # perform single and double excitations of determinants


    #CI_coefficient_matrix, transformation_matrix, det_basis = determinant.get_transformation_matrix(csf_coefficients, csfs, CI_coefficients) # TODO write test
    
