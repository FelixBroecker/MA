import numpy as np
from genealogical import create_genealogical_spin_functions, printX
from fractions import Fraction
from charactertables import CharacterTable
from spincoupling import SpinCoupling


class generate_determinants():
    """generate excited determinants"""
    def __init__(self):
        self.spinfuncs = SpinCoupling()
    
    def custom_sort(self,x):
        return (abs(x), x < 0)
    
    def write_AMOLQC(self, csf_coefficients, csfs, MO_coefficients, write_file = True, verbose=False):
        """determinant representation in csfs needs to be sorted for alpha spins first
        and then beta spins"""
        out="$csfs\n"
        out += f"{int(len(csfs)): >7}\n"
        for i, csf in enumerate(csfs):
            out += f"{MO_coefficients[i]: >10.7f}       {len(csf)}\n"
            for j, determinant in enumerate(csf):
                out += f" {csf_coefficients[i][j]: 9.7f}"
                for electron in determinant:
                    out += f"  {abs(electron)}"
                out += "\n"
        out += "$end"
        if verbose:
            print(out)
        if write_file:
            with open("./csfs.out", "w") as printfile:
                printfile.write(out)

    def read_AMOLQC_csfs(self, filename, n_elec, type="csf"):
        """read in csfs of AMOLQC format with CI coefficients"""
        csf_coefficients = []
        csfs = []
        MO_coefficients = []
        csf_tmp = []
        csf_coefficient_tmp = []
        # read csfs
        if type == "csf":
            with open("amolqc.wf", "r") as f:
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
                            MO_coefficients.append(float(entries[0]))
                            n_summands = int(entries[1])
                            summand_counter = 0
                            new_csf = False
                            csf_counter +=1
                        else:
                            det = []
                            for i in range(1,len(entries)):
                                if i < n_elec:
                                    det.append(i * int(entries[i]))
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
        return csf_coefficients, csfs, MO_coefficients

        
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
    
    def get_excitations(self, n_elecs, n_orbitals, excitations,orbital_symmetry=[], tot_sym="", det_ini=[]):
        """create all excitation determinants"""
        n_doubly_occ = n_elecs // 2

        # create HF determinant, if no initial determinant is passed
        orbital = 1
        if not det_ini:
            if n_elecs > 1:
                for _ in range(n_doubly_occ):
                    det_ini.append((orbital))
                    det_ini.append(-(orbital))
                    orbital += 1
            if n_elecs%2:
                det_ini.append((orbital))

        # all unoccupied MOs are virtual orbitals
        virtuals = [i for i in range(-n_orbitals,n_orbitals+1) if i not in det_ini and i != 0]

        # consider symmetry if symmetry is specified in input
        consider_symmetry = bool(orbital_symmetry)

        # determine symmetry of input determinant
        if consider_symmetry:
            symm_of_det_ini = self.get_determinant_symmetry(det_ini, orbital_symmetry, tot_sym)

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
            get_n_fold_excitation(det_ini, virtuals, excitation)
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
        res = [det_ini] + excited_determinants
        return res
        
        
    def get_unique_csfs(self, determinant_basis, S, M_s):
        """clean determinant basis to obtain unique determinants to construct same csf only once"""
        csf_determinants = []
        csf_coefficients = []
        N = len(determinant_basis[0])

        # remove spin information and consider only occupation
        for i, det in enumerate(determinant_basis):
            for j, orbital in enumerate(det):
                determinant_basis[i][j] = abs(orbital)
        # keep only unique determinants
        #det_basis_temp = [list(t) for t in set(tuple(x) for x in determinant_basis)]
        seen = set()
        det_basis_temp = []
        for sublist in determinant_basis:
            tuple_sublist = tuple(sublist)
            if tuple_sublist not in seen:
                seen.add(tuple_sublist)
                det_basis_temp.append(sublist)
        #print(det_basis_temp)

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


if __name__ == "__main__":
    # set quantities
    N = 10
    n_MO = 14
    S = 0
    M_s = 0
    #orbital_symmetry = ['Ag', 'B1u', 'Ag', 'B2u', 'B3u', 'B1u', 'Ag', 'B2g', 'B3g', 'Ag', 'B1u', 'B1u'] # H2 TZPAE
    #orbital_symmetry =['Ag', 'B1u', 'Ag', 'B1u', 'B3u', 'B2u', 'Ag', 'B3g', 'B2g', 'B1u', 'B1u', 'B2u', 'Ag', 'B3g', 'B2g', 'Ag', 'B1u', 'B1g'] # N2 PBE0 TZPAE
    orbital_symmetry = ['A1', 'A1', 'B2', 'A1', 'B1', 'A1', 'B2', 'B2', 'A1', 'B1', 'A1', 'B2', 'A1', 'A1']
    tot_sym = "c2v"
    #orbital_symmetry =[]

    # call own implementation
    
    determinant = generate_determinants()
    
    n_elec = 2
    csf_coefficients, csfs, MO_coefficients = determinant.read_AMOLQC_csfs("amolqc.wf", n_elec) # TODO write test

    determinants = determinant.get_excitations(N,n_MO,[1,2],orbital_symmetry,tot_sym) # TODO write test
    print(f"number of determinant basis {len(determinants)}")


    csf_coefficients, csfs = determinant.get_unique_csfs(determinants, S, M_s) 

    # sort determinants to obtain AMOLQC format
    csf_coefficients, csfs = determinant.sort_determinants_in_csfs(csf_coefficients, csfs) # TODO write test

    # generate MO initial list
    MO_coefficients = [1 if n == 0 else 0 for n in range(len(csfs))]
    
    determinant.write_AMOLQC(csf_coefficients, csfs, MO_coefficients) 
    print()
    print(f"number of csfs {len(csf_coefficients)}")
