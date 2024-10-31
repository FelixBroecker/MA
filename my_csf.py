import numpy as np
from genealogical import create_genealogical_spin_functions, printX
from fractions import Fraction

class spinfunction():
    """generate configuration state function"""
    def __init__(self):
        pass

    def sign(self,num):
        return -1 if num < 0 else (1 if num > 0 else 1)
    
    def get_permutations(self,x,l,u):
        """compute all permutations from x in range l to u"""
        def sub(i):
            if i==u:
                res.append(x[:])
            else:
                for k in range(i,u):
                    x[i], x[k] = x[k], x[i]
                    sub(i+1)
                    x[i], x[k] = x[k], x[i]
        res = []
        sub(l)
        return res
    
    def remove_duplicates(self,x):
        """remove duplicate entries from list"""
        res = []
        seen = set()
        for sublist in x:
            # Convert sublist to tuple 
            sublist_tuple = tuple(sublist)
            if sublist_tuple not in seen:
                res.append(sublist)  
                seen.add(sublist_tuple) 
        return res
    
    def check_geneological(self,x, major_spin):
        """check if particular path is allowed according to geneological coupling scheme"""
        res = []
        # define criterion for step in genealogical spin depending on major spin
        if major_spin >= 0:
            step_allowed = lambda S: S < 0
        elif major_spin < 0:
            step_allowed = lambda S: S > 0
        # check which steps are allowed
        for path in x:
            S = 0
            for i, step in enumerate(path):
                S += step
                if step_allowed(S):
                    break
                if i == len(path)-1:
                    res.append(path)
        return res
    
    def C_plus(self,S,M,sgm):
        """Clebsch Gordan coefficients for spin addition"""
        return np.sqrt((S+2*sgm*M)/(2*S))

    def C_minus(self,S,M,sgm):
        """Clebsch Gordan coefficients for spin addition"""
        return -2* sgm * np.sqrt((S+1 -2*sgm*M)/(2*(S+1)))

    def proj_prim_spin(self, uncoupled, coupled):
        """compute contibution of single primitive spin function of csf."""
        n_elec = len(coupled)
        # get M_s for uncoupled spin functions
        uncoupled_ms = []
        for i, item in enumerate(uncoupled):
            if i == 0:
                uncoupled_ms.append(item*.5)
            else:
                uncoupled_ms.append(item*.5+uncoupled_ms[i-1])
        # get S for uncoupled spin functions
        coupled_s = []
        for i, item in enumerate(coupled):
            if i == 0:
                coupled_s.append(item*.5)
            else:
                coupled_s.append(item*.5+coupled_s[i-1])
        # compute coefficients
        prod = 1
        ref_uncoupled = 0
        ref_coupled = 0 
        for i in range(n_elec):
            S = coupled_s[i]
            M = uncoupled_ms[i]
            sgm = uncoupled_ms[i] - ref_uncoupled
            t = coupled_s[i] - ref_coupled 
            if t == .5:
                C = self.C_plus(S,M,sgm)
            elif t == -.5:
                C = self.C_minus(S,M,sgm)
            # update quantities
            prod *= C
            ref_coupled = coupled_s[i]
            ref_uncoupled = uncoupled_ms[i]
        return prod
    
    def print_csfs(self, path, primitives, coeffs):
        """print csf output as linearcombinations of primitive spin functions"""
        for i, func in enumerate(path):
            for spin in func:
                if spin == 1:
                    print("/", end="")
                if spin == -1:
                    print("\\", end="")
                else:
                    pass
            print()
            for j, primitive in enumerate(primitives[i]):
                coeff = '{:18.16f}'.format(coeffs[i][j])
                print(f"    {coeff}", end="\t")
                for spin in primitive:
                    if spin == 1:
                        print("a", end="")
                    if spin == -1:
                        print("b", end="")
                print()
        print()

    def get_all_csfs(self, N, S, M_s):
        """return all csfs for a certain S state"""
        # TODO check if input is allowed

        # generate list of involved spins represented by 1 and -1
        spins = []
        n_major = int(S * 2) # number of majority spins
        major_spin = int(self.sign(M_s)) # return alpha if S=0 and eliminate unvalid solution before return
        n_residual = int((N-n_major)/2) # number of residual spin pairs
        for _ in range(n_major):
            spins.append(major_spin)
        for _ in range(n_residual):
            spins.append(+1)
            spins.append(-1)
        assert len(spins) == N, "input state does not exist"

        # get path according to geneological scheme
        perms = self.get_permutations(spins,0,len(spins))
        unique_perms = self.remove_duplicates(perms)
        paths = self.check_geneological(unique_perms,major_spin)
        # get all primitive spin functions 
        spin_basis = [unique_perms]

        # get also primitive spin function basis from lower S
        for i, spin in enumerate(spins):
            if spin==1:
                spins[i] = -1
            if sum(spins)>=0:
                perms = self.get_permutations(spins,0,len(spins))
                unique_perms = self.remove_duplicates(perms)
                spin_basis.append(unique_perms)
            else:
                break

        # get Clebsch Gordan coefficients for each primitive spin function
        # overlapping with configuration state function. loop over all csf paths
        # to cover all possible csf function for the input state
        csf_coefficients = []
        csf_primitives = []
        csf_paths = []
        for primitive_spin in spin_basis:
            #print(f"spinbasis: {primitive_spin[0]}")
            for path in paths:
                coeff_tmp = []
                primitives_tmp = []
                for primitive in primitive_spin:
                    if sum(primitive)* .5 == M_s:
                        return_csf = True
                    else:
                        return_csf = False
                    coeff = self.proj_prim_spin(primitive, path)
                    # append coefficients and spin functions if coefficient not 0  
                    if coeff:
                        coeff_tmp.append(coeff)
                        primitives_tmp.append(primitive)
                # add only to return list, if M_s corresponds to input 
                if return_csf:
                    csf_paths.append(path)
                    csf_coefficients.append(coeff_tmp)
                    csf_primitives.append(primitives_tmp)

        return csf_paths, csf_primitives, csf_coefficients
        
class generate_determinants():
    """generate excited determinants"""
    def __init__(self):
        spinfuncs = spinfunction()
    
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
        
    def get_determinant_symmetry(self, determinant, orbital_symmetry):
        """determine symmetry of total determinant based on occupation of molecular orbitals 
        with certain symmetry"""
        total_symmetric = ['A1g','Ag','A1','A']

        ...

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
    
    def get_excitations(self, n_elecs, n_orbitals, excitations, orbital_symmetry=[], det_ini=[]):
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

        # determine symmetry of ground state
        if consider_symmetry:
            symm = get_determinant_symmetry(det_ini, orbital_symmetry)

        # create list of all virtual orbitals seperatly for beta (minus) and alpha (plus) spin.
        #virtual_plus = [i for i in range(1,n_orbitals+1) if i not in det]
        #virtual_minus = [i for i in range(-1,-(n_orbitals+1), -1) if i not in det]
        #virtual_all = [virtual_minus, virtual_plus]

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
            #print(occ_mask)
            #print(occupied)
            #print(virt_mask)
            #print(virtual)
            #print()
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
        excited_determinants = spinfuncs.remove_duplicates(excited_determinants)
        #for det in excited_determinants:
        #    print(det)
        #print(len(excited_determinants))

        # add initial determinant, of which excitations have been performed
        res = [det_ini] + excited_determinants
        return res

        #if 1 in excitations:
        #    print()
        #    print("single excitations")
        #    print()
        #    # single excitations
        #    for i in range(n_elecs):
        #        # choose virtual orbitals depending on the spin of det[i]
        #        virtual = virtual_all[det[i] > 0]
        #        for a in virtual:
        #            new_det = det.copy()
        #            new_det[i] = a
        #            new_det = sorted(new_det,key=self.custom_sort)
        #            determinants.append(new_det)
        #            print(new_det)
        #    print()
        #        
        #if 2 in excitations:
        #    print()
        #    print("double excitations")
        #    print()
        #    # double excitations
        #    counter = 1
        #    for i in range(n_elecs):
        #        for j in range(i+1,n_elecs):
        #            if det[i] > 0:
        #                for idx_a, a in enumerate(virtual_plus):
        #                    new_det = det.copy()
        #                    new_det[i] = a
        #                    if det[j] > 0:
        #                        for b in virtual_plus[idx_a+1:]:
        #                            new_det_2 = new_det.copy()
        #                            new_det_2[j] = b
        #                            new_det_2 = sorted(new_det_2,key=self.custom_sort)
        #                            determinants.append(new_det_2)
        #                            print(new_det_2)
        #                    if det[j] < 0:
        #                        for b in virtual_minus:
        #                                new_det_2 = new_det.copy()
        #                                new_det_2[j] = b
        #                                new_det_2 = sorted(new_det_2,key=self.custom_sort)
        #                                determinants.append(new_det_2)
        #                                print(new_det_2)
        #            if det[i] < 0:
        #                for idx_a, a in enumerate(virtual_minus):
        #                    new_det = det.copy()
        #                    new_det[i] = a
        #                    if det[j] < 0:
        #                        for b in virtual_minus[idx_a+1:]:
        #                            new_det_2 = new_det.copy()
        #                            new_det_2[j] = b
        #                            new_det_2 = sorted(new_det_2,key=self.custom_sort)
        #                            determinants.append(new_det_2)
        #                            print(new_det_2)
        #                    if det[j] > 0:
        #                        for b in virtual_plus:
        #                            new_det_2 = new_det.copy()
        #                            new_det_2[j] = b
        #                            new_det_2 = sorted(new_det_2,key=self.custom_sort)
        #                            determinants.append(new_det_2)
        #                            print(new_det_2)
        #                            counter +=1
        
        
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
        
        #print()
        #print(f"determinant basis for csf formation:")
        #print(det_basis)

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
        #print()
        #print("masked electrons:")
        #print(masked_electrons)
        #print()
        #print()
        #print("print csf list ")
        #print(csf_determinants)
        #print(csf_coefficients)
        #print()
        #print("start generation of CSFs for each determinant")
        # generate csf from unique determinant
        for determinant, mask in  zip(det_basis, masked_electrons):
            n_uncoupled = sum(mask)
            #print("determinant:")
            #print(determinant)
            # get spin eigenfunctions for corresponding determinant
            geneological_path, primitive_spin_summands, coupling_coefficients = spinfuncs.get_all_csfs(n_uncoupled,S,M_s)
            #print()
            #print("primitives")
            #print(primitive_spin_summands)
            #print("coupling coefficients")
            #print(coupling_coefficients)
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
    N = 2
    n_MO = 12
    S = 0
    M_s = 0
    orbital_symmetry = ['Ag', 'B1u', 'Ag', 'B2u', 'B3u', 'B1u', 'Ag', 'B2g', 'B3g', 'Ag', 'B1u', 'B1u']

    # call own implementation
    spinfuncs = spinfunction()
    #csf_paths, primitive_spin, csfs = spinfuncs.get_all_csfs(N,S,M_s)
    #spinfuncs.print_csfs(csf_paths, primitive_spin, csfs)
    

    # call genealogical script
    #print()
    #print("call genealogical script")
    #print()
    #X = create_genealogical_spin_functions(N)
    #print(X[1][0][0][0])

    #printX(X,N,S,M_s)

    # call excitations
    #print()
    #print("excited determinants")
    determinant = generate_determinants()

    determinants = determinant.get_excitations(N,n_MO,[1,2],orbital_symmetry)
    #print(len(determinants))
    csf_coefficients, csfs = determinant.get_unique_csfs(determinants, S, M_s)

    # sort determinants to obtain AMOLQC format
    csf_coefficients, csfs = determinant.sort_determinants_in_csfs(csf_coefficients, csfs)

    # generate MO initial list
    MO_coefficients = [1 if n == 0 else 0 for n in range(len(csfs))]
    
    #print()
    #print("AMOLQC Output")
    determinant.write_AMOLQC(csf_coefficients, csfs, MO_coefficients)
    #print()
