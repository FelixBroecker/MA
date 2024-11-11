
import numpy as np

class SpinCoupling():
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
        #print(f"S {S}\n M {M}  \n sgm {sgm} \n")
        res = (S+2*sgm*M)/(2*S)
        if res<0:
            #print(res)
            print(f"{res} with M {M}, sgm {sgm} S {S}")
        #assert res < 0, "Square root in computation  of Clebsch Gordan Coefficient cannot be zero."
        return np.sqrt(res)

    def C_minus(self,S,M,sgm):
        """Clebsch Gordan coefficients for spin addition"""
        res = (S+1 -2*sgm*M)/(2*(S+1))
        if res< 0:
            print(f"{res} with M {M}, sgm {sgm} S {S}")
        #assert res < 0, "Square root in computation  of Clebsch Gordan Coefficient cannot be zero."
        return -2* sgm * np.sqrt(res)

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