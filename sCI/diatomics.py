#!/usr/bin/env python3

import numpy as np

# from salc import SALC
from csf import SelectedCI
from salc import SALC


class Diatomic:
    """
    couple angular momentum and generate excited determinants
    for linear diatomic molecules
    """

    def __init__(self):
        """Initialize operators, matrices, basis for px and py orbitals"""
        # transformation matrix from imaginary (p-, p+)
        # to real orbitals (px, py)
        self.U = np.array(
            [
                [-1 / np.sqrt(2), 1 / np.sqrt(2)],
                [1j / np.sqrt(2), +1j / np.sqrt(2)],
            ]
        )
        self.U_inv = np.linalg.inv(self.U)
        self.lz_imaginary = np.array([[1, 0], [0, -1]])
        self.lz = self.U @ self.lz_imaginary @ self.U_inv
        self.pi_x = np.array([[1], [0]])
        self.pi_y = np.array([[0], [1]])
        self.basis = [self.pi_x, self.pi_y]
        self.dim_basis = 0
        self.dim_tuple = 0
        self.product_basis = []

    def tensorProd(self, tup):
        """Return tensor product of matrices in tuple"""
        res = tup[0]
        for mat in tup[1:]:
            res = np.kron(res, mat)
        return res

    def test(self):
        """ """
        n_elec = 2
        self.dim_basis = len(self.basis)
        self.dim_tuple = self.dim_basis**n_elec

        self.product_basis = self.get_product_basis(n_elec)

        Lz_square_mat = self.get_lz_sq_matrix()

        eig, evec = np.linalg.eigh(Lz_square_mat)
        print(eig)
        print(evec)
        evec = evec.T

        salc = SALC("d4h_expanded", [])
        mulliken_lables, linear_combs = self.get_irrep_of_linear_comb(
            salc, evec[0], inversion="_u"
        )
        for i, arr in enumerate(linear_combs):
            if np.any(arr):
                print(mulliken_lables[i])
                print(arr)

        # print(np.kron(self.pi_x, self.pi_y))

    def apply_lz2(self, basis_product):
        """
        Return result of Lz^2 operator acting on basis function, coupling
        the angular momentum of orbitals in same tuple"""
        res = np.zeros((self.dim_tuple, 1), dtype=np.complex128)

        # lz^2 operation for the multiplication of functions from same electron
        for i, func in enumerate(basis_product):

            basis_tmp = basis_product.copy()

            # let lz act on the electron i
            basis_tmp[i] = self.lz @ self.lz @ func

            # get tensor product to obtain the index of new basis product in
            # the product basis. Add up result as the Lz^2 operator is a sum of
            # lz^2 operators acting on each electron
            res += self.tensorProd(basis_tmp)

        # lz^2 operation for the multiplication of functions from mixed
        # electron indices
        for i, func_i in enumerate(basis_product):
            for i_offset, func_j in enumerate(basis_product[i + 1 :]):
                j = i + 1 + i_offset
                basis_tmp = basis_product.copy()
                basis_tmp[i] = self.lz @ func_i
                basis_tmp[j] = self.lz @ func_j
                res += 2 * self.tensorProd(basis_tmp)
        return res

    def get_lz_sq_matrix(self):
        """Return Lz^2 matrix according to <i|Lz^2|j>."""
        lz2Mat = np.zeros((self.dim_tuple, self.dim_tuple))
        for i, func_i in enumerate(self.product_basis):
            for j, func_j in enumerate(self.product_basis):
                res = self.tensorProd(func_i).T @ self.apply_lz2(func_j)
                if np.abs(res.imag) > 1e-14:
                    raise ValueError("complex result")
                lz2Mat[i, j] = np.real(res)

        return lz2Mat

    def get_product_basis(self, n_elec):
        """
        Return all basis function combinations of product basis functions.
        """
        all_combinations = []
        for n in range(self.dim_tuple):
            combination = []
            # set start index for function from basis functions
            i = n
            # get one tuple of basis functions. go from largest to smallest to
            # get the correct order of basis functions
            # ("lexicographical order")
            for _ in range(n_elec, 0, -1):
                # get the right most "least significant digit" in basis of
                # self.dim_basis
                r = i % self.dim_basis
                # put basis function of highest index under constraint i
                # to the front of the combination list
                combination.insert(0, self.basis[r])
                # remove the right most "least significant digit" in basis of
                # self.dim_basis
                i = i // self.dim_basis
            all_combinations.append(combination)
        return all_combinations

    def get_irrep_of_linear_comb(
        self, salc: SALC, linear_combination: list[float], inversion=""
    ):
        """Return all irreducible representations of of the linear
        combintation of productbasis. Apply projection operator
        p = dim/order * sum(character * symmetry_operator * basis)"""

        res_mulliken_labels: list[str] = []
        res_linear_comb: list[list] = []

        for mulliken in salc.characTab.characters:
            mulliken_tmp = np.zeros((self.dim_tuple, 1), dtype=np.complex128)
            for i, contribution in enumerate(linear_combination):
                # if contribution is zero, go to next product with contribution
                if not contribution:
                    continue
                # get result of applying projection operator on basis
                # function product
                res_operation = salc.apply_symmetry_operator_on_product(
                    self.product_basis[i], inversion
                )
                sum_res = np.zeros((self.dim_tuple, 1), dtype=np.complex128)
                for prods, character in zip(
                    res_operation, salc.characTab.characters[mulliken]
                ):
                    sum_res += character * self.tensorProd(prods)
                mulliken_tmp += np.sign(contribution) * sum_res
            res_mulliken_labels.append(mulliken)
            res_linear_comb.append(
                salc.characTab.get_dimension(mulliken)
                / salc.characTab.order
                * mulliken_tmp
            )
        return res_mulliken_labels, res_linear_comb


diatom = Diatomic()
diatom.test()

salc = SALC("d4h_expanded", [])
orb_bas_x = salc.operation_matrices["pi_x_g"]
orb_bas_y = salc.operation_matrices["pi_y_g"]

# print(salc.orbital_basis["pi_y"])
