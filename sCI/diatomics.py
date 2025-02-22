#!/usr/bin/env python3

import numpy as np

# from salc import SALC
from csf import SelectedCI


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
        self.Lz2_mat = np.zeros((self.dim_tuple, self.dim_tuple))

    def tensorProd(self, tup):
        """Return tensor product of matrices in tuple"""
        res = tup[0]
        for mat in tup[1:]:
            res = np.kron(res, mat)
        return res

    def update_Lz2(self):
        """"""
        self.Lz2_mat = np.zeros((self.dim_tuple, self.dim_tuple))

    def test(self):
        """ """
        n_elec = 4
        self.dim_basis = len(self.basis)
        self.dim_tuple = n_elec ** self.dim_basis

        productBasis = self.get_product_basis(n_elec)

        example = self.apply_lz2(productBasis[0])
        self.get_lz_sq_matrix()

        print(example)

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
        """Return results of Lz^2 operator acting on basis functions."""
        self.update_Lz2()

    def get_product_basis(self, n_elec):
        """
        Return all basis function combinations of product basis functions.
        """
        all_combinations = []
        print(self.dim_tuple)
        for n in range(self.dim_tuple):
            combination = []
            # set start index for function from basis functions
            i = n
            # get one tuple of basis functions. go from largest to smallest to
            # get the correct order of basis functions ("lexicographical order")
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

diatom = Diatomic()
diatom.test()
