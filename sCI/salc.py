import numpy as np
from charactertables import CharacterTable


class SALC:
    def __init__(self, point_group: str, basis: list):
        """"""
        self.point_group = point_group
        self.characTab = CharacterTable(point_group)
        self.basis = basis
        self.operation_matrices: dict[str, list] = {}
        self.spanned_basis: dict[str, list] = {}
        self.orbital_basis: dict[str, list] = {}
        self.get_operations()

    def get_operations(self):
        """"""
        # get for each operation the corresponding matrix. load for now the
        # matrices for d2h, d4h or d8h from memory
        # TODO get by symmetry tools
        self.load_d2h_matrices()


    def load_d2h_matrices(self):
        """load the opertion matrices for linear diatomics for
        s, px, py, pz orbitals."""

        px_orbs = {
            '1 E': np.array([[1, 0], [0, 1]]),
            '1 C2_z': np.array([[-1, 0], [0, -1]]),
            '1 C2_y': np.array([[0, -1], [-1, 0]]),
            '1 C2_x': np.array([[0, 1], [1, 0]]),
            '1 i': np.array([[0, -1], [-1, 0]]),
            '1 s_xy': np.array([[0, 1], [1, 0]]),
            '1 s_xz': np.array([[1, 0], [0, 1]]),
            '1 s_yz': np.array([[-1, 0], [0, -1]])
        }
        px_reducable_basis = [2, 0, 0, 0, 0, 0, 2, 0]
        px_orbital_basis = [np.array([1, 0]), np.array([0, 1])]

        self.operation_matrices['px'] = px_orbs
        self.spanned_basis['px'] = px_reducable_basis
        self.orbital_basis['px'] = px_orbital_basis

    def get_symmetry_adapted_basis(self):
        """get symmetry adapted basis for the given orbitals"""
        contributions, mulliken_labels = self.characTab.get_reduction(self.spanned_basis['px'])
        print(mulliken_labels)
        px_basis = self.orbital_basis['px']
        order = self.characTab.order
        # get symmetry adapted basis with projection operator
        for contribution, label in zip(contributions, mulliken_labels):
            if contribution != 0:
                print(label)
                dim = self.characTab.get_dimension(label)
                counter = 0
                tmp = [np.array([0,0]), np.array([0,0])]
                for operation, matrix in self.operation_matrices['px'].items():
                    res = np.dot(matrix, px_basis)*self.characTab.characters[label][counter]
                    tmp += res
                    # print(np.dot(matrix, px_basis)*self.characTab.characters[label][counter])
                    # print(operation)
                    counter += 1
                print(dim/order * tmp)


salc = SALC('d2h', ['px'])
salc.get_symmetry_adapted_basis()
