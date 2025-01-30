import numpy as np
import re
import yaml
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
        self.proj_results = {}
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
        s_orbs = {
            "1 E": np.array([[1, 0], [0, 1]]),
            "1 C2_z": np.array([[1, 0], [0, 1]]),
            "1 C2_y": np.array([[0, 1], [1, 0]]),
            "1 C2_x": np.array([[0, 1], [1, 0]]),
            "1 i": np.array([[0, 1], [1, 0]]),
            "1 s_xy": np.array([[0, 1], [1, 0]]),
            "1 s_xz": np.array([[1, 0], [0, 1]]),
            "1 s_yz": np.array([[1, 0], [0, 1]]),
        }
        s_reducable_basis = [2, 2, 0, 0, 0, 0, 2, 2]
        s_orbital_basis = [np.array([1, 0]), np.array([0, 1])]

        px_orbs = {
            "1 E": np.array([[1, 0], [0, 1]]),
            "1 C2_z": np.array([[-1, 0], [0, -1]]),
            "1 C2_y": np.array([[0, -1], [-1, 0]]),
            "1 C2_x": np.array([[0, 1], [1, 0]]),
            "1 i": np.array([[0, -1], [-1, 0]]),
            "1 s_xy": np.array([[0, 1], [1, 0]]),
            "1 s_xz": np.array([[1, 0], [0, 1]]),
            "1 s_yz": np.array([[-1, 0], [0, -1]]),
        }
        px_reducable_basis = [2, 0, 0, 0, 0, 0, 2, 0]
        px_orbital_basis = [np.array([1, 0]), np.array([0, 1])]

        py_orbs = {
            "1 E": np.array([[1, 0], [0, 1]]),
            "1 C2_z": np.array([[-1, 0], [0, -1]]),
            "1 C2_y": np.array([[0, 1], [1, 0]]),
            "1 C2_x": np.array([[0, -1], [-1, 0]]),
            "1 i": np.array([[0, -1], [-1, 0]]),
            "1 s_xy": np.array([[0, 1], [1, 0]]),
            "1 s_xz": np.array([[-1, 0], [0, -1]]),
            "1 s_yz": np.array([[1, 0], [0, 1]]),
        }
        py_reducable_basis = [2, 0, 0, 0, 0, 0, 0, 2]
        py_orbital_basis = [np.array([1, 0]), np.array([0, 1])]

        pz_orbs = {
            "1 E": np.array([[1, 0], [0, 1]]),
            "1 C2_z": np.array([[1, 0], [0, 1]]),
            "1 C2_y": np.array([[0, -1], [-1, 0]]),
            "1 C2_x": np.array([[0, -1], [-1, 0]]),
            "1 i": np.array([[0, -1], [-1, 0]]),
            "1 s_xy": np.array([[0, -1], [-1, 0]]),
            "1 s_xz": np.array([[1, 0], [0, 1]]),
            "1 s_yz": np.array([[1, 0], [0, 1]]),
        }
        pz_reducable_basis = [2, 2, 0, 0, 0, 0, 2, 2]
        pz_orbital_basis = [np.array([1, 0]), np.array([0, 1])]

        # load in variable
        self.operation_matrices["s"] = s_orbs
        self.operation_matrices["px"] = px_orbs
        self.operation_matrices["py"] = py_orbs
        self.operation_matrices["pz"] = pz_orbs
        self.spanned_basis["s"] = s_reducable_basis
        self.spanned_basis["px"] = px_reducable_basis
        self.spanned_basis["py"] = py_reducable_basis
        self.spanned_basis["pz"] = pz_reducable_basis
        self.orbital_basis["s"] = s_orbital_basis
        self.orbital_basis["px"] = px_orbital_basis
        self.orbital_basis["py"] = py_orbital_basis
        self.orbital_basis["pz"] = pz_orbital_basis

    def get_symmetry_adapted_basis(self, orbital):
        """get symmetry adapted basis for the given orbitals"""
        contributions, mulliken_labels = self.characTab.get_reduction(
            self.spanned_basis[orbital]
        )
        px_basis = self.orbital_basis[orbital]
        order = self.characTab.order

        mulliken_label_res = []
        projection_res = []
        # get symmetry adapted basis with projection operator
        for contribution, label in zip(contributions, mulliken_labels):
            if contribution != 0:
                dim = self.characTab.get_dimension(label)
                counter = 0
                tmp = [np.array([0, 0]), np.array([0, 0])]
                for operation, matrix in self.operation_matrices[
                    orbital
                ].items():
                    res = (
                        np.dot(matrix, px_basis)
                        * self.characTab.characters[label][counter]
                    )
                    tmp += res
                    # print(np.dot(matrix, px_basis)*self.characTab.characters[label][counter])
                    # print(operation)
                    counter += 1

                projection = dim / order * tmp
                if not np.all(projection == 0):
                    mulliken_label_res.append(label)
                    projection_res.append(projection)
        return mulliken_label_res, projection_res

    def get_salcs(self):
        """"""
        orb_idx = {}
        orb_xyz = []
        # count number of different orbitals and save indices
        for i, orb in enumerate(self.basis):
            ao = orb.split("_")[-1]
            if ao not in orb_idx:
                orb_idx[ao] = []
            orb_idx[ao].append(i)

            # get the different orbital species in terms of angular momentum l
            orb_species = re.search(r"\d+(\D+)", ao).group(1)
            if orb_species not in orb_xyz:
                orb_xyz.append(orb_species)

        for orb in orb_xyz:
            lab, op = salc.get_symmetry_adapted_basis(orb)
            for l, o in zip(lab, op):
                if l not in self.proj_results:
                    self.proj_results[l] = {
                        "labels": [],
                        "operations": [],
                    }
                self.proj_results[l]["labels"].append(orb)
                self.proj_results[l]["operations"].append(o)

        # construct for each reducible representation the salcs as
        # linear combination
        lst = [0 for _ in self.basis]
        for mulliken, data in self.proj_results.items():
            summands = []
            for orb, idx in orb_idx.items():
                for label, operation in zip(
                    data["labels"], data["operations"]
                ):
                    if label in orb:
                        tmp = lst.copy()
                        # assign linear combinations to the orbitals to
                        # the input orbital list
                        for j, id in enumerate(idx):
                            tmp[id] = np.sign(operation[0][j])
                        summands.append(np.array(tmp))
            self.proj_results[mulliken]["salcs"] = self.generate_combinations(
                summands
            )

    def generate_combinations(self, vectors):
        """generate all linear combinations of list of vectors"""
        signs = [-1, 1]  # Possible signs
        num_vectors = len(vectors)
        combinations = []

        # Generate all combinations using only + and -
        for i in range(2**num_vectors):  # 2^n combinations
            combination = np.zeros_like(vectors[0])
            for j in range(num_vectors):
                # Use bitwise operations to decide + or - for each vector
                sign = signs[(i >> j) & 1]
                combination += sign * vectors[j]
            combinations.append(combination)

        return combinations

    def assign_mo_coefficients(self, mos):
        """Assigns the molecular orbital coefficients to the generated SALCs
        with respect to the signs"""
        symmetries = ["" for _ in mos]
        for i, mo in enumerate(mos):
            for mul, data in self.proj_results.items():
                if not symmetries[i]:
                    for salc in data["salcs"]:
                        same = np.all(np.sign(mo) == np.sign(salc))
                        if same:
                            symmetries[i] = mul
                            break
        print(symmetries)


# parse MO coefficients
with open(
    "/home/broecker/research/molecules/c2/pbe0/sto-3g/orca.yaml", "r"
) as file:
    data = yaml.safe_load(file)

mos = data["molecularOrbitals"]["coefficients"].values()


salc = SALC(
    "d2h",
    [
        "C1_1s",
        "C1_2s",
        "C1_2px",
        "C1_2py",
        "C1_2pz",
        "C2_1s",
        "C2_2s",
        "C2_2px",
        "C2_2py",
        "C2_2pz",
    ],
)

salc.get_salcs()
salc.assign_mo_coefficients(mos)
