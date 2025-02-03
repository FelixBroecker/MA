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
        self.orbital_basisfunctions: dict[str, list] = {}
        self.proj_results = {}
        self.get_operations()

    def get_operations(self):
        """"""
        # get for each operation the corresponding matrix. load for now the
        # matrices for d2h, d4h or d8h from memory
        # TODO get by symmetry tools
        if self.point_group == "d2h":
            self.load_d2h_matrices()
        elif self.point_group == "d4h":
            self.load_d4h_matrices()

    def get_transformation_matrix(
        self, string_list: list[str], orb: str, mat: np.ndarray
    ):
        """Transforms string representation in transformation matrix"""
        mat = mat.copy()
        pattern = r"([+-])(\w+)"
        for op in string_list:
            func_val = op.split(" -> ")
            splitted = re.findall(
                pattern,
                func_val[1],
            )
            splitted = [
                (1 if sign == "+" else -1, word) for sign, word in splitted
            ]
            mat[
                self.orbital_basisfunctions[orb].index(func_val[0]),
                self.orbital_basisfunctions[orb].index(splitted[0][1]),
            ] = splitted[0][0]
        return mat

    def load_d2h_matrices(self):
        """load the opertion matrices for linear diatomics for
        s, px, py, pz orbitals."""

        self.orbital_basisfunctions = {
            "s": ["s1", "s2"],
            "p": [
                "px1",
                "py1",
                "pz1",
                "px2",
                "py2",
                "pz2",
            ],
        }
        s_orbital_basis = [np.array([1, 0]), np.array([0, 1])]
        p_orbital_basis = [
            np.array([1, 0, 0, 0, 0, 0]),
            np.array([0, 1, 0, 0, 0, 0]),
            np.array([0, 0, 1, 0, 0, 0]),
            np.array([0, 0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 0, 1]),
        ]

        s_orbs = {
            "1 E": ["s1 -> +s1", "s2 -> +s2"],
            "1 C2_z": ["s1 -> +s1", "s2 -> +s2"],
            "1 C2_y": ["s1 -> +s2", "s2 -> +s1"],
            "1 C2_x": ["s1 -> +s2", "s2 -> +s1"],
            "1 i": ["s1 -> +s2", "s2 -> +s1"],
            "1 s_xy": ["s1 -> +s2", "s2 -> +s1"],
            "1 s_xz": ["s1 -> +s1", "s2 -> +s2"],
            "1 s_yz": ["s1 -> +s1", "s2 -> +s2"],
        }
        s_reducable_basis = [2, 2, 0, 0, 0, 0, 2, 2]
        orb_empty = np.zeros((len(s_orbital_basis), len(s_orbital_basis)))
        # convert to transformation matrix
        for mulliken, operations in s_orbs.items():
            s_orbs[mulliken] = self.get_transformation_matrix(
                operations, "s", orb_empty
            )

        px_orbs = {
            "1 E": ["px1 -> +px1", "px2 -> +px2"],
            "1 C2_z": ["px1 -> -px1", "px2 -> -px2"],
            "1 C2_y": ["px1 -> -px2", "px2 -> -px1"],
            "1 C2_x": ["px1 -> +px2", "px2 -> +px1"],
            "1 i": ["px1 -> -px2", "px2 -> -px1"],
            "1 s_xy": ["px1 -> +px2", "px2 -> +px1"],
            "1 s_xz": ["px1 -> +px1", "px2 -> +px2"],
            "1 s_yz": ["px1 -> -px1", "px2 -> -px2"],
        }
        orb_empty = np.zeros((len(p_orbital_basis), len(p_orbital_basis)))
        px_reducable_basis = [2, -2, 0, 0, 0, 0, 2, -2]
        # convert to transformation matrix
        for mulliken, operations in px_orbs.items():
            px_orbs[mulliken] = self.get_transformation_matrix(
                operations, "p", orb_empty
            )

        py_orbs = {
            "1 E": ["py1 -> +py1", "py2 -> +py2"],
            "1 C2_z": ["py1 -> -py1", "py2 -> -py2"],
            "1 C2_y": ["py1 -> +py2", "py2 -> +py1"],
            "1 C2_x": ["py1 -> -py2", "py2 -> -py1"],
            "1 i": ["py1 -> -py2", "py2 -> -py1"],
            "1 s_xy": ["py1 -> +py2", "py2 -> +py1"],
            "1 s_xz": ["py1 -> -py1", "py2 -> -py2"],
            "1 s_yz": ["py1 -> +py1", "py2 -> +py2"],
        }
        py_reducable_basis = [2, -2, 0, 0, 0, 0, -2, 2]
        # convert to transformation matrix
        for mulliken, operations in py_orbs.items():
            py_orbs[mulliken] = self.get_transformation_matrix(
                operations, "p", orb_empty
            )

        pz_orbs = {
            "1 E": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 C2_z": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 C2_y": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 C2_x": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 i": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 s_xy": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 s_xz": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 s_yz": ["pz1 -> +pz1", "pz2 -> +pz2"],
        }
        pz_reducable_basis = [2, 2, 0, 0, 0, 0, 2, 2]
        # convert to transformation matrix
        for mulliken, operations in pz_orbs.items():
            pz_orbs[mulliken] = self.get_transformation_matrix(
                operations, "p", orb_empty
            )
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
        self.orbital_basis["px"] = p_orbital_basis
        self.orbital_basis["py"] = p_orbital_basis
        self.orbital_basis["pz"] = p_orbital_basis

    def load_d4h_matrices(self):
        """load the opertion matrices for linear diatomics for
        s, px, py, pz, dxy, dxz, dyz, dx2-y2, dz2 orbitals."""

        self.orbital_basisfunctions = {
            "s": ["s1", "s2"],
            "p": [
                "px1",
                "py1",
                "pz1",
                "px2",
                "py2",
                "pz2",
            ],
            "d": [
                "dxy1",
                "dxz1",
                "dyz1",
                "dxxyy1",
                "dzz1",
                "dxy2",
                "dxz2",
                "dyz2",
                "dxxyy2",
                "dzz2",
            ],
        }
        s_orbital_basis = [np.array([1, 0]), np.array([0, 1])]
        p_orbital_basis = [
            np.array([1, 0, 0, 0, 0, 0]),
            np.array([0, 1, 0, 0, 0, 0]),
            np.array([0, 0, 1, 0, 0, 0]),
            np.array([0, 0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 0, 1]),
        ]
        d_orbital_basis = [
            np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        ]

        s_orbs = {
            "1 E": ["s1 -> +s1", "s2 -> +s2"],
            "2 C4_z": ["s1 -> +s1", "s2 -> +s2"],
            "1 C2": ["s1 -> +s1", "s2 -> +s2"],
            "2 C2''": ["s1 -> +s2", "s2 -> +s1"],
            "2 C2'''": ["s1 -> +s2", "s2 -> +s1"],
            "1 i": ["s1 -> +s2", "s2 -> +s1"],
            "2 S4": ["s1 -> +s2", "s2 -> +s1"],
            "1 sh": ["s1 -> +s2", "s2 -> +s1"],
            "2 sv": ["s1 -> +s1", "s2 -> +s2"],
            "2 sd": ["s1 -> +s1", "s2 -> +s2"],
        }
        orb_empty = np.zeros((len(s_orbital_basis), len(s_orbital_basis)))
        s_reducable_basis = [2, 2, 2, 0, 0, 0, 0, 0, 2, 2]
        # convert to transformation matrix
        for mulliken, operations in s_orbs.items():
            s_orbs[mulliken] = self.get_transformation_matrix(
                operations, "s", orb_empty
            )

        px_orbs = {
            "1 E": ["px1 -> +px1", "px2 -> +px2"],
            "2 C4_z": ["px1 -> +py1", "px2 -> +py2"],
            "1 C2": ["px1 -> -px1", "px2 -> -px2"],
            "2 C2''": ["px1 -> -px2", "px2 -> -px1"],
            "2 C2'''": ["px1 -> -py2", "px2 -> -py1"],
            "1 i": ["px1 -> -px2", "px2 -> -px1"],
            "2 S4": ["px1 -> +py2", "px2 -> +py1"],
            "1 sh": ["px1 -> +px2", "px2 -> +px1"],
            "2 sv": ["px1 -> +px1", "px2 -> +px2"],
            "2 sd": ["px1 -> -px1", "px2 -> -px2"],
        }
        orb_empty = np.zeros((len(p_orbital_basis), len(p_orbital_basis)))
        px_reducable_basis = [2, -2, 0, 0, 0, 0, 2, -2]
        # convert to transformation matrix
        for mulliken, operations in px_orbs.items():
            px_orbs[mulliken] = self.get_transformation_matrix(
                operations, "p", orb_empty
            )

        py_orbs = {
            "1 E": ["py1 -> +py1", "py2 -> +py2"],
            "2 C4_z": ["py1 -> +px1", "py2 -> +px2"],
            "1 C2": ["py1 -> -py1", "py2 -> -py2"],
            "2 C2''": ["py1 -> +py2", "py2 -> +py1"],
            "2 C2'''": ["py1 -> -px2", "py2 -> -px1"],
            "1 i": ["py1 -> -py2", "py2 -> -py1"],
            "2 S4": ["py1 -> +px2", "py2 -> +px1"],
            "1 sh": ["py1 -> +py2", "py2 -> +py1"],
            "2 sv": ["py1 -> -py1", "py2 -> -py2"],
            "2 sd": ["py1 -> -px1", "py2 -> -px2"],
        }
        py_reducable_basis = [2, -2, 0, 0, 0, 0, -2, 2]
        # convert to transformation matrix
        for mulliken, operations in py_orbs.items():
            py_orbs[mulliken] = self.get_transformation_matrix(
                operations, "p", orb_empty
            )

        pz_orbs = {
            "1 E": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "2 C4_z": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 C2": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "2 C2''": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "2 C2'''": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 i": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "2 S4": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 sh": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "2 sv": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "2 sd": ["pz1 -> +pz1", "pz2 -> +pz2"],
        }
        pz_reducable_basis = [2, 2, 0, 0, 0, 0, 2, 2]
        # convert to transformation matrix
        for mulliken, operations in pz_orbs.items():
            pz_orbs[mulliken] = self.get_transformation_matrix(
                operations, "p", orb_empty
            )

        dxy_orbs = {
            "1 E": ["dxy1 -> +dxy1", "dxy2 -> +dxy2"],
            "2 C4_z": ["dxy1 -> -dxy1", "dxy2 -> -dxy2"],
            "1 C2": ["dxy1 -> -dxy1", "dxy2 -> -dxy2"],
            "2 C2''": ["dxy1 -> -dxy2", "dxy2 -> -dxy1"],
            "2 C2'''": ["dxy1 -> +dxy2", "dxy2 -> +dxy1"],
            "1 i": ["dxy1 -> +dxy2", "dxy2 -> +dxy1"],
            "2 S4": ["dxy1 -> -dxy2", "dxy2 -> -dxy1"],
            "1 sh": ["dxy1 -> +dxy2", "dxy2 -> +dxy1"],
            "2 sv": ["dxy1 -> -dxy1", "dxy2 -> -dxy2"],
            "2 sd": ["dxy1 -> -dxy1", "dxy2 -> -dxy2"],
        }
        orb_empty = np.zeros((len(d_orbital_basis), len(d_orbital_basis)))
        dxy_reducable_basis = [2, -2, 2, 0, 0, 0, 0, 0, -2, -2]
        # convert to transformation matrix
        for mulliken, operations in dxy_orbs.items():
            dxy_orbs[mulliken] = self.get_transformation_matrix(
                operations, "d", orb_empty
            )

        dxz_orbs = {
            "1 E": ["dxz1 -> +dxz1", "dxz2 -> +dxz2"],
            "2 C4_z": ["dxz1 -> +dyz1", "dxz2 -> +dyz2"],
            "1 C2": ["dxz1 -> -dxz1", "dxz2 -> -dxz2"],
            "2 C2''": ["dxz1 -> -dxz2", "dxz2 -> -dxz1"],
            "2 C2'''": ["dxz1 -> +dyz2", "dxz2 -> +dyz1"],
            "1 i": ["dxz1 -> +dxz2", "dxz2 -> +dxz1"],
            "2 S4": ["dxz1 -> -dyz2", "dxz2 -> -dyz1"],
            "1 sh": ["dxz1 -> -dxz2", "dxz2 -> -dxz1"],
            "2 sv": ["dxz1 -> +dxz1", "dxz2 -> +dxz2"],
            "2 sd": ["dxz1 -> -dyz1", "dxz2 -> -dyz2"],
        }
        dxz_reducable_basis = [2, 0, -2, 0, 0, 0, 0, 0, 2, -2]
        # convert to transformation matrix
        for mulliken, operations in dxz_orbs.items():
            dxz_orbs[mulliken] = self.get_transformation_matrix(
                operations, "d", orb_empty
            )

        dyz_orbs = {
            "1 E": ["dyz1 -> +dyz1", "dyz2 -> +dyz2"],
            "2 C4_z": ["dyz1 -> +dxz1", "dyz2 -> +dxz2"],
            "1 C2": ["dyz1 -> -dyz1", "dyz2 -> -dyz2"],
            "2 C2''": ["dyz1 -> -dyz2", "dyz2 -> -dyz1"],
            "2 C2'''": ["dyz1 -> +dxz2", "dyz2 -> +dxz1"],
            "1 i": ["dyz1 -> +dyz2", "dyz2 -> +dyz1"],
            "2 S4": ["dyz1 -> -dxz2", "dyz2 -> -dxz1"],
            "1 sh": ["dyz1 -> -dyz2", "dyz2 -> -dyz1"],
            "2 sv": ["dyz1 -> +dyz1", "dyz2 -> +dyz2"],
            "2 sd": ["dyz1 -> -dxz1", "dyz2 -> -dxz2"],
        }
        dyz_reducable_basis = [2, 0, -2, 0, 0, 0, 0, 0, -2, 2]
        # convert to transformation matrix
        for mulliken, operations in dyz_orbs.items():
            dyz_orbs[mulliken] = self.get_transformation_matrix(
                operations, "d", orb_empty
            )

        dxx_yy_orbs = {
            "1 E": ["dxxyy1 -> +dxxyy1", "dxxyy2 -> +dxxyy2"],
            "2 C4_z": ["dxxyy1 -> -dxxyy1", "dxxyy2 -> -dxxyy2"],
            "1 C2": ["dxxyy1 -> +dxxyy1", "dxxyy2 -> +dxxyy2"],
            "2 C2''": ["dxxyy1 -> +dxxyy2", "dxxyy2 -> +dxxyy1"],
            "2 C2'''": ["dxxyy1 -> -dxxyy2", "dxxyy2 -> -dxxyy1"],
            "1 i": ["dxxyy1 -> +dxxyy2", "dxxyy2 -> +dxxyy1"],
            "2 S4": ["dxxyy1 -> -dxxyy2", "dxxyy2 -> -dxxyy1"],
            "1 sh": ["dxxyy1 -> +dxxyy2", "dxxyy2 -> +dxxyy1"],
            "2 sv": ["dxxyy1 -> +dxxyy1", "dxxyy2 -> +dxxyy2"],
            "2 sd": ["dxxyy1 -> +dxxyy1", "dxxyy2 -> +dxxyy2"],
        }
        dxx_yy_reducable_basis = [2, -2, 2, 0, 0, 0, 0, 0, 2, 2]
        # convert to transformation matrix
        for mulliken, operations in dxx_yy_orbs.items():
            dxx_yy_orbs[mulliken] = self.get_transformation_matrix(
                operations, "d", orb_empty
            )

        dzz_orbs = {
            "1 E": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "2 C4_z": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C2": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "2 C2''": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "2 C2'''": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 i": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "2 S4": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 sh": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "2 sv": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "2 sd": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
        }
        dzz_reducable_basis = [2, 2, 2, 0, 0, 0, 0, 0, 2, 2]
        # convert to transformation matrix
        for mulliken, operations in dzz_orbs.items():
            dzz_orbs[mulliken] = self.get_transformation_matrix(
                operations, "d", orb_empty
            )

        # general information
        self.operation_matrices["s"] = s_orbs
        self.operation_matrices["px"] = px_orbs
        self.operation_matrices["py"] = py_orbs
        self.operation_matrices["pz"] = pz_orbs
        self.operation_matrices["dxy"] = dxy_orbs
        self.operation_matrices["dxz"] = dxz_orbs
        self.operation_matrices["dyz"] = dyz_orbs
        self.operation_matrices["dxxyy"] = dxx_yy_orbs
        self.operation_matrices["dzz"] = dzz_orbs
        self.spanned_basis["s"] = s_reducable_basis
        self.spanned_basis["px"] = px_reducable_basis
        self.spanned_basis["py"] = py_reducable_basis
        self.spanned_basis["pz"] = pz_reducable_basis
        self.spanned_basis["dxy"] = dxy_reducable_basis
        self.spanned_basis["dxz"] = dxz_reducable_basis
        self.spanned_basis["dyz"] = dyz_reducable_basis
        self.spanned_basis["dxxyy"] = dxx_yy_reducable_basis
        self.spanned_basis["dzz"] = dzz_reducable_basis
        self.orbital_basis["dxy"] = d_orbital_basis
        self.orbital_basis["dxz"] = d_orbital_basis
        self.orbital_basis["dyz"] = d_orbital_basis
        self.orbital_basis["dxxyy"] = d_orbital_basis
        self.orbital_basis["dzz"] = d_orbital_basis

    def get_symmetry_adapted_basis(self, orbital):
        """get symmetry adapted basis for the given orbitals"""
        contributions, mulliken_labels = self.characTab.get_reduction(
            self.spanned_basis[orbital]
        )

        orb_basis = self.orbital_basis[orbital]
        order = self.characTab.order

        mulliken_label_res = []
        projection_res = []
        # get symmetry adapted basis with projection operator
        for contribution, label in zip(contributions, mulliken_labels):
            if contribution != 0:
                dim = self.characTab.get_dimension(label)
                counter = 0
                tmp = [np.zeros(len(orb_basis)) for _ in orb_basis]
                for operation, matrix in self.operation_matrices[
                    orbital
                ].items():
                    res = (
                        np.dot(matrix, orb_basis)
                        * self.characTab.characters[label][counter]
                        * int(operation.split()[0])
                    )
                    tmp += res
                    # print(np.dot(matrix, px_basis)*self.characTab.characters[label][counter])
                    # print(operation)
                    counter += 1

                projection = dim / order * tmp
                if not np.all(projection == 0):
                    mulliken_label_res.append(label)
                    projection_res.append(projection)
        print(orbital)
        print(mulliken_label_res)
        print()
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
        # TODO write this section more elegant and readable
        lst = [0 for _ in self.basis]
        for mulliken, data in self.proj_results.items():
            summands = []
            for orb, idx in orb_idx.items():
                for label, operation in zip(
                    data["labels"], data["operations"]
                ):
                    if label in orb:
                        # check at which postion the orbital is in the basis
                        for bas in self.orbital_basisfunctions.values():
                            for i, f in enumerate(bas):
                                if label in f:
                                    j = i
                                    break
                        tmp = lst.copy()
                        # assign linear combinations to the orbitals to
                        # the input orbital list
                        ops = []
                        for i, val in enumerate(operation[j]):
                            if val:
                                ops.append(i)
                        for id, o in zip(idx, ops):
                            tmp[id] = np.sign(operation[j][o])
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


data_set = "c2_tz"
if data_set == "c2_sz":
    # C2 in minimal basis
    path = "/home/broecker/research/molecules/c2/pbe0/sto-3g/orca.yaml"
    point_group = "d2h"
    orbital_basis = [
        "C1_1s",
        "C1_2s",
        "C1_1px",
        "C1_1py",
        "C1_1pz",
        "C2_1s",
        "C2_2s",
        "C2_1px",
        "C2_1py",
        "C2_1pz",
    ]
    orca_reference = [
        "Ag",
        "B1u",
        "Ag",
        "B1u",
        "B3u",
        "B2u",
        "Ag",
        "B2g",
        "B3g",
        "B1u",
    ]
elif data_set == "c2_dz":
    # C2 in double zeta
    path = "/home/broecker/research/molecules/c2/pbe0/dzae/orca.yaml"
    point_group = "d2h"
    orbital_basis = [
        "C1_1s",
        "C1_2s",
        "C1_3s",
        "C1_4s",
        "C1_1px",
        "C1_1py",
        "C1_1pz",
        "C1_2px",
        "C1_2py",
        "C1_2pz",
        "C2_1s",
        "C2_2s",
        "C2_3s",
        "C2_4s",
        "C2_1px",
        "C2_1py",
        "C2_1pz",
        "C2_2px",
        "C2_2py",
        "C2_2pz",
    ]
    orca_reference = [
        "Ag",
        "B1u",
        "Ag",
        "B1u",
        "B2u",
        "B3u",
        "Ag",
        "B2g",
        "B3g",
        "B1u",
        "Ag",
        "B2u",
        "B3u",
        "Ag",
        "B2g",
        "B3g",
        "B1u",
        "B1u",
        "Ag",
        "B1u",
    ]
elif data_set == "c2_tz":
    path = "/home/broecker/research/molecules/c2/pbe0/tzpae/orca.yaml"
    point_group = "d4h"
    orbital_basis = [
        "C1_1s",
        "C1_2s",
        "C1_3s",
        "C1_4s",
        "C1_5s",
        "C1_1px",
        "C1_1py",
        "C1_1pz",
        "C1_2px",
        "C1_2py",
        "C1_2pz",
        "C1_3px",
        "C1_3py",
        "C1_3pz",
        "C1_1dzz",
        "C1_1dxz",
        "C1_1dyz",
        "C1_1dxxyy",
        "C1_1dxy",
        "C2_1s",
        "C2_2s",
        "C2_3s",
        "C2_4s",
        "C2_5s",
        "C2_1px",
        "C2_1py",
        "C2_1pz",
        "C2_2px",
        "C2_2py",
        "C2_2pz",
        "C2_3px",
        "C2_3py",
        "C2_3pz",
        "C2_1dzz",
        "C2_1dxz",
        "C2_1dyz",
        "C2_1dxxyy",
        "C2_1dxy",
    ]
    orca_reference = [
        "Ag",
        "B1u",
        "Ag",
        "B1u",
        "B2u",
        "B3u",
        "Ag",
        "B3g",
        "B2g",
        "B1u",
        "B2u",
        "B3u",
        "Ag",
        "B3g",
        "B2g",
        "Ag",
        "B1u",
        "B1u",
        "B1g",
        "Ag",
        "B2u",
        "B3u",
        "Ag",
        "Au",
        "B1u",
        "B3u",
        "B2u",
        "B3g",
        "B2g",
        "B1u",
        "B2g",
        "B3g",
        "Ag",
        "B1u",
        "Ag",
        "B1u",
        "Ag",
        "B1u",
    ]
else:
    print("Invalid input.")
    exit()

# parse MO coefficients
with open(path, "r") as file:
    data = yaml.safe_load(file)

mos = data["molecularOrbitals"]["coefficients"].values()


salc = SALC(
    point_group,
    orbital_basis,
)

salc.get_salcs()
salc.assign_mo_coefficients(mos)
print("Orca reference:")
print(orca_reference)
