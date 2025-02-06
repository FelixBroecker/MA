import numpy as np
import re
import yaml
from charactertables import CharacterTable


class SALC:
    def __init__(self, point_group: str, basis: list, cartesian: bool = False):
        """"""
        self.point_group = point_group
        self.characTab = CharacterTable(point_group)
        self.basis = basis
        self.cartesian = cartesian
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
        elif self.point_group == "d4h_expanded":
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
            if splitted[0][1] == "0":
                return mat
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
        if self.cartesian:
            self.orbital_basisfunctions["d"] = [
                "dxy1",
                "dxz1",
                "dyz1",
                "dxx1",
                "dyy1",
                "dzz1",
                "dxy2",
                "dxz2",
                "dyz2",
                "dxx2",
                "dyy2",
                "dzz2",
            ]
        s_orbital_basis = [np.array([1, 0]), np.array([0, 1])]
        p_orbital_basis = [
            np.array([1, 0, 0, 0, 0, 0]),
            np.array([0, 1, 0, 0, 0, 0]),
            np.array([0, 0, 1, 0, 0, 0]),
            np.array([0, 0, 0, 1, 0, 0]),
            np.array([0, 0, 0, 0, 1, 0]),
            np.array([0, 0, 0, 0, 0, 1]),
        ]
        n_d_orbitals = len(self.orbital_basisfunctions["d"])
        d_orbital_basis = [
            np.eye(n_d_orbitals)[i] for i in range(n_d_orbitals)
        ]
        s_orbs = {
            "1 E": ["s1 -> +s1", "s2 -> +s2"],
            "1 C4_z+": ["s1 -> +s1", "s2 -> +s2"],
            "1 C4_z-": ["s1 -> +s1", "s2 -> +s2"],
            "1 C2": ["s1 -> +s1", "s2 -> +s2"],
            "1 C2''x": ["s1 -> +s2", "s2 -> +s1"],
            "1 C2''y": ["s1 -> +s2", "s2 -> +s1"],
            "1 C2'''1": ["s1 -> +s2", "s2 -> +s1"],
            "1 C2'''2": ["s1 -> +s2", "s2 -> +s1"],
            "1 i": ["s1 -> +s2", "s2 -> +s1"],
            "1 S4+": ["s1 -> +s2", "s2 -> +s1"],
            "1 S4-": ["s1 -> +s2", "s2 -> +s1"],
            "1 sh": ["s1 -> +s2", "s2 -> +s1"],
            "1 sv'": ["s1 -> +s1", "s2 -> +s2"],
            "1 sv''": ["s1 -> +s1", "s2 -> +s2"],
            "1 sd'": ["s1 -> +s1", "s2 -> +s2"],
            "1 sd''": ["s1 -> +s1", "s2 -> +s2"],
        }
        orb_empty = np.zeros((len(s_orbital_basis), len(s_orbital_basis)))
        s_reducable_basis = [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]
        # convert to transformation matrix
        for mulliken, operations in s_orbs.items():
            s_orbs[mulliken] = self.get_transformation_matrix(
                operations, "s", orb_empty
            )

        px_orbs = {
            "1 E": ["px1 -> +px1", "px2 -> +px2"],
            "1 C4_z+": ["px1 -> +py1", "px2 -> +py2"],
            "1 C4_z-": ["px1 -> -py1", "px2 -> -py2"],
            "1 C2": ["px1 -> -px1", "px2 -> -px2"],
            "1 C2''x": ["px1 -> -px2", "px2 -> -px1"],
            "1 C2''y": ["px1 -> +px2", "px2 -> +px1"],
            "1 C2'''1": ["px1 -> +py2", "px2 -> +py1"],
            "1 C2'''2": ["px1 -> -py2", "px2 -> -py1"],
            "1 i": ["px1 -> -px2", "px2 -> -px1"],
            "1 S4+": ["px1 -> +py2", "px2 -> +py1"],
            "1 S4-": ["px1 -> -py2", "px2 -> -py1"],
            "1 sh": ["px1 -> +px2", "px2 -> +px1"],
            "1 sv'": ["px1 -> -px1", "px2 -> -px2"],
            "1 sv''": ["px1 -> +px1", "px2 -> +px2"],
            "1 sd'": ["px1 -> +py1", "px2 -> +py2"],
            "1 sd''": ["px1 -> -py1", "px2 -> -py2"],
        }
        orb_empty = np.zeros((len(p_orbital_basis), len(p_orbital_basis)))
        px_reducable_basis = [
            2,
            0,
            0,
            -2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -2,
            -2,
            2,
            2,
        ]
        # convert to transformation matrix
        for mulliken, operations in px_orbs.items():
            px_orbs[mulliken] = self.get_transformation_matrix(
                operations, "p", orb_empty
            )

        py_orbs = {
            "1 E": ["py1 -> +py1", "py2 -> +py2"],
            "1 C4_z+": ["py1 -> +px1", "py2 -> +px2"],
            "1 C4_z-": ["py1 -> -px1", "py2 -> -px2"],
            "1 C2": ["py1 -> -py1", "py2 -> -py2"],
            "1 C2''x": ["py1 -> +py2", "py2 -> +py1"],
            "1 C2''y": ["py1 -> -py2", "py2 -> -py1"],
            "1 C2'''1": ["py1 -> +px2", "py2 -> +px1"],
            "1 C2'''2": ["py1 -> -px2", "py2 -> -px1"],
            "1 i": ["py1 -> -py2", "py2 -> -py1"],
            "1 S4+": ["py1 -> +px2", "py2 -> +px1"],
            "1 S4-": ["py1 -> -px2", "py2 -> -px1"],
            "1 sh": ["py1 -> +py2", "py2 -> +py1"],
            "1 sv'": ["py1 -> +py1", "py2 -> +py2"],
            "1 sv''": ["py1 -> -py1", "py2 -> -py2"],
            "1 sd'": ["py1 -> +px1", "py2 -> +px2"],
            "1 sd''": ["py1 -> -px1", "py2 -> -px2"],
        }
        py_reducable_basis = [
            2,
            0,
            0,
            -2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            2,
            -2,
            -2,
        ]
        # convert to transformation matrix
        for mulliken, operations in py_orbs.items():
            py_orbs[mulliken] = self.get_transformation_matrix(
                operations, "p", orb_empty
            )

        pz_orbs = {
            "1 E": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 C4_z+": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 C4_z-": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 C2": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 C2''x": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 C2''y": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 C2'''1": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 C2'''2": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 i": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 S4+": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 S4-": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 sh": ["pz1 -> -pz2", "pz2 -> -pz1"],
            "1 sv'": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 sv''": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 sd'": ["pz1 -> +pz1", "pz2 -> +pz2"],
            "1 sd''": ["pz1 -> +pz1", "pz2 -> +pz2"],
        }
        pz_reducable_basis = [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]
        # convert to transformation matrix
        for mulliken, operations in pz_orbs.items():
            pz_orbs[mulliken] = self.get_transformation_matrix(
                operations, "p", orb_empty
            )

        dxy_orbs = {
            "1 E": ["dxy1 -> +dxy1", "dxy2 -> +dxy2"],
            "1 C4_z+": ["dxy1 -> -dxy1", "dxy2 -> -dxy2"],
            "1 C4_z-": ["dxy1 -> -dxy1", "dxy2 -> -dxy2"],
            "1 C2": ["dxy1 -> +dxy1", "dxy2 -> +dxy2"],
            "1 C2''x": ["dxy1 -> -dxy2", "dxy2 -> -dxy1"],
            "1 C2''y": ["dxy1 -> -dxy2", "dxy2 -> -dxy1"],
            "1 C2'''1": ["dxy1 -> +dxy2", "dxy2 -> +dxy1"],
            "1 C2'''2": ["dxy1 -> +dxy2", "dxy2 -> +dxy1"],
            "1 i": ["dxy1 -> +dxy2", "dxy2 -> +dxy1"],
            "1 S4+": ["dxy1 -> -dxy2", "dxy2 -> -dxy1"],
            "1 S4-": ["dxy1 -> -dxy2", "dxy2 -> -dxy1"],
            "1 sh": ["dxy1 -> +dxy2", "dxy2 -> +dxy1"],
            "1 sv'": ["dxy1 -> -dxy1", "dxy2 -> -dxy2"],
            "1 sv''": ["dxy1 -> -dxy1", "dxy2 -> -dxy2"],
            "1 sd'": ["dxy1 -> +dxy1", "dxy2 -> +dxy2"],
            "1 sd''": ["dxy1 -> +dxy1", "dxy2 -> +dxy2"],
        }
        orb_empty = np.zeros((len(d_orbital_basis), len(d_orbital_basis)))
        dxy_reducable_basis = [
            2,
            -2,
            -2,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -2,
            -2,
            +2,
            +2,
        ]

        dxz_orbs = {
            "1 E": ["dxz1 -> +dxz1", "dxz2 -> +dxz2"],
            "1 C4_z+": ["dxz1 -> +dyz1", "dxz2 -> +dyz2"],
            "1 C4_z-": ["dxz1 -> -dyz1", "dxz2 -> -dyz2"],
            "1 C2": ["dxz1 -> -dxz1", "dxz2 -> -dxz2"],
            "1 C2''x": ["dxz1 -> +dxz2", "dxz2 -> +dxz1"],
            "1 C2''y": ["dxz1 -> -dxz2", "dxz2 -> -dxz1"],
            "1 C2'''1": ["dxz1 -> +dyz2", "dxz2 -> +dyz1"],
            "1 C2'''2": ["dxz1 -> -dyz2", "dxz2 -> -dyz1"],
            "1 i": ["dxz1 -> +dxz2", "dxz2 -> +dxz1"],
            "1 S4+": ["dxz1 -> +dyz2", "dxz2 -> +dyz1"],
            "1 S4-": ["dxz1 -> -dyz2", "dxz2 -> -dyz1"],
            "1 sh": ["dxz1 -> -dxz2", "dxz2 -> -dxz1"],
            "1 sv'": ["dxz1 -> +dxz1", "dxz2 -> +dxz2"],
            "1 sv''": ["dxz1 -> -dxz1", "dxz2 -> -dxz2"],
            "1 sd'": ["dxz1 -> +dyz1", "dxz2 -> +dyz2"],
            "1 sd''": ["dxz1 -> -dyz1", "dxz2 -> -dxz2"],
        }
        dxz_reducable_basis = [
            2,
            0,
            0,
            -2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            -2,
            +2,
            -2,
        ]

        dyz_orbs = {
            "1 E": ["dyz1 -> +dyz1", "dyz2 -> +dyz2"],
            "1 C4_z+": ["dyz1 -> +dxz1", "dyz2 -> +dxz2"],
            "1 C4_z-": ["dyz1 -> -dxz1", "dyz2 -> -dxz2"],
            "1 C2": ["dyz1 -> -dyz1", "dyz2 -> -dyz2"],
            "1 C2''x": ["dyz1 -> -dyz2", "dyz2 -> -dyz1"],
            "1 C2''y": ["dyz1 -> +dyz2", "dyz2 -> +dyz1"],
            "1 C2'''1": ["dyz1 -> +dxz2", "dyz2 -> +dxz1"],
            "1 C2'''2": ["dyz1 -> -dxz2", "dyz2 -> -dxz1"],
            "1 i": ["dyz1 -> +dyz2", "dyz2 -> +dyz1"],
            "1 S4+": ["dyz1 -> -dxz2", "dyz2 -> -dxz1"],
            "1 S4-": ["dyz1 -> +dxz2", "dyz2 -> +dxz1"],
            "1 sh": ["dyz1 -> -dyz2", "dyz2 -> -dyz1"],
            "1 sv'": ["dyz1 -> +dyz1", "dyz2 -> +dyz1"],
            "1 sv''": ["dyz1 -> -dyz1", "dyz2 -> -dyz1"],
            "1 sd'": ["dyz1 -> +dxz1", "dyz2 -> +dxz2"],
            "1 sd''": ["dyz1 -> -dxz1", "dyz2 -> -dxz2"],
        }
        dyz_reducable_basis = [
            2,
            0,
            0,
            -2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            -2,
            -2,
            2,
            2,
        ]

        dxxyy_orbs = {
            "1 E": ["dxxyy1 -> +dxxyy1", "dxxyy2 -> +dxxyy2"],
            "1 C4_z+": ["dxxyy1 -> -dxxyy1", "dxxyy2 -> -dxxyy2"],
            "1 C4_z-": ["dxxyy1 -> -dxxyy1", "dxxyy2 -> -dxxyy2"],
            "1 C2": ["dxxyy1 -> +dxxyy1", "dxxyy2 -> +dxxyy2"],
            "1 C2''x": ["dxxyy1 -> +dxxyy2", "dxxyy2 -> +dxxyy1"],
            "1 C2''y": ["dxxyy1 -> +dxxyy2", "dxxyy2 -> +dxxyy1"],
            "1 C2'''1": ["dxxyy1 -> -dxxyy2", "dxxyy2 -> -dxxyy1"],
            "1 C2'''2": ["dxxyy1 -> -dxxyy2", "dxxyy2 -> -dxxyy1"],
            "1 i": ["dxxyy1 -> +dxxyy2", "dxxyy2 -> +dxxyy1"],
            "1 S4+": ["dxxyy1 -> -dxxyy2", "dxxyy2 -> -dxxyy1"],
            "1 S4-": ["dxxyy1 -> -dxxyy2", "dxxyy2 -> -dxxyy1"],
            "1 sh": ["dxxyy1 -> +dxxyy2", "dxxyy2 -> +dxxyy1"],
            "1 sv'": ["dxxyy1 -> +dxxyy1", "dxxyy2 -> +dxxyy2"],
            "1 sv''": ["dxxyy1 -> +dxxyy1", "dxxyy2 -> +dxxyy2"],
            "1 sd'": ["dxxyy1 -> -dxxyy1", "dxxyy2 -> -dxxyy2"],
            "1 sd''": ["dxxyy1 -> -dxxyy1", "dxxyy2 -> -dxxyy2"],
        }
        dxxyy_reducable_basis = [
            2,
            -2,
            -2,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            2,
            -2,
            -2,
        ]

        dzz_orbs = {
            "1 E": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C4_z+": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C4_z-": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C2": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C2''x": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 C2''y": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 C2'''1": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 C2'''2": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 i": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 S4+": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 S4-": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 sh": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 sv'": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 sv''": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 sd'": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 sd''": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
        }
        dzz_reducable_basis = [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2]

        dxx_orbs = {
            "1 E": ["dxx1 -> +dxx1", "dxx2 -> +dxx2"],
            "1 C4_z+": ["dxx1 -> +dyy1", "dxx2 -> +dyy2"],
            "1 C4_z-": ["dxx1 -> +dyy1", "dxx2 -> +dyy2"],
            "1 C2": ["dxx1 -> +dxx1", "dxx2 -> +dxx2"],
            "1 C2''x": ["dxx1 -> +dxx2", "dxx2 -> +dxx1"],
            "1 C2''y": ["dxx1 -> +dxx2", "dxx2 -> +dxx1"],
            "1 C2'''1": ["dxx1 -> +dyy2", "dxx2 -> +dyy1"],
            "1 C2'''2": ["dxx1 -> +dyy2", "dxx2 -> +dyy1"],
            "1 i": ["dxx1 -> +dxx2", "dxx2 -> +dxx1"],
            "1 S4+": ["dxx1 -> +dyy2", "dxx2 -> +dyy1"],
            "1 S4-": ["dxx1 -> +dyy2", "dxx2 -> +dyy1"],
            "1 sh": ["dxx1 -> +dxx2", "dxx2 -> +dxx1"],
            "1 sv'": ["dxx1 -> +dxx1", "dxx2 -> +dxx2"],
            "1 sv''": ["dxx1 -> +dxx1", "dxx2 -> +dxx2"],
            "1 sd'": ["dxx1 -> +dyy1", "dxx2 -> +dyy2"],
            "1 sd''": ["dxx1 -> +dyy1", "dxx2 -> +dyy2"],
        }
        dxx_reducable_basis = [
            2,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            2,
            0,
            0,
        ]

        dyy_orbs = {
            "1 E": ["dyy1 -> +dyy1", "dyy2 -> +dyy2"],
            "1 C4_z+": ["dyy1 -> +dxx1", "dyy2 -> +dxx2"],
            "1 C4_z-": ["dyy1 -> +dxx1", "dyy2 -> +dxx2"],
            "1 C2": ["dyy1 -> +dyy1", "dyy2 -> +dyy2"],
            "1 C2''x": ["dyy1 -> +dyy2", "dyy2 -> +dyy1"],
            "1 C2''y": ["dyy1 -> +dyy2", "dyy2 -> +dyy1"],
            "1 C2'''1": ["dyy1 -> +dxx2", "dyy2 -> +dxx1"],
            "1 C2'''2": ["dyy1 -> +dxx2", "dyy2 -> +dxx1"],
            "1 i": ["dyy1 -> +dyy2", "dyy2 -> +dyy1"],
            "1 S4+": ["dyy1 -> +dxx2", "dyy2 -> +dxx1"],
            "1 S4-": ["dyy1 -> +dxx2", "dyy2 -> +dxx1"],
            "1 sh": ["dyy1 -> +dyy2", "dyy2 -> +dyy1"],
            "1 sv'": ["dyy1 -> +dyy1", "dyy2 -> +dyy2"],
            "1 sv''": ["dyy1 -> +dyy1", "dyy2 -> +dyy2"],
            "1 sd'": ["dyy1 -> +dxx1", "dyy2 -> +dxx2"],
            "1 sd''": ["dyy1 -> +dxx1", "dyy2 -> +dxx2"],
        }
        dyy_reducable_basis = [
            2,
            0,
            0,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            2,
            0,
            0,
        ]

        dzz_cart_orbs = {
            "1 E": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C4_z+": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C4_z-": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C2": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 C2''x": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 C2''y": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 C2'''1": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 C2'''2": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 i": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 S4+": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 S4-": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 sh": ["dzz1 -> +dzz2", "dzz2 -> +dzz1"],
            "1 sv'": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 sv''": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 sd'": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
            "1 sd''": ["dzz1 -> +dzz1", "dzz2 -> +dzz2"],
        }
        dzz_cart_reducable_basis = [
            2,
            2,
            2,
            2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            2,
            2,
            2,
            2,
        ]
        cart_bas = [
            dyz_orbs,
            dxz_orbs,
            dxy_orbs,
            dxx_orbs,
            dyy_orbs,
            dzz_cart_orbs,
        ]
        sph_bas = [dyz_orbs, dxz_orbs, dxy_orbs, dxxyy_orbs, dzz_orbs]
        if self.cartesian:
            for orbs in cart_bas:
                for mulliken, operations in orbs.items():
                    orbs[mulliken] = self.get_transformation_matrix(
                        operations, "d", orb_empty
                    )
        else:
            for orbs in sph_bas:
                for mulliken, operations in orbs.items():
                    orbs[mulliken] = self.get_transformation_matrix(
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
        if self.cartesian:
            self.operation_matrices["dxx"] = dxx_orbs
            self.operation_matrices["dyy"] = dyy_orbs
            self.operation_matrices["dzz"] = dzz_cart_orbs
        else:
            self.operation_matrices["dxxyy"] = dxxyy_orbs
            self.operation_matrices["dzz"] = dzz_orbs
        self.spanned_basis["s"] = s_reducable_basis
        self.spanned_basis["px"] = px_reducable_basis
        self.spanned_basis["py"] = py_reducable_basis
        self.spanned_basis["pz"] = pz_reducable_basis
        self.spanned_basis["dxy"] = dxy_reducable_basis
        self.spanned_basis["dxz"] = dxz_reducable_basis
        self.spanned_basis["dyz"] = dyz_reducable_basis
        if self.cartesian:
            self.spanned_basis["dxx"] = dxx_reducable_basis
            self.spanned_basis["dyy"] = dyy_reducable_basis
            self.spanned_basis["dzz"] = dzz_cart_reducable_basis
        else:
            self.spanned_basis["dxxyy"] = dxxyy_reducable_basis
            self.spanned_basis["dzz"] = dzz_reducable_basis
        self.orbital_basis["s"] = s_orbital_basis
        self.orbital_basis["px"] = p_orbital_basis
        self.orbital_basis["py"] = p_orbital_basis
        self.orbital_basis["pz"] = p_orbital_basis
        self.orbital_basis["dxy"] = d_orbital_basis
        self.orbital_basis["dxz"] = d_orbital_basis
        self.orbital_basis["dyz"] = d_orbital_basis
        if self.cartesian:
            self.orbital_basis["dxx"] = d_orbital_basis
            self.orbital_basis["dyy"] = d_orbital_basis
            self.orbital_basis["dzz"] = d_orbital_basis
        else:
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
        # print(orbital)
        # print(mulliken_label_res)
        # print()
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
            lab, op = self.get_symmetry_adapted_basis(orb)
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
            # seperate summands if degenerate
            # TODO Adapt for other orbital labels or types.
            # only valid for x,y degeneracy
            deg = False
            if "E" in mulliken:
                deg = True
                summands_x = []
                summands_y = []
            # print(data["operations"])
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
                                # TODO This solution is really not nice
                                # and should be changed
                                bas_func = re.search(
                                    r"([a-zA-Z]+)(\d+)",
                                    self.orbital_basisfunctions[
                                        list(label)[0]
                                    ][i],
                                ).group(1)
                                if bas_func in orb:
                                    ops.append(i)
                        for id, o in zip(idx, ops):
                            tmp[id] = np.sign(operation[j][o])
                        summands.append(np.array(tmp))
                        # for E group
                        if deg:
                            if "x" in label:
                                summands_x.append(np.array(tmp))
                            elif "y" in label:
                                summands_y.append(np.array(tmp))
            if deg:
                self.proj_results[mulliken]["salcs"] = (
                    self.generate_combinations(summands_x)
                )
                self.proj_results[mulliken][
                    "salcs"
                ] += self.generate_combinations(summands_y)
            else:
                self.proj_results[mulliken]["salcs"] = (
                    self.generate_combinations(summands)
                )

    def generate_combinations(self, vectors):
        """generate all linear combinations of list of vectors"""
        signs = [-1, 1]  # Possible signs
        num_vectors = len(vectors)
        combinations = []

        # Generate all combinations using only + and -
        for i in range(2**num_vectors):  # 2^n combinations
            combination = np.zeros_like(vectors[0]).astype(float)

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
                    for j, salc in enumerate(data["salcs"]):
                        same = np.all(np.sign(mo) == np.sign(salc))
                        if same:
                            symmetries[i] = mul
                            break
        print(symmetries)
        print(len(symmetries))


data_set = "c2_tz"
if data_set == "c2_sz":
    # C2 in minimal basis
    path = "/home/broecker/research/molecules/c2/pbe0/orca/sto-3g/orca.yaml"
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
    path = "/home/broecker/research/molecules/c2/pbe0/orca/dzae/orca.yaml"
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
    path = "/home/broecker/research/molecules/c2/pbe0/orca/tzpae/orca.yaml"
    point_group = "d4h_expanded"
    cartesian = True
    if not cartesian:
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
    else:
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
            "C1_1dxx",
            "C1_1dyy",
            "C1_1dzz",
            "C1_1dxy",
            "C1_1dxz",
            "C1_1dyz",
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
            "C2_1dxx",
            "C2_1dyy",
            "C2_1dzz",
            "C2_1dxy",
            "C2_1dxz",
            "C2_1dyz",
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
    gamess_reference = [
        "A1G",
        "A2U",
        "A1G",
        "A2U",
        "EU",
        "EU",
        "A1G",
        "EG",
        "EG",
        "A2U",
        "EU",
        "EU",
        "A1G",
        "A1G",
        "EG",
        "EG",
        "A2U",
        "A2U",
        "B1G",
        "B2G",
        "EU",
        "EU",
        "A1G",
        "B2U",
        "B1U",
        "A1G",
        "EU",
        "EU",
        "EG",
        "EG",
        "A2U",
        "A2U",
        "EG",
        "EG",
        "A1G",
        "A2U",
        "A1G",
        "A2U",
        "A1G",
        "A2U",
    ]
else:
    print("Invalid input.")
    exit()

data = [[] for _ in orbital_basis]

# parse orca mos
with open(
    "/home/broecker/research/molecules/c2/pbe0/orca/tzpae/orca.mkl", "r"
) as reffile:
    found = False
    for line in reffile:
        if "$END" in line:
            found = False
        if "$COEFF_ALPHA" in line:
            found = True
            continue
        if "a1g" in line:
            counter = 0
            line = reffile.readline()
            continue
        if found:
            items = line.split()
            for val in items:
                data[counter].append(float(val))
            counter += 1
transposed = list(map(list, zip(*data)))

if cartesian:
    data = [[] for _ in orbital_basis]
    # parse gamess mos
    counter = 0
    with open(
        "/home/broecker/research/molecules/c2/pbe0/gamess/gamess.out", "r"
    ) as reffile:
        found = False
        for line in reffile:
            if "EIGENVECTORS" in line:
                found = True
                for _ in range(6):
                    line = reffile.readline()
            if "...... END OF RHF CALCULATION ......" in line:
                found = False
                break
            if counter == len(orbital_basis):
                counter = 0
                for _ in range(4):
                    line = reffile.readline()
            if found:
                items = line.split()
                items = items[4:]
                for val in items:
                    data[counter].append(float(val))
                counter += 1
    transposed = list(map(list, zip(*data)))

# parse MO coefficients
with open(path, "r") as file:
    data = yaml.safe_load(file)

mos = data["molecularOrbitals"]["coefficients"].values()
if cartesian:
    mos = transposed

salc = SALC(
    point_group,
    orbital_basis,
    cartesian=cartesian,
)

salc.get_salcs()
salc.assign_mo_coefficients(mos)
print("Orca reference:")
print(orca_reference)
print(len(orca_reference))

if data_set == "c2_tz":
    print("Gamess reference (cartesian):")
    print(gamess_reference)
    print(len(gamess_reference))
