class CharacterTable:
    def __init__(self, point_group: str):
        """"""
        self.point_group = point_group
        self.operations = []
        self.characters = {}
        self.linear_funcs = {}
        self.quadratic_funcs = {}
        self.cubic_funcs = {}
        self.order = 0
        self.init_table()

    def init_table(self):
        """load the desired character table"""
        if self.point_group == "d2h":
            self.d2h()
        if self.point_group == "c2v":
            self.c2v()
        if self.point_group == "d4h":
            self.d4h()
        if self.point_group == "d4h_expanded":
            self.d4h_expanded()
        if self.point_group == "cs":
            self.cs()

    def multiply(self, characters_1, characters_2):
        """elementwise character multiplication of two irreps"""
        assert len(characters_1) == len(
            characters_2
        ), "characters are of unequal length"
        res = [i * j for i, j in zip(characters_1, characters_2)]
        return res

    def get_reduction(self, representation: list):
        """"""
        g = [int(i.split(" ")[0]) for i in self.operations]
        contributions = []
        for label, charac in self.characters.items():
            assert len(representation) == len(
                charac
            ), "Length of representation does not match with length of characters."
            contribution = float(
                sum(
                    [
                        g[i] * representation[i] * char
                        for i, char in enumerate(charac)
                    ]
                )
                / self.order
            )
            contributions.append(contribution)
        return contributions, list(self.characters.keys())

    def get_dimension(self, label: str):
        """Return dimension of inserted mulliken label"""
        dimensions = {
            "A": 1,
            "B": 1,
            "E": 2,
            "T": 3,
            "G": 4,
            "H": 5,
        }
        return dimensions[list(label)[0]]

    def character2label(self, character):
        """translate character to mulliken label of character table"""
        for label, charac in self.characters.items():
            if charac == character:
                return label
        return None

    def cs(self):
        """load character table d2h"""
        self.operations = [
            "1 E",
            "1 sh",
        ]
        self.characters = {"A'": [1, 1], "A''": [1, -1]}
        self.linear_funcs = {
            "1 A'": ["x", "y", "Rz"],
            "A''": ["z", "Rx", "Ry"],
        }
        self.quadratic_funcs = {
            "1 A'": ["xx", "yy", "zz", "xy"],
            "1 A''": ["yz", "xz"],
        }
        self.order = 2

    def d2h(self):
        """load character table d2h"""
        self.operations = [
            "1 E",
            "1 C2_z",
            "1 C2_y",
            "1 C2_x",
            "1 i",
            "1 s_xy",
            "1 s_xz",
            "1 s_yz",
        ]
        self.characters = {
            "Ag": [1, 1, 1, 1, 1, 1, 1, 1],
            "B1g": [1, 1, -1, -1, 1, 1, -1, -1],
            "B2g": [1, -1, 1, -1, 1, -1, 1, -1],
            "B3g": [1, -1, -1, 1, 1, -1, -1, 1],
            "Au": [1, 1, 1, 1, -1, -1, -1, -1],
            "B1u": [1, 1, -1, -1, -1, -1, 1, 1],
            "B2u": [1, -1, 1, -1, -1, 1, -1, 1],
            "B3u": [1, -1, -1, 1, -1, 1, 1, -1],
        }
        self.linear_funcs = {
            "Ag": [],
            "B1g": ["Rz"],
            "B2g": ["Ry"],
            "B3g": ["Rx"],
            "Au": [],
            "B1u": ["z"],
            "B2u": ["y"],
            "B3u": ["x"],
        }
        self.quadratic_funcs = {
            "Ag": ["xx", "yy", "zz"],
            "B1g": ["xy"],
            "B2g": ["xz"],
            "B3g": ["yz"],
            "Au": [],
            "B1u": [],
            "B2u": [],
            "B3u": [],
        }
        self.order = 8

    def d4h(self):
        """load character table d4h"""
        self.operations = [
            "1 E",
            "2 C4_z",
            "1 C2",
            "2 C2'",
            "2 C2''",
            "1 i",
            "2 S4",
            "1 sh",
            "2 sv",
            "2 sd",
        ]
        self.characters = {
            "A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "A2g": [1, 1, 1, -1, -1, 1, 1, 1, -1, -1],
            "B1g": [1, -1, 1, 1, -1, 1, -1, 1, 1, -1],
            "B2g": [1, -1, 1, -1, 1, 1, -1, 1, -1, 1],
            "Eg": [2, 0, -2, 0, 0, 2, 0, -2, 0, 0],
            "A1u": [
                1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
            "A2u": [1, 1, 1, -1, -1, -1, -1, -1, 1, 1],
            "B1u": [1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
            "B2u": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
            "Eu": [2, 0, -2, 0, 0, -2, 0, 2, 0, 0],
        }
        self.linear_funcs = {
            "A1g": [],
            "A2g": ["Rz"],
            "B1g": [],
            "B2g": [],
            "Eg": ["Rx", "Ry"],
            "A1u": [],
            "A2u": ["z"],
            "B1u": [],
            "B2u": [],
            "Eu": ["x", "y"],
        }
        self.quadratic_funcs = {
            "A1g": ["xx+yy", "zz"],
            "A2g": [],
            "B1g": ["xx-yy"],
            "B2g": ["xy"],
            "Eg": ["xz", "yz"],
            "A1u": [],
            "A2u": [],
            "B1u": [],
            "B2u": [],
            "Eu": [],
        }
        self.order = 16

    def d4h_expanded(self):
        """load character table d4h"""
        self.operations = [
            "1 E",
            "1 C4_z+",
            "1 C4_z-",
            "1 C2",
            "1 C2''x",
            "1 C2''y",
            "1 C2'''1",
            "1 C2'''2",
            "1 i",
            "1 S4+",
            "1 S4-",
            "1 sh",
            "1 sv'",
            "1 sv''",
            "1 sd'",
            "1 sd''",
        ]
        self.characters = {
            "A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "A2g": [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
            "B1g": [1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1],
            "B2g": [1, -1, -1, 1, -1, -1, 1, 1, 1, -1, -1, 1, -1, -1, 1, 1],
            "Eg": [2, 0, 0, -2, 0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0],
            "A1u": [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
                -1,
            ],
            "A2u": [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1],
            "B1u": [1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1],
            "B2u": [1, -1, -1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, -1],
            "Eu": [2, 0, 0, -2, 0, 0, 0, 0, -2, 0, 0, 2, 0, 0, 0, 0],
        }
        self.order = 16

    def c2v(self):
        """load character table c2v"""
        self.operations = [
            "1 E",
            "1 C2_z",
            "1 sv_xz",
            "1 sv_yz",
        ]
        self.characters = {
            "A1": [1, 1, 1, 1],
            "A2": [1, 1, -1, -1],
            "B1": [1, -1, 1, -1],
            "B2": [1, -1, -1, 1],
        }
        self.linear_funcs = {
            "A1": ["z"],
            "A2": ["Rz"],
            "B1": ["x", "Ry"],
            "B2": ["y", "Rx"],
        }
        self.quadratic_funcs = {
            "A1": ["xx", "yy", "zz"],
            "A2": ["xy"],
            "B1": ["xz"],
            "B2": ["yz"],
        }
        self.cubic_funcs = {
            "A1": ["zzz", "xxz", "yyz"],
            "A2": ["xyz"],
            "B1": ["xzz", "xxx", "xyy"],
            "B2": ["yzz", "yyy", "xxy"],
        }
        self.order = 4


if __name__ == "__main__":
    symmetry = CharacterTable("c2v")
    print(symmetry.characters)
    res = symmetry.multiply(
        symmetry.characters["A1"], symmetry.characters["A2"]
    )
    print(res)
    print(f"reduction: {symmetry.get_reduction([2, 0, 0, 2])}")

    symmetry = CharacterTable("d2h")
