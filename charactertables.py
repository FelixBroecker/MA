class CharacterTable:
    def __init__(self, point_group: str):
        """"""
        self.point_group = point_group
        self.operations = []
        self.characters = {}
        self.linear_funcs = {}
        self.quadratic_funcs = {}
        self.order = 0
        self.init_table()

    def init_table(self):
        """load the desired character table"""
        if self.point_group == "d2h":
            self.d2h()
    
    def multiply(self, characters_1, characters_2):
        """elementwise character multiplication of two irreps"""
        assert len(characters_1) == len(characters_2) 
        res = [i * j for i, j in zip(characters_1, characters_2)]
        return res
    
    def d2h(self):
        """load character table d2h"""
        self.operations = [
            "1 E",
            "1 C2",
            "1 sv_xz",
            "1 sv_yz",
            ]
        self.characters = {
            "Ag" : [1,1,1,1,1,1,1,1],
            "B1g": [1,1,-1,-1,1,1,-1,-1],
            "B2g": [1,-1,1,-1,1,-1,1,-1],
            "B3g": [1,-1,-1,1,1,-1,-1,1],
            "Au" : [1,1,1,1,-1,-1,-1,-1],
            "B1u": [1,1,-1,-1,-1,-1,1,1],
            "B2u": [1,-1,1,-1,-1,1,-1,1],
            "B3u": [1,-1,-1,1,-1,1,1,-1],
        }
        self.linear_funcs = {
            "Ag" : [],
            "B1g": ["Rz"],
            "B2g": ["Ry"],
            "B3g": ["Rx"],
            "Au" : [], 
            "B1u": ["z"],
            "B2u": ["y"],
            "B3u": ["x"],
        }
        self.quadratic_funcs = {
            "Ag" : ["xx","yy","zz"],
            "B1g": ["xy"],
            "B2g": ["xz"],
            "B3g": ["yz"],
            "Au" : [], 
            "B1u": [],
            "B2u": [],
            "B3u": [],
        }
        self.order = 4
    
if __name__ == "__main__":
    symmetry = CharacterTable("d2h")
    res = symmetry.multiply(symmetry.characters["B1g"], symmetry.characters["B1g"])
    print(res)