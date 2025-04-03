class Utils:
    def hex2bin(self, hex, bit_len):
        """"""
        integer_value = int(hex, 16)
        binary_string = bin(integer_value)[2:]

        return binary_string.zfill(bit_len)

    def bin2det(self, bin, sgn=1):
        """"""
        return [
            sgn * (i + 1) for i, bit in enumerate(reversed(bin)) if bit == "1"
        ]

    def parse_cipsi_dets(self, filename: str):
        """
        parse the hexadecimal abbreviation and amplitudes
        in quantum package2 output.
        """
        n_mo = 0
        index = []
        determinant = []
        amplitude = []
        counter = 0
        found = False
        with open(filename, "r") as reffile:
            for line in reffile:
                counter += 1
                if "mo_num" in line:
                    n_mo = int(line.split()[-1])
                if "i =" in line:
                    index.append(int(line.split()[-1]))
                    found = True
                    counter = 0
                if "amplitude" in line:
                    amplitude.append(float(line.split()[-1]))
                if found and counter == 2:
                    determinant.append(line.replace("\n", "").split("|"))

        # get deterinant format
        determinant = self.parse_qp_dets(determinant, n_mo)
        return amplitude, determinant

    def parse_qp_dets(self, determinants, n_mo):
        """ """
        res = []
        for pair in determinants:
            bin = self.hex2bin(pair[0], n_mo)
            alpha = self.bin2det(bin, sgn=1)

            bin = self.hex2bin(pair[1], n_mo)
            beta = self.bin2det(bin, sgn=-1)
            res.append(alpha + beta)
        return res


if __name__ == "__main__":
    ut = Utils()
    ut.parse_cipsi_dets("fci.wf")
