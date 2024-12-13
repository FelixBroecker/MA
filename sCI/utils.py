class Utils:
    def hex2bin(self, hex, bit_len):
        """"""
        integer_value = int(hex, 16)
        binary_string = bin(integer_value)[2:]

        return binary_string.zfill(bit_len)

    def bin2det(self, bin, sgn=1):
        """"""
        return [sgn * i for i, bit in enumerate(reversed(bin)) if bit == "1"]

    def parse_cipsi_dets(self, filename: str):
        """
        parse the hexadecimal abbreviation and amplitudes
        in quantum package2 output.
        """
        n_mo = 0
        with open(filename, "r") as reffile:
            for line in reffile:
                if "mo_num" in line:
                    n_mo = int(line.split()[-1])
        print(n_mo)

    def parse_qp_dets(self):
        """ """
        hex_string = "000000000000040F"
        bin = self.hex2bin(hex_string, 31)

        # Output the result
        print(f"Hexadecimal: {hex_string}")
        print(f"Binary: {bin}")

        indices = self.bin2det(bin, sgn=-1)

        # Output the result
        print(f"Binary: {bin}")
        print(f"Indices of set bits (1-based): {indices}")


if __name__ == "__main__":
    ut = Utils()
    ut.parse_cipsi_dets("fci.wf")
