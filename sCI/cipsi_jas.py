from csf import SelectedCI


class AddSingles:
    def __init__(self):
        self.sCI = SelectedCI()

    def add_singles_det(
            self, N, n_MO, orbital_symmetry, point_group,
            frozen_electrons, frozen_MOs, wavefunction_name, wftype
            ):
        initial_determinant = self.sCI.build_energy_lowest_detetminant(N)
        excited_determinants = self.sCI.get_excitations(
            n_MO,
            [1],
            initial_determinant,
            orbital_symmetry=orbital_symmetry,
            tot_sym=point_group,
            core=frozen_electrons,
            frozen_MOs=frozen_MOs,
        )
        # sort determinants
        temp = []
        for det in excited_determinants:
            _, det_tmp = self.sCI.sort_determinant(1, det)
            temp.append(det_tmp)
        excited_determinants = temp.copy()
        csf_coefficients, csfs, CI_coefficients, wfpretext = (
            self.sCI.read_AMOLQC_csfs(f"{wavefunction_name}.wf", N)
        )
        _, _, det_basis = self.sCI.get_transformation_matrix(
            csf_coefficients, csfs, CI_coefficients
        )
        temp = []
        for det in det_basis:
            _, det_tmp = self.sCI.sort_determinant(1, det)
            temp.append(det_tmp)
        det_basis = temp.copy()
        all_determinants = det_basis + excited_determinants
        all_determinants = self.sCI.spinfuncs.remove_duplicates(
            all_determinants
            )

        CI_coefficients = [
            1 if n == 0 else 0 for n in range(len(all_determinants))
        ]

        self.sCI.write_AMOLQC(
            [],
            all_determinants,
            CI_coefficients,
            pretext=wfpretext,
            file_name=f"{wavefunction_name}_add_sgls.wf",
            wftype=wftype,
        )
