#import pytest
from csf import SelectedCI
#from my_csf import *

sCI = SelectedCI()

#@pytest.fixture
def test_set_1():
    """test set 1 consists of 4 electrons in 4 orbitals with lowest energy determinant"""
    number_of_MOs = 4 
    excitations_to_perform = [1,2,3,4] # single, double, triple and quadruple excitations
    determinant = [1,-1,2,-2]
    return number_of_MOs, excitations_to_perform, determinant

def test_set_2():
    """test set of csfs from 4 electrons in 4 orbitals with 1,2,3,4-tuple excitations from 
    lowest energy determinant without symmetry considerations"""
    csfs = [
        [[1, -1, 2, -2]], [[2, -2, 3, -3]], [[2, -2, 4, -4]], [[1, -1, 3, -3]], 
        [[1, -1, 4, -4]], [[3, -3, 4, -4]], [[1, 2, -2, -3], [-1, 2, -2, 3]], 
        [[1, 2, -2, -4], [-1, 2, -2, 4]], [[1, -1, 2, -3], [1, -1, -2, 3]], 
        [[1, -1, 2, -4], [1, -1, -2, 4]], [[2, -2, 3, -4], [2, -2, -3, 4]], 
        [[1, -2, 3, -4], [1, -2, -3, 4], [-1, 2, 3, -4], [-1, 2, -3, 4]], 
        [[1, -2, 3, -4], [1, -2, -3, 4], [1, 2, -3, -4], [-1, 2, 3, -4], [-1, 2, -3, 4], [-1, -2, 3, 4]], 
        [[1, -2, 3, -3], [-1, 2, 3, -3]], [[1, -2, 4, -4], [-1, 2, 4, -4]], [[1, -1, 3, -4], [1, -1, -3, 4]], 
        [[2, -3, 4, -4], [-2, 3, 4, -4]], [[2, 3, -3, -4], [-2, 3, -3, 4]], 
        [[1, -3, 4, -4], [-1, 3, 4, -4]], [[1, 3, -3, -4], [-1, 3, -3, 4]]
        ]
    csf_coefficients = [
        [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.5000000000000001, -0.5000000000000001, -0.5000000000000001, 0.5000000000000001], 
        [-0.2886751345948129, -0.2886751345948129, 0.5773502691896258, -0.2886751345948129, -0.2886751345948129, 0.5773502691896258], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476]
        ]
    return csf_coefficients,csfs

def test_set_3():
    """test set from an AMOLQC calculation of hydrogen with 2 electrons in 12 orbitals with 24 csfs"""
    csfs = [
        [[1, -1]], [[1, -3], [3, -1]], [[1, -7], [7, -1]], [[1, -10], 
        [10, -1]], [[2, -2]], [[2, -6], [6, -2]], [[2, -11], [11, -2]], 
        [[2, -12], [12, -2]], [[3, -3]], [[3, -7], [7, -3]], [[3, -10], [10, -3]], 
        [[4, -4]], [[5, -5]], [[6, -6]], [[6, -11], [11, -6]], [[6, -12], [12, -6]], 
        [[7, -7]], [[7, -10], [10, -7]], [[8, -8]], [[9, -9]], [[10, -10]], 
        [[11, -11]], [[11, -12], [12, -11]], [[12, -12]]
        ]
    csf_coefficients = [
        [1.0], [0.707107, 0.707107], [0.707107, 0.707107], [0.707107, 0.707107], 
        [1.0], [0.707107, 0.707107], [0.707107, 0.707107], [0.707107, 0.707107], 
        [1.0], [0.707107, 0.707107], [0.707107, 0.707107], [1.0], [1.0], [1.0], 
        [0.707107, 0.707107], [0.707107, 0.707107], [1.0], [0.707107, 0.707107], 
        [1.0], [1.0], [1.0], [1.0], [0.707107, 0.707107], [1.0]
        ]
    CI_coefficients = [
        0.991812, 0.0134374, 0.0144379, -0.0119714, -0.0495558, 
        0.0584739, 0.0169282, -0.00363802, -0.0306023, -0.0263075, 
        0.0147238, -0.0439413, -0.0439769, -0.0474732, -0.0216545, 
        0.00443761, -0.0232408, 0.0114742, -0.00760275, -0.00762334, 
        -0.0138386, -0.012523, 0.00291433, -0.00229089
        ]
    return csf_coefficients, csfs, CI_coefficients


def test_set_4():
    """test set 4 consists of 4 electrons in 4 orbitals with a already single excited determinant."""
    number_of_MOs = 4 
    excitations_to_perform = [2] # double excitations
    determinant = [1,-1,2,-3]
    return number_of_MOs, excitations_to_perform, determinant

def n_tuple_excitations(number_of_MOs, excitations_to_perform, determinant):
    """test simple n-tuple excitations from get_excitations function. ref_excitations correspond to
        4 fold excitation of determinant [1,-1,2,-2]"""
    ref_excitations = [
        [-1, 2, -2, 3], [-1, 2, -2, 4],
        [1, 2, -2, -4], [1, 2, -2, -3], 
        [1, -1, -2, 3], [1, -1, -2, 4], 
        [1, -1, 2, -4], [1, -1, 2, -3], 
        [2, -2, 3, -4], [2, -2, 3, -3], 
        [-1, -2, 3, 4], [-1, 2, 3, -4], 
        [-1, 2, 3, -3], [2, -2, 4, -4], 
        [2, -2, -3, 4], [-1, 2, 4, -4], 
        [-1, 2, -3, 4], [1, -2, 3, -4], 
        [1, -2, 4, -4], [1, 2, -3, -4], 
        [1, -2, 3, -3], [1, -2, -3, 4], 
        [1, -1, 3, -4], [1, -1, 3, -3], 
        [1, -1, 4, -4], [1, -1, -3, 4], 
        [-2, 3, 4, -4], [2, 3, -3, -4], 
        [-2, 3, -3, 4], [-1, 3, 4, -4], 
        [-1, 3, -3, 4], [2, -3, 4, -4], 
        [1, 3, -3, -4], [1, -3, 4, -4], 
        [3, -3, 4, -4]
        ]
    excitations = sCI.get_excitations(number_of_MOs, excitations_to_perform, determinant)

    assert excitations == ref_excitations, "test for simple n-tuple excitation without symmetry failed"


def n_tuple_excitations_symmetry(number_of_MOs, excitations_to_perform, determinant, orbital_symmetry, total_symmetry):
    """test simple n-tuple excitations from get_excitations function. ref_excitations correspond to
        4 fold excitation of determinant [1,-1,2,-2]. Only determinants with same symmetry of initial determinant."""
    ref_excitations = [
        [1, -1, -2, 4], [1, -1, 2, -4], 
        [2, -2, 3, -3], [2, -2, 4, -4], 
        [1, -1, 3, -3], [1, -1, 4, -4], 
        [2, 3, -3, -4], [-2, 3, -3, 4], 
        [3, -3, 4, -4]
        ]    
    excitations = sCI.get_excitations(number_of_MOs, excitations_to_perform, determinant, orbital_symmetry=orbital_symmetry, tot_sym=total_symmetry)
    assert excitations == ref_excitations, "test for simple n-tuple excitation with symmetry considerations failed."


def csfs_without_symmetry(number_of_MOs, excitations_to_perform, determinant, S, M_s):
    """form all csfs from determinant basis without considerations of symmetry."""
    ref_csfs = [
        [[1, -1, 2, -2]], [[2, -2, 3, -3]], [[2, -2, 4, -4]], [[1, -1, 3, -3]], 
        [[1, -1, 4, -4]], [[3, -3, 4, -4]], [[1, 2, -2, -3], [-1, 2, -2, 3]], 
        [[1, 2, -2, -4], [-1, 2, -2, 4]], [[1, -1, 2, -3], [1, -1, -2, 3]], 
        [[1, -1, 2, -4], [1, -1, -2, 4]], [[2, -2, 3, -4], [2, -2, -3, 4]], 
        [[1, -2, 3, -4], [1, -2, -3, 4], [-1, 2, 3, -4], [-1, 2, -3, 4]], 
        [[1, -2, 3, -4], [1, -2, -3, 4], [1, 2, -3, -4], [-1, 2, 3, -4], [-1, 2, -3, 4], [-1, -2, 3, 4]], 
        [[1, -2, 3, -3], [-1, 2, 3, -3]], [[1, -2, 4, -4], [-1, 2, 4, -4]], [[1, -1, 3, -4], [1, -1, -3, 4]], 
        [[2, -3, 4, -4], [-2, 3, 4, -4]], [[2, 3, -3, -4], [-2, 3, -3, 4]], 
        [[1, -3, 4, -4], [-1, 3, 4, -4]], [[1, 3, -3, -4], [-1, 3, -3, 4]]
        ]
    ref_csf_coefficients = [
        [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.5000000000000001, -0.5000000000000001, -0.5000000000000001, 0.5000000000000001], 
        [-0.2886751345948129, -0.2886751345948129, 0.5773502691896258, -0.2886751345948129, -0.2886751345948129, 0.5773502691896258], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476]
        ]
    # get determinant basis
    determinant_basis = []
    determinant_basis += [determinant]
    excitations = sCI.get_excitations(number_of_MOs, excitations_to_perform, determinant)
    determinant_basis += excitations
    # get csfs
    csf_coefficients, csfs = sCI.get_unique_csfs(determinant_basis, S, M_s) 
    assert csf_coefficients == ref_csf_coefficients, "test of csfs without symmetry failed with wrong csf coefficients."
    assert csfs == ref_csfs, "test of csfs without symmetry failed with wrong configuration state functions."


def csfs_with_symmetry(number_of_MOs, excitations_to_perform, determinant, S, M_s, orbital_symmetry, total_symmetry):
    """form all csfs from determinant basis with consideration of symmetry."""
    ref_csfs = [
        [[1, -1, 2, -2]], [[2, -2, 3, -3]], 
        [[2, -2, 4, -4]], [[1, -1, 3, -3]], 
        [[1, -1, 4, -4]], [[3, -3, 4, -4]], 
        [[1, -1, 2, -4], [1, -1, -2, 4]], 
        [[2, 3, -3, -4], [-2, 3, -3, 4]]
    ]
    ref_csf_coefficients = [
        [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], 
        [0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, -0.7071067811865476]
                ]
     # get determinant basis
    determinant_basis = []
    determinant_basis += [determinant]
    excitations = sCI.get_excitations(number_of_MOs, excitations_to_perform, determinant, orbital_symmetry, total_symmetry)
    determinant_basis += excitations
    # get csfs
    csf_coefficients, csfs = sCI.get_unique_csfs(determinant_basis, S, M_s) 
    assert csf_coefficients == ref_csf_coefficients, "test of csfs with symmetry failed with wrong csf coefficients."
    assert csfs == ref_csfs, "test of csfs with symmetry failed with wrong configuration state functions."

def sort_determinants_in_csfs(csf_coefficients, csfs):
    """sort determinants in csfs in AMOLQC format. alpha spins appear first and beta spins second."""
    ref_csfs = [
        [[1, 2, -1, -2]], [[2, 3, -2, -3]], [[2, 4, -2, -4]], [[1, 3, -1, -3]], 
        [[1, 4, -1, -4]], [[3, 4, -3, -4]], [[1, 2, -2, -3], [2, 3, -1, -2]], 
        [[1, 2, -2, -4], [2, 4, -1, -2]], [[1, 2, -1, -3], [1, 3, -1, -2]], 
        [[1, 2, -1, -4], [1, 4, -1, -2]], [[2, 3, -2, -4], [2, 4, -2, -3]], 
        [[1, 3, -2, -4], [1, 4, -2, -3], [2, 3, -1, -4], [2, 4, -1, -3]], 
        [[1, 3, -2, -4], [1, 4, -2, -3], [1, 2, -3, -4], [2, 3, -1, -4], [2, 4, -1, -3], [3, 4, -1, -2]], 
        [[1, 3, -2, -3], [2, 3, -1, -3]], [[1, 4, -2, -4], [2, 4, -1, -4]], 
        [[1, 3, -1, -4], [1, 4, -1, -3]], [[2, 4, -3, -4], [3, 4, -2, -4]], 
        [[2, 3, -3, -4], [3, 4, -2, -3]], [[1, 4, -3, -4], [3, 4, -1, -4]], 
        [[1, 3, -3, -4], [3, 4, -1, -3]]
                            ]
    ref_csf_coefficients = [
        [-1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0], 
        [0.7071067811865476, 0.7071067811865476], 
        [0.7071067811865476, 0.7071067811865476], 
        [-0.7071067811865476, -0.7071067811865476], 
        [-0.7071067811865476, -0.7071067811865476], 
        [-0.7071067811865476, -0.7071067811865476], 
        [-0.5000000000000001, -0.5000000000000001, -0.5000000000000001, -0.5000000000000001], 
        [0.2886751345948129, -0.2886751345948129, 0.5773502691896258, -0.2886751345948129, 0.2886751345948129, 0.5773502691896258], 
        [-0.7071067811865476, -0.7071067811865476], 
        [-0.7071067811865476, -0.7071067811865476], 
        [-0.7071067811865476, -0.7071067811865476], 
        [-0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, 0.7071067811865476], 
        [-0.7071067811865476, -0.7071067811865476], 
        [0.7071067811865476, 0.7071067811865476]
                ]
    sorted_csf_coefficients, sorted_csfs = sCI.sort_determinants_in_csfs(csf_coefficients, csfs)
    assert sorted_csf_coefficients == ref_csf_coefficients, "test of sorting determinants in csfs in AMOLQ format failed with wrong coefficients."
    assert sorted_csfs == ref_csfs, "test of sorting determinants in csfs in AMOLQ format failed with wrong configuration state functions."


def cut_of_csfs(csf_coefficients, csfs, CI_coefficients,CI_coefficient_thresh):
    """cut off csfs with respect to the CI coefficients and keep cutted part"""
    ref_csfs = [
        [[1, -1]], [[2, -6], [6, -2]], [[2, -2]], [[6, -6]], [[5, -5]], 
        [[4, -4]], [[3, -3]], [[3, -7], [7, -3]], [[7, -7]], 
        [[6, -11], [11, -6]], [[2, -11], [11, -2]], [[3, -10], [10, -3]], 
        [[1, -7], [7, -1]], [[10, -10]], [[1, -3], [3, -1]], [[11, -11]], 
        [[1, -10], [10, -1]], [[7, -10], [10, -7]]
                ]
    ref_csf_coefficients = [
        [1.0], [0.707107, 0.707107], [1.0], [1.0], [1.0], [1.0], [1.0], 
        [0.707107, 0.707107], [1.0], [0.707107, 0.707107], [0.707107, 0.707107], 
        [0.707107, 0.707107], [0.707107, 0.707107], [1.0], [0.707107, 0.707107], 
        [1.0], [0.707107, 0.707107], [0.707107, 0.707107]
                ]
    ref_CI_coefficients = [
        0.991812, 0.0584739, -0.0495558, -0.0474732, -0.0439769, 
        -0.0439413, -0.0306023, -0.0263075, -0.0232408, -0.0216545, 
        0.0169282, 0.0147238, 0.0144379, -0.0138386, 0.0134374, 
        -0.012523, -0.0119714, 0.0114742
        ]
    ref_csfs_cut = [
        [[9, -9]], [[8, -8]], [[6, -12], [12, -6]], 
        [[2, -12], [12, -2]], [[11, -12], [12, -11]], [[12, -12]]
        ]
    ref_csf_coefficients_cut = [
        [1.0], [1.0], [0.707107, 0.707107], [0.707107, 0.707107], [0.707107, 0.707107], [1.0]
        ]
    ref_CI_coefficients_cut = [
        -0.00762334, -0.00760275, 0.00443761, -0.00363802, 0.00291433, -0.00229089
        ]
    csf_coefficients, csfs, CI_coefficients, csf_coefficients_cut, csfs_cut, CI_coefficients_cut \
    = sCI.cut_csfs(csf_coefficients, csfs, CI_coefficients, CI_coefficient_thresh)
    assert csf_coefficients == ref_csf_coefficients, "cutting of csfs with respect to the MO coefficients failed with respect to the csf coefficients"
    assert csfs == ref_csfs, "cutting of csfs with respect to the MO coefficients failed with respect to the csfs"
    assert CI_coefficients == ref_CI_coefficients, "cutting of csfs with respect to the MO coefficients failed with respect to the CI coefficients"
    assert csf_coefficients_cut == ref_csf_coefficients_cut, "cutting of csfs with respect to the MO coefficients failed with respect to the cut csf coefficients"
    assert csfs_cut == ref_csfs_cut, "cutting of csfs with respect to the MO coefficients failed with respect to the cut csfs"
    assert CI_coefficients_cut == ref_CI_coefficients_cut, "cutting of csfs with respect to the MO coefficients failed with respect to the cut CI coefficients"


def n_tuple_excitations_with_reference(number_of_MOs, excitations_to_perform, determinant, reference_determinant):
    """perform n-tuple excitation with electrons in orbitals that have not been excited with respect to the reference determinant."""
    ref_excitations = [
        [2, 3, -3, -4], [-1, 3, -3, 4], [2, -3, 4, -4], [1, 3, -3, -4], [1, -3, 4, -4]
        ]
        
    excitations = sCI.get_excitations(number_of_MOs, excitations_to_perform, determinant, det_reference=reference_determinant)
    assert excitations == ref_excitations, "test for n-tuple excitation with respect to a reference determinant without symmetry failed"


def n_tuple_excitations_frozen_core(number_of_MOs, excitations_to_perform, determinant, frozen_core_electrons):
    """test simple n-tuple excitation by freezing core electrons."""
    ref_excitations = [
        [1, -1, -2, 3], [1, -1, -2, 4], 
        [1, -1, 2, -4], [1, -1, 2, -3], 
        [1, -1, 3, -4], [1, -1, 3, -3], 
        [1, -1, 4, -4], [1, -1, -3, 4]
        ]
    excitations = sCI.get_excitations(number_of_MOs, excitations_to_perform, determinant, core=frozen_core_electrons)

    assert excitations == ref_excitations, "test for n-tuple excitation with frozen core without symmetry failed"

# test simple n-tuple excitations
number_of_MOs, excitations_to_perform, determinant = test_set_1()
n_tuple_excitations(number_of_MOs, excitations_to_perform, determinant)

# test simple n-tuple excitations envoking symmetry example
number_of_MOs, excitations_to_perform, determinant = test_set_1()
molecule_symmetrie = "c2v"
orbital_symmetry = ['A1', 'B1', 'B2', 'B1']
n_tuple_excitations_symmetry(number_of_MOs, excitations_to_perform, determinant, orbital_symmetry, molecule_symmetrie)

# test formation of csfs without symmetry
number_of_MOs, excitations_to_perform, determinant = test_set_1()
spin_quantum_number = 0
magnetic_spin_number = 0
csfs_without_symmetry(number_of_MOs, excitations_to_perform, determinant,spin_quantum_number,magnetic_spin_number)

# test formation of csfs with symmetry
number_of_MOs, excitations_to_perform, determinant = test_set_1()
molecule_symmetrie = "c2v"
orbital_symmetry = ['A1', 'B1', 'B2', 'B1']
spin_quantum_number = 0
magnetic_spin_number = 0
csfs_with_symmetry(number_of_MOs, excitations_to_perform, determinant, spin_quantum_number, magnetic_spin_number, orbital_symmetry, molecule_symmetrie)

# test sorting of determinants in csfs to obtain AMOLQC format
csf_coefficients, csfs = test_set_2()
sort_determinants_in_csfs(csf_coefficients,csfs)

# test cut off of csfs with respect to CI coefficients
csf_coefficients, csfs, CI_coefficients = test_set_3()
CI_coefficient_thresh = 1e-2
cut_of_csfs(csf_coefficients, csfs, CI_coefficients,CI_coefficient_thresh)

# test excitations that are performed from occupied orbitals from a reference determinant
number_of_MOs, excitations_to_perform, determinant = test_set_4()
reference_determinant = [1,-1,2,-2]
n_tuple_excitations_with_reference(number_of_MOs, excitations_to_perform, determinant, reference_determinant)

# test simple excitations without symmetry by freezing core orbitals
number_of_MOs, excitations_to_perform, determinant = test_set_1()
frozen_core_electrons = [1,-1]
n_tuple_excitations_frozen_core(number_of_MOs, excitations_to_perform, determinant, frozen_core_electrons)

print("All tests passed ✅")
