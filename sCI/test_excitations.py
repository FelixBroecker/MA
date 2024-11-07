#import pytest
from my_csf import Generate_Determinants
#from my_csf import *

sCI = Generate_Determinants()

#@pytest.fixture
def test_set_1():
    """test set 1 consists of 4 electrons in 4 orbitals with lowest energy determinant"""
    number_of_MOs = 4 
    excitations_to_perform = [1,2,3,4] # single, double, triple and quadruple excitations
    determinant = [1,-1,2,-2]
    return number_of_MOs, excitations_to_perform, determinant

def n_tuple_excitations(number_of_MOs, excitations_to_perform, determinant):
    """test simple n-tuple excitations from get_excitations function"""
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

    assert excitations == ref_excitations


# test simple n-tuple excitations
number_of_MOs, excitations_to_perform, determinant = test_set_1()
n_tuple_excitations(number_of_MOs, excitations_to_perform, determinant)

#
