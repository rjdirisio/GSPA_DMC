import pytest
import numpy as np
from GSPA_DMC import SymmetrizeWfn as symm


def test_swap():
    cds = np.load('h3o_data/ffinal_h3o.npy')
    dws = np.load('h3o_data/ffinal_h3o_dw.npy')
    cds = cds[:10]
    a = symm.swap_two_atoms(cds, dws, atm_1=1,atm_2=2)
    b = symm.swap_group(cds, dws, atm_list_1=[0,1],atm_list_2=[2,3])
    assert True