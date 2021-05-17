import pytest
from GSPA_DMC import *
import matplotlib.pyplot as plt

h2o_internals = InternalCoordinateManager(
    int_function='h3o_internals',
    int_directory='',
    python_file='h3o_internals.py',
    int_names=['ROH_1', 'ROH_2', 'ROH_3', '2Th_1-Th2-Th3', 'Th2-Th3', 'Umb'])

dws = np.load('h3o_data/ffinal_h3o_dw.npy')
eng_dip = np.load("h3o_data/eng_dip_ffinal_h3o_eckart.npy")
vs = eng_dip[:, 0]

def test_freqs():
    q_coords = np.load("test_h3o/nms.npy")
    my_gspa = GSPA(res_dir='test_h3o',
                   normal_modes=q_coords,
                   desc_weights=dws,
                   potential_energies=vs,
                   dipoles=eng_dip[:, 1:],
                   ham_overlap=False)
    my_gspa.run()
    assert True
