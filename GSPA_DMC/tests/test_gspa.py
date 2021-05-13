import pytest
from GSPA_DMC import *
import matplotlib.pyplot as plt

h2o_internals = InternalCoordinateManager(
    int_function='h3o_internals',
    int_directory='',
    python_file='h3o_internals.py',
    int_names=['ROH_1', 'ROH_2', 'ROH_3', '2Th_1-Th2-Th3', 'Th2-Th3', 'Umb'])

cds = np.load('h3o_data/ffinal_h3o.npy')[:500]
dws = np.load('h3o_data/ffinal_h3o_dw.npy')[:500]
eng_dip = np.load("h3o_data/eng_dip_ffinal_h3o_eckart.npy")[:500]
vs = eng_dip[:, 0]

def test_freqs():
    norms = NormalModes(run_name='test_h3o',
                        atoms=['H', 'H', 'H', 'O'],
                        walkers=cds,
                        descendant_weights=dws,
                        ic_manager=h2o_internals)
    gmat, my_internals = norms.calc_gmat()
    q_coords = norms.calc_normal_modes(gmat, my_internals)
    my_gspa = GSPA(run_name='test_h3o',
                   normal_modes=q_coords,
                   desc_weights=dws,
                   potential_energies=vs,
                   dipoles=eng_dip[:, 1:],
                   ham_overlap=False)
    my_gspa.run()
    assert True
