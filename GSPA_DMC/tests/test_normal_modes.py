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


def test_gmat():
    from pyvibdmc.analysis import AnalyzeWfn, Plotter
    norms = NormalModes(run_name='test_h2o',
                        atoms=['H', 'H', 'H', 'O'],
                        walkers=cds,
                        descendant_weights=dws,
                        ic_manager=h2o_internals)
    gmat, my_internals = norms.calc_gmat()
    hist = AnalyzeWfn.projection_1d(np.degrees(my_internals[:, -1]), desc_weights=dws)
    Plotter.plt_hist1d(hist, xlabel='Dihedral')
    assert True


def test_nms():
    norms = NormalModes(run_name='test_h2o',
                        atoms=['H', 'H', 'H', 'O'],
                        walkers=cds,
                        descendant_weights=dws,
                        ic_manager=h2o_internals)
    gmat, my_internals = norms.calc_gmat()
    q_coords = norms.calc_normal_modes(gmat, my_internals)
    assert True
