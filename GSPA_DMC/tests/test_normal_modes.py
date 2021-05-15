import pytest
from GSPA_DMC import *
import matplotlib.pyplot as plt


h2o_internals = InternalCoordinateManager(
    int_function='h3o_internals',
    int_directory='',
    python_file='h3o_internals.py',
    int_names=['ROH_1', 'ROH_2', 'ROH_3', 'Umb','2Th_1-Th2-Th3', 'Th2-Th3'])

cds = np.load('h3o_data/ffinal_h3o.npy')
dws = np.load('h3o_data/ffinal_h3o_dw.npy')

def test_gmat():
    norms = NormalModes(res_dir='test_h3o',
                        atoms=['H', 'H', 'H', 'O'],
                        walkers=cds,
                        descendant_weights=dws,
                        ic_manager=h2o_internals)
    gmat, my_internals = norms.calc_gmat()
    np.save("test_h3o/internals.npy", my_internals)
    assert True

def test_internals():
    from pyvibdmc.analysis import AnalyzeWfn, Plotter
    internals = h2o_internals.get_ints(cds)
    int_names = h2o_internals.get_int_names()
    for i_num, name in enumerate(int_names):
        histie = AnalyzeWfn.projection_1d(internals[:,i_num],desc_weights=dws)
        Plotter.plt_hist1d(histie,xlabel=f'test_h3o/{name}',save_name=f'{name}')
    assert True

def test_just_nms():
    gmat = np.load("test_h3o/gmat.npy")
    my_internals = np.load("test_h3o/internals.npy")
    norms = NormalModes(res_dir='test_h3o',
                        atoms=['H', 'H', 'H', 'O'],
                        walkers=cds,
                        descendant_weights=dws,
                        ic_manager=h2o_internals)
    norms.calc_normal_modes(gmat=gmat,
                            internal_coordinates=my_internals,
                            save_nms=True)
    assert True