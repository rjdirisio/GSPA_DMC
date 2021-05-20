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
                   ham_overlap=True)
    my_gspa.run()
    assert True

def test_analyze_ov():
    import matplotlib.pyplot as plt
    fl = np.load("test_h3o/red_ham/ov_ham.npz")
    this_ov = fl['ov']
    other_ov = np.loadtxt("test_h3o/red_ham/old_correct_results/overlapMatrix2_ffinal_h3ornspca.dat")[1:,1:]
    fig, ax = plt.subplots(nrows=1, ncols=3)
    cax = ax[0].matshow(this_ov - np.eye(len(this_ov)))
    # fig.colorbar(cax)
    dax = ax[1].matshow(other_ov - np.eye(len(this_ov)))
    # fig.colorbar(dax)
    eax = ax[2].matshow(this_ov - other_ov )
    fig.savefig("comparison.png",bbox_inches='tight',dpi=400)
    assert True