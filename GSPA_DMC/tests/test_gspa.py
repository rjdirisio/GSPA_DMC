import pytest
from GSPA_DMC import GSPA, MixedStates
import numpy as np

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


def test_mixing():
    assignments = np.load('test_h3o/assign_order.npy')
    red_stuff = np.load("test_h3o/red_ham/ov_ham.npz")
    overlap_mat = red_stuff['ov']
    ham_mat = red_stuff['ham']
    dipole_matels = red_stuff['mus']
    my_mix = MixedStates(res_dir='sample',
                        overlap_mat=overlap_mat,
                        ham_mat=ham_mat,
                        assignments=assignments,
                        energy_range=[1000,4000],
                        dip_matels=dipole_matels)
    new_freqs, new_intensities, chunk = my_mix.run()
    assert True

# def test_analyze_ov():
#     import matplotlib.pyplot as plt
#     fl = np.load("test_h3o/red_ham/ov_ham.npz")
#     this_ov = fl['ov']
#     other_ov = np.loadtxt("test_h3o/red_ham/old_correct_results/overlapMatrix2_ffinal_h3ornspca.dat")[1:,1:]
#     eval_other, evec_other = np.linalg.eigh(other_ov)
#     eval_this, evec_this = np.linalg.eigh(this_ov)
#     c = np.abs(evec_other) - np.abs(evec_this)
#     c_a = np.linalg.det(other_ov)
#     c_b = np.linalg.det(this_ov)
#     diff = this_ov - other_ov
#     abs_diff = np.abs(this_ov) - np.abs(other_ov)
#     asdf = np.where(np.abs(np.triu(diff)) > 0.01)
#     asdf = np.array([asdf[0],asdf[1]]).T
#     fig, ax = plt.subplots(nrows=1, ncols=3)
#     cax = ax[0].matshow(this_ov - np.eye(len(this_ov)))
#     # fig.colorbar(cax)
#     dax = ax[1].matshow(other_ov - np.eye(len(this_ov)))
#     # fig.colorbar(dax)
#     eax = ax[2].matshow(diff)
#     fig.savefig("comparison.png",bbox_inches='tight',dpi=400)
#     assert True
