import pytest
from GSPA_DMC import *
import numpy as np

def test_reg_plot():
    keyz = ['funds', 'overs', 'combos']
    np_e = np.load("test_h3o/energies.npz")
    np_i = np.load("test_h3o/intensities.npz")
    energies = []
    intensities = []
    for key in keyz:
        energies.append(np_e[key])
        intensities.append(np_i[key])
    energies = np.concatenate(energies)
    intensities = np.concatenate(intensities)
    plottie = PlotSpectrum(energies=energies,
                           intensities=intensities,
                           plt_energy_range=[0, 6000],
                           gauss_width=50,
                           )
    xy_data, gauss_data = plottie.get_xy_data()
    plottie.plot_xy_data(stick_xy=xy_data,
                         savefig_name='example_unc.png',
                         stick_color='b',
                         gauss_xy=gauss_data,
                         pdf=False)


def test_mixed_plot():
    assignments = np.load('test_h3o/assign_order.npy')
    red_stuff = np.load("test_h3o/red_ham/ov_ham.npz")
    overlap_mat = red_stuff['ov']
    ham_mat = red_stuff['ham']
    dipole_matels = red_stuff['mus']
    my_mix = MixedStates(res_dir='sample',
                         overlap_mat=overlap_mat,
                         ham_mat=ham_mat,
                         assignments=assignments,
                         energy_range=[3000, 4000],
                         dip_matels=dipole_matels)
    new_freqs, new_intensities, mixed_indcs = my_mix.run()


    plottie = PlotSpectrum(energies=new_freqs,
                 intensities=new_intensities,
                 plt_energy_range=[0,6000],
                 gauss_width=50,
                 mixed_indcs=mixed_indcs,
                 norm_contrib_fl='sample/contribs.txt',
                 )
    xy_data, gauss_data = plottie.get_xy_data()
    plottie.plot_xy_data(stick_xy=xy_data,
                         savefig_name='example.png',
                         gauss_xy=gauss_data,
                         stick_color='b',
                         mixed_stick_color='r',
                         gauss_color='k',
                         pdf=False)