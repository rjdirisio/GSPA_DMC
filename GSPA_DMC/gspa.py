from .normal_mode_src import *
from .gspa_src import *

import os,sys

from pyvibdmc.simulation_utilities import Constants


class NormalModes:
    def __init__(self,
                 res_dir,
                 atoms,
                 walkers,
                 descendant_weights,
                 ic_manager,
                 atomic_units=True,
                 masses=None
                 ):

        self.res_dir = res_dir
        self.atoms = atoms
        self.coords = walkers
        self.dw = descendant_weights
        self.atomic_units = atomic_units
        self.masses = masses
        self.coordinate_func = ic_manager
        self._initialize()

    def _initialize(self):
        # Make directory where results will be
        if not os.path.isdir(self.res_dir):
            os.makedirs(self.res_dir)
        # Masses and atoms
        if not self.atomic_units:
            self.coords = Constants.convert(self.coords, 'angstroms', to_AU=True)
        if self.masses is None:
            self.masses = np.array([Constants.mass(a) for a in self.atoms])
        # Useful variables that don't need to be recalculated all the time
        self.num_atoms = len(self.atoms)
        self.num_vibs = 3 * self.num_atoms - 6

    def calc_gmat(self):
        print('Begin Calculation of G-Matrix...')
        this_gmat = Gmat(coords=self.coords,
                                    masses=self.masses,
                                    num_vibs=self.num_vibs,
                                    dw=self.dw,
                                    coordinate_func=self.coordinate_func)
        avg_gmat, internals = this_gmat.run()
        np.save(f"{self.res_dir}/gmat.npy", avg_gmat)
        int_cds = self.coordinate_func.get_ints(self.coords)
        return avg_gmat, int_cds

    def save_assignments(self,trans_mat):
        written_tmat = np.copy(trans_mat)
        with open(f'{self.res_dir}/assignments.txt','w') as assign_f:
            for i, vec in enumerate(written_tmat):
                assign_f.write("%d\n" % i)
                new_vec = np.abs(vec) / np.max(np.abs(vec))
                sort_vec_idx = np.argsort(new_vec)[::-1]
                sorted_vec = vec[sort_vec_idx]
                name = self.coordinate_func.get_int_names()
                sorted_name = [name[i] for i in sort_vec_idx]
                for hu in range(self.num_vibs):
                    assign_f.write("%s %d %5.12f " % (sorted_name[hu], sort_vec_idx[hu], sorted_vec[hu]))
                assign_f.write('\n \n')
            assign_f.close()

    def calc_normal_modes(self, gmat, internal_coordinates, save_nms=True):
        print('Begin Calculation of Normal Modes...')
        my_calc_q = CalcQCoords(gmat, internal_coordinates, self.dw)
        transformation_matrix, nms = my_calc_q.run()
        self.save_assignments(transformation_matrix)
        if save_nms:
            np.save(f"{self.res_dir}/nms.npy", nms)
            np.save(f"{self.res_dir}/trans_mat.npy", transformation_matrix)
        return nms


class GSPA:
    def __init__(self,
                 res_dir,
                 normal_modes,
                 desc_weights,
                 potential_energies,
                 dipoles,
                 ham_overlap):
        self.res_dir = res_dir
        self.nms = normal_modes
        self.desc_weights = desc_weights
        self.vs = potential_energies
        self.dips = dipoles
        self.ham = ham_overlap

    def run(self):
        print('Begin GSPA Approximation Code...')
        my_eng = CalcEngsInts(nms=self.nms,
                              potentials=self.vs,
                              desc_weights=self.desc_weights,
                              dipoles=self.dips)
        if not self.ham:
            energies, intensities = my_eng.run()
            labz = ['Fundamentals', 'Overtones','Combinations']
            for eng_num, eng in enumerate(energies):
                eng = Constants.convert(eng, 'wavenumbers', to_AU=False)
                print(f'{labz[eng_num]}:')
                print(np.column_stack((eng, intensities[eng_num])))

            np.savez(f'{self.res_dir}/energies.npz',
                     funds=energies[0],
                     overs=energies[1],
                     combos=energies[2])
            np.savez(f'{self.res_dir}/intensities.npz',
                     funds=intensities[0],
                     overs=intensities[1],
                     combos=intensities[2])
        else:
            overlap, hamiltonian = my_eng.calc_ham_mat()
            np.savez(f'{self.res_dir}/ov_ham.npz',
                     ov=overlap,
                     ham=hamiltonian)