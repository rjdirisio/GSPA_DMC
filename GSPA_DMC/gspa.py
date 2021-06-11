from .normal_mode_src import *
from .gspa_src import *

import os
import itertools as itt

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
            print('No res_dir found. creating...')
            os.makedirs(self.res_dir)
        if not os.path.isdir(f'{self.res_dir}/red_ham/'):
            os.makedirs(f'{self.res_dir}/red_ham/')
        # Masses and atoms
        if not self.atomic_units:
            self.coords = Constants.convert(self.coords, 'angstroms', to_AU=True)
        if self.masses is None:
            self.masses = np.array([Constants.mass(a) for a in self.atoms])
        # Useful variables that don't need to be recalculated all the time
        self.num_atoms = len(self.atoms)
        self.num_vibs = 3 * self.num_atoms - 6

    def _save_assignments(self,trans_mat):
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

    def calc_gmat(self):
        print("Calculating internal coordinates for r-<r>")
        int_cds = self.coordinate_func.get_ints(self.coords)
        if int_cds.shape[1] != self.num_vibs: #num_walkers x three_n_minus_6
            print(f"WARNING: Internal coordinates are NOT num_walkers x 3n-6. Instead, it's {int_cds.shape}")
            print(f"Make sure that you are doing a reduced dimensional calculation. Otherwise, FIX!!!!")
        print('Begin Calculation of G-Matrix...')
        this_gmat = Gmat(coords=self.coords,
                                    masses=self.masses,
                                    num_vibs=self.num_vibs,
                                    dw=self.dw,
                                    coordinate_func=self.coordinate_func)
        avg_gmat, internals = this_gmat.run()
        np.save(f"{self.res_dir}/gmat.npy", avg_gmat)
        return avg_gmat, int_cds

    def calc_normal_modes(self, gmat, internal_coordinates, save_nms=True):
        print('Begin Calculation of Normal Modes...')
        my_calc_q = CalcQCoords(gmat, internal_coordinates, self.dw)
        transformation_matrix, nms = my_calc_q.run()
        self._save_assignments(transformation_matrix)
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
                 ham_overlap,
                 text_file=True):
        self.res_dir = res_dir
        self.nms = normal_modes
        self.desc_weights = desc_weights
        self.vs = potential_energies
        self.dips = dipoles
        self.ham = ham_overlap
        self.text_file = text_file
        self._initialize()

    def _initialize(self):
        if not os.path.isdir(self.res_dir):
            print('No res_dir found. creating...')
            os.makedirs(self.res_dir)
        if not os.path.isdir(f'{self.res_dir}/red_ham/'):
            os.makedirs(f'{self.res_dir}/red_ham/')

    def _assignment_order(self):
        num_modes = len(self.nms.T)
        fund_assign = np.column_stack((np.arange(num_modes), np.array(np.repeat(999,num_modes))))
        over_assign = np.column_stack((np.arange(num_modes), (np.arange(num_modes))))
        combo_assign = np.array(list(itt.combinations(range(num_modes), 2)))
        assignments_ordering = np.concatenate((fund_assign,over_assign,combo_assign))
        return assignments_ordering

    def _write_text(self,energies_wvn,intensities,assignment_order):
        energies_wvn_tot = np.concatenate(energies_wvn)
        print(energies_wvn_tot)
        intensities_tot = np.concatenate(intensities)
        with open(f'{self.res_dir}/all_transitions.txt','w') as fll:
            fll.write(f'E\t I\t Assign_1  Assign_2\n')
            for i in range(len(energies_wvn_tot)):
                fll.write(f'{energies_wvn_tot[i]}\t {intensities_tot[i]}\t {assignment_order[i][0]}  {assignment_order[i][1]}\n')


    def run(self):
        print('Begin GSPA Approximation Code...')
        my_eng = CalcEngsInts(nms=self.nms,
                              potentials=self.vs,
                              desc_weights=self.desc_weights,
                              dipoles=self.dips)
        energies, intensities, mus = my_eng.run()

        labz = ['Fundamentals', 'Overtones','Combinations']
        energies_wvn = []
        for eng_num, eng in enumerate(energies):
            eng = Constants.convert(eng, 'wavenumbers', to_AU=False)
            energies_wvn.append(eng)
            print(f'{labz[eng_num]}:')
            print(np.column_stack((eng, intensities[eng_num])))

        np.savez(f'{self.res_dir}/energies.npz',
                 funds=energies_wvn[0],
                 overs=energies_wvn[1],
                 combos=energies_wvn[2])
        np.savez(f'{self.res_dir}/intensities.npz',
                 funds=intensities[0],
                 overs=intensities[1],
                 combos=intensities[2])
        assignment_order = self._assignment_order()
        np.save(f'{self.res_dir}/assign_order.npy',
                assignment_order)

        if self.text_file:
            self._write_text(energies_wvn, intensities, assignment_order)

        if self.ham:
            overlap, hamiltonian = my_eng.calc_ham_mat(energies_wvn)
            np.savez(f'{self.res_dir}/red_ham/ov_ham.npz',
                     ov=overlap,
                     ham=hamiltonian,
                     mus=mus)
