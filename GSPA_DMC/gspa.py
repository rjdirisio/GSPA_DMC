from .normal_mode_src import *
from .gspa_src import *

from pyvibdmc.simulation_utilities import Constants


class NormalModes:
    def __init__(self,
                 run_name,
                 atoms,
                 walkers,
                 descendant_weights,
                 ic_manager,
                 atomic_units=True,
                 masses=None
                 ):
        self.run_name = run_name
        self.atoms = atoms
        self.coords = walkers
        self.dw = descendant_weights
        self.atomic_units = atomic_units
        self.masses = masses
        self.coordinate_func = ic_manager
        self._initialize()

    def _initialize(self):
        if not self.atomic_units:
            self.coords = Constants.convert(self.coords, 'angstroms', to_AU=True)
        if self.masses is None:
            self.masses = np.array([Constants.mass(a) for a in self.atoms])
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
        np.save(f"{self.run_name}_gmat.npy", avg_gmat)
        int_cds = self.coordinate_func.get_ints(self.coords)
        return avg_gmat, int_cds

    def save_assignments(self,trans_mat):
        written_tmat = np.copy(trans_mat)
        with open(f'{self.run_name}_assignments.txt','w') as assign_f:
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
            np.save(f"{self.run_name}_nms.npy", nms)
            np.save(f"{self.run_name}_trans_mat.npy", transformation_matrix)
        return nms


class GSPA:
    def __init__(self,
                 run_name,
                 normal_modes,
                 potential_energies,
                 dipoles,
                 ham_overlap):
        self.run_name = run_name
        self.nms = normal_modes
        self.vs = potential_energies
        self.dips = dipoles
        self.ham = ham_overlap

    def run(self):
        print('Begin GSPA Approximation Code...')
        my_eng = CalcEngsInts(nms=self.normal_modes,
                              potentials=self.vs,
                              dipoles=self.dips,
                              ham_overlap=self.ham)
        if not self.ham:
            energies, intensities = my_eng.run()
            np.save(f'{self.run_name}_energies.npy', energies)
            np.save(f'{self.run_name}_intensities.npy', intensities)
        else:
            energies, intensities, overlap, ham = my_eng.run()
            np.save(f'{self.run_name}_energies.npy', energies)
            np.save(f'{self.run_name}_intensities.npy', intensities)
            np.save(f'{self.run_name}_ov_mat.npy', overlap)
            np.save(f'{self.run_name}_ham_mat.npy', ham)
        print('Done.')
