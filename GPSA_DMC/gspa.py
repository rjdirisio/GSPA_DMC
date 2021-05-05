from .normal_mode_src import *
from .gspa_src import *

from pyvibdmc.simulation_utilities import Constants


class NormalModes:
    def __init__(self,
                 run_name,
                 atoms,
                 walkers,
                 descendant_weights,
                 internal_coord_func,
                 atomic_units=True,
                 masses=None
                 ):
        self.run_name = run_name
        self.atoms = atoms
        self.coords = walkers
        self.dw = descendant_weights
        self.atomic_units = atomic_units
        self.masses = masses
        self.coordinate_func = internal_coord_func
        self._initialize()

    def _initialize(self):
        if not self.atomic_units:
            self.coords = Constants.convert(self.coords, 'angstroms', to_AU=True)
        if self.masses is None:
            self.masses = np.array([Constants.mass(a) for a in self.atoms])
        self.num_atoms = len(self.atoms)
        if self.num_atoms > 3:
            self.num_vibs = 3 * self.num_atoms - 6
        else:
            raise Exception("WARNING: FEWER THAN 3 ATOMS ENTERED. ARE YOU SURE YOU WANT TO DO THIS?")

    def calc_gmat(self):
        print('Begin Calculation of G-Matrix...')
        this_gmat, internals = Gmat(coords=self.coords,
                                    masses=self.masses,
                                    num_vibs=self.num_vibs,
                                    dw=self.dw,
                                    coordinate_func=self.coordinate_func)
        avg_gmat = this_gmat.run()
        np.save(f"{self.run_name}_gmat.npy", avg_gmat)
        int_cds = self.coordinate_func(self.coords)
        return avg_gmat, int_cds

    def calc_normal_modes(self, gmat, internal_coordinates, save_nms=True):
        print('Begin Calculation of Normal Modes...')
        my_calc_q = CalcQCoords(gmat, internal_coordinates, self.dw)
        transformation_matrix, nms = my_calc_q.run()
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
