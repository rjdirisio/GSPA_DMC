import numpy as np

class SymmetrizeWfn:
    @staticmethod
    def swap_two_atoms(cds, dws, atm_1,atm_2):
        """Given two atoms, return a copy of the walkers in which the two atoms are swapped"""
        cop = np.copy(cds)
        cop[:,[atm_1, atm_2]] = cop[:,[atm_2, atm_1]]
        tot_cds = np.concatenate((cds,cop))
        return tot_cds, np.concatenate([dws,dws])

    @staticmethod
    def swap_group(cds, dws, atm_list_1, atm_list_2):
        """Given a list of atoms, return a copy of the walkers in where the two atom groups are swapped
        (given [0,1,2] and [3,4,5], we will then get a copy of the cds array with ordering [3,4,5] and [0,1,2])"""
        tot_atms = np.arange(cds.shape[1])
        tot_atms_c = np.copy(tot_atms)
        tot_atms_c[atm_list_1] = atm_list_2
        tot_atms_c[atm_list_2] = atm_list_1
        cop = np.copy(cds[:,tot_atms_c])
        return np.concatenate((cds,cop)), np.concatenate([dws,dws])

    @staticmethod
    def reflect_about_xyp(cds, dws):
        """Reflect about the xy-plane by flipping z component of all atoms"""
        cop = np.copy(cds)
        cop[:,:,-1] = -1 * cop[:,:,-1]
        return np.concatenate((cds, cop)), np.concatenate([dws,dws])
