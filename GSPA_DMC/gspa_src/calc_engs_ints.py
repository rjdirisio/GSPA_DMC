import numpy as np
import itertools as itt

class CalcEngsInts:
    def __init__(self,
                 nms,
                 desc_weights,
                 potentials,
                 dipoles):
        self.q = nms
        self.dw = desc_weights
        self.vs = potentials
        self.dips = dipoles
        self._initialize()

    def _initialize(self):
        # Useful exp vals to store
        self.expv_q2 = np.average(self.q ** 2, weights=self.dw, axis=0)
        self.expv_q4 = np.average(self.q ** 4, weights=self.dw, axis=0)
        self.expv_q8 = np.average(self.q ** 8, weights=self.dw, axis=0)
        self.v0 = np.average(self.vs, weights=self.dw)
        # Overtone polynomials + exp val
        a = -1 / self.expv_q2
        b = np.average(self.q ** 3 / self.expv_q2, axis=0, weights=self.dw) * (1 / self.expv_q2)
        self.overtone_poly = a * self.q ** 2 + b * self.q + 1
        self.expv_over2 = np.average(self.overtone_poly ** 2, axis=0, weights=self.dw)
        # Combionation states
        self.all_combos = list(itt.combinations(range(len(self.q.T)), 2))

    def kinetic_one_quantum(self):
        beta = self.expv_q2 / (self.expv_q4 - self.expv_q2 ** 2)
        return beta / 2

    def kinetic_two_quanta(self):
        beta = np.sqrt(8 * self.expv_q4 / (self.expv_q8 - self.expv_q4 ** 2))
        return 2 * beta / 2

    def calc_funds(self):
        # Calculate <V_1> - <V_0>
        fund_v = np.average(self.q ** 2 * self.vs[:,None], axis=0, weights=self.dw) / self.expv_q2
        fund_dv = fund_v - self.v0
        # Calculate <T_1> - <T_0>
        fund_dt = self.kinetic_one_quantum()
        fundamentals = fund_dt + fund_dv
        return fundamentals

    def calc_overtones(self):
        # Calculate <V_2> - <V_0>
        over_v = np.average(self.overtone_poly * self.vs[:,None] * self.overtone_poly, axis=0, weights=self.dw) / \
                 self.expv_over2
        over_dv = over_v - self.v0
        # Calculate <T_1> - <T_0>
        over_dt = self.kinetic_two_quanta()
        overtones = over_dt + over_dv
        return overtones

    def calc_combos(self):

        combo_tot = []
        fund_dts = self.kinetic_one_quantum()
        for combo_1, combo_2 in self.all_combos:
            q1xq2 = self.q[:, combo_1] * self.q[:, combo_2]
            combo_v = np.average(q1xq2 * self.vs * q1xq2, weights=self.dw) / \
                      np.average(q1xq2**2, weights=self.dw)
            combo_dv = combo_v - self.v0
            combo_tot.append(fund_dts[combo_1]+fund_dts[combo_2] + combo_dv)
        return np.array(combo_tot)

    def calc_freqs(self):
        funds = self.calc_funds()
        overs = self.calc_overtones()
        combos = self.calc_combos()
        return funds, overs, combos

    def calc_ints(self):
        # <1 | u | 0>
        fund_mus = np.average(self.q[:,np.newaxis,:] * self.dips[:,:,np.newaxis], weights=self.dw, axis=0)
        fund_mus /= np.sqrt(self.expv_q2)
        fund_ints = np.linalg.norm(fund_mus, axis=0)**2

        # <2 | u | 0 >
        over_mus = np.average(self.overtone_poly[:,np.newaxis,:] * self.dips[:,:,np.newaxis], weights=self.dw, axis=0)
        over_mus /= np.sqrt(self.expv_over2)
        over_ints = np.linalg.norm(over_mus, axis=0) ** 2

        # < 1,1 | U | 0 >
        combo_ints = []
        combo_mus = []
        for combo_1, combo_2 in self.all_combos:
            q1xq2 = self.q[:, combo_1] * self.q[:, combo_2]
            combo_mu = np.average(q1xq2[:,np.newaxis]* self.dips , axis=0, weights=self.dw) / \
                      np.sqrt(np.average(q1xq2**2, weights=self.dw))
            combo_mus.append(combo_mu)
            combo_ints.append(np.linalg.norm(combo_mu)**2)
        combo_mus = np.array(combo_mus)
        combo_ints = np.array(combo_ints)
        return (fund_ints, over_ints, combo_ints), np.vstack((fund_mus.T, over_mus.T, combo_mus))

    def run(self):
        energies = self.calc_freqs()
        intensities, mus = self.calc_ints()
        return energies, intensities, mus


    def calc_ham_mat(self,energies):
        from .calc_hamiltonian import CalcHamOverlap
        this_ham = CalcHamOverlap(self, energies)
        overlap, ham = this_ham.run()
        return overlap, ham