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
        self.expv_q2 = np.average(self.q ** 2, weights=self.dw, axis=0)
        self.expv_q4 = np.average(self.q ** 4, weights=self.dw, axis=0)
        self.expv_q8 = np.average(self.q ** 8, weights=self.dw, axis=0)
        self.v0 = np.average(self.vs, weights=self.dw)

    def kinetic_one_quantum(self):
        beta = self.expv_q2 / (self.expv_q4 - self.expv_q2 ** 2)
        return beta / 2

    def kinetic_two_quanta(self):
        beta = 8 * self.expv_q4 / (self.expv_q8 - self.expv_q4 ** 2)
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
        a = -1 / self.expv_q2
        b = np.average(self.q ** 3 / self.expv_q2, axis=0, weights=self.dw) * (1 / self.expv_q2)
        overtone_poly = a * self.q ** 2 + b * self.q + 1
        over_v = np.average(overtone_poly * self.vs[:,None] * overtone_poly, axis=0, weights=self.dw) / \
                 np.average(overtone_poly ** 2, axis=0, weights=self.dw)
        over_dv = over_v - self.v0
        # Calculate <T_1> - <T_0>
        over_dt = self.kinetic_two_quanta()
        overtones = over_dt + over_dv
        return overtones

    def calc_combos(self):
        all_combos = itt.combinations(range(len(self.q.T)), 2)
        combo_tot = []
        fund_dts = self.kinetic_one_quantum()
        for combo_1, combo_2 in all_combos:
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
        dip = self.dips
        intensities = []
        return intensities

    def run(self):
        energies = self.calc_freqs()
        intensities = self.calc_ints()
        return energies, intensities
