import itertools as itt

import numpy as np


class CalcHamOverlap:
    def __init__(self, eng_obj, energies):
        self.eng_obj = eng_obj
        self.energies = energies
        self._initialize()

    def _initialize(self):
        """
        Calculate important quantities and extract data from eng_obj
        """
        self.num_vibs = len(self.eng_obj.q.T)
        self.combo_inds = np.array(self.eng_obj.all_combos)
        self.num_combos = len(self.combo_inds)
        # fundamentals, overtones, combinations
        self.overlap_mat = np.zeros((self.num_vibs + self.num_vibs + self.num_combos,
                                     self.num_vibs + self.num_vibs + self.num_combos))
        self.ham_mat = np.copy(self.overlap_mat)
        self.q = self.eng_obj.q
        self.over = self.eng_obj.overtone_poly
        self.dw = self.eng_obj.dw
        self.block_inds = np.arange(self.num_vibs)

    def fund_funds(self):
        # On diags
        self.overlap_mat[self.block_inds, self.block_inds] = np.average(self.q ** 2, axis=0, weights=self.dw)
        # Off diags
        mat_inds = np.triu_indices_from(np.zeros((self.num_vibs, self.num_vibs)), k=1)
        fund_each_other = self.q[:, self.combo_inds[:, 0]] * self.q[:, self.combo_inds[:, 1]]
        self.overlap_mat[mat_inds] = np.average(fund_each_other, axis=0, weights=self.dw)

    def fund_overs(self):
        # Off diags
        mat_inds = np.array([self.num_vibs, self.num_vibs * 2])
        # Outer product of q and over to get all combos of the two for assignment
        fund_w_overs = self.q[:, :, np.newaxis] * self.over[:, np.newaxis, :]
        self.overlap_mat[self.block_inds, mat_inds[0]:mat_inds[1]] = np.average(fund_w_overs, axis=0, weights=self.dw)

    def fund_combos(self):
        # Off diags
        mat_inds = np.array([self.num_vibs * 2, self.num_vibs * 2 + self.num_combos])
        # Outer product
        combos = self.q[:, self.combo_inds[:, 0]] * self.q[:, self.combo_inds[:, 1]]
        tot_f_c = []
        for i in range(self.num_vibs):
            fund_w_combos = np.average(self.q[:, i, np.newaxis] * combos, axis=0, weights=self.dw)
            tot_f_c.append(fund_w_combos)
        self.overlap_mat[self.block_inds, mat_inds[0]:mat_inds[1]] = tot_f_c

    def over_overs(self):
        # On diags
        self.block_inds_2 = self.block_inds + self.num_vibs
        self.overlap_mat[self.block_inds_2, self.block_inds_2] = np.average(self.over ** 2, axis=0, weights=self.dw)
        # Off diags
        mat_inds = np.array(np.triu_indices_from(np.zeros((self.num_vibs, self.num_vibs)), k=1))
        mat_inds += self.num_vibs
        over_each_other = self.over[:, self.combo_inds[:, 0]] * self.over[:, self.combo_inds[:, 1]]
        self.overlap_mat[mat_inds[0], mat_inds[1]] = np.average(over_each_other, axis=0, weights=self.dw)

    def over_combos(self):
        # Off diags
        mat_inds = np.array([self.num_vibs * 2, self.num_vibs * 2 + self.num_combos])
        # Outer product
        combos = self.q[:, self.combo_inds[:, 0]] * self.q[:, self.combo_inds[:, 1]]
        tot_o_c = []
        for i in range(self.num_vibs):
            over_w_combos = np.average(self.over[:, i, np.newaxis] * combos, axis=0, weights=self.dw)
            tot_o_c.append(over_w_combos)
        self.overlap_mat[self.block_inds_2, mat_inds[0]:mat_inds[1]] = tot_o_c
        return None

    def combo_combos(self):
        # On diagonals
        block_inds_3 = np.arange(self.num_combos)+ self.num_vibs * 2
        q_n_q_m = self.q[:, self.combo_inds[:, 0]] * self.q[:, self.combo_inds[:, 1]]
        self.overlap_mat[block_inds_3, block_inds_3] = np.average(q_n_q_m ** 2, axis=0, weights=self.dw)

        # Off diagonals
        mat_inds = np.array(np.triu_indices_from(np.zeros((self.num_combos, self.num_combos)), k=1)) + self.num_vibs * 2
        tot_c_c = []
        for i in range(1,self.num_combos):
            comb_w_comb = np.average(q_n_q_m[:,i-1, np.newaxis] * q_n_q_m[:,i:], axis=0, weights=self.dw)
            tot_c_c.extend(comb_w_comb)
        self.overlap_mat[mat_inds[0],mat_inds[1]] = tot_c_c
        return None

    def run(self):
        self.fund_funds()
        self.fund_overs()
        self.fund_combos()
        self.over_overs()
        self.over_combos()
        self.combo_combos()
        # Add normalization
        diag_ov = np.copy(np.diagonal(self.overlap_mat))
        self.overlap_mat = self.overlap_mat/np.sqrt(diag_ov[:,np.newaxis])
        self.overlap_mat = self.overlap_mat/np.sqrt(diag_ov)
        self.overlap_mat = self.overlap_mat + self.overlap_mat.T - np.eye(len(self.overlap_mat))
        # Ham mat cleanup with normalization
        self.ham_mat = self.ham_mat/np.sqrt(diag_ov[:,np.newaxis])
        self.ham_mat = self.ham_mat/np.sqrt(diag_ov)
        self.ham_mat = self.ham_mat + self.ham_mat.T
        # Ham mat is missing second term: <P*V*P> - <V><P*P>
        msk = np.copy(self.overlap_mat)
        msk = msk * self.eng_obj.v0
        np.fill_diagonal(msk, np.zeros(len(msk)))
        self.ham_mat = self.ham_mat - msk
        # Put in transition frequencies
        np.fill_diagonal(self.ham_mat,np.concatenate(self.energies))
        return self.overlap_mat, self.ham_mat
