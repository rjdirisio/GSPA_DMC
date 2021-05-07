import numpy as np


class CalcQCoords:
    def __init__(self,
                 gmat,
                 internal_coordinates,
                 descendant_weights):

        self.gmat = gmat
        self.int_cds = internal_coordinates
        self.dw = descendant_weights

    def first_moments(self):
        dispz = self.int_cds - np.average(self.int_cds, axis=0, weights=self.dw)
        return dispz

    def second_moments(self):
        try:
            smom = np.average(self.int_cds[:, :, np.newaxis] * self.int_cds[:, np.newaxis, :],
                              axis=0,
                              weights=self.dw)
        except MemoryError:
            print("Memory constraints on second moments matrix. Looping..")
            walker_size = len(self.coords)
            smom = np.zeros((self.nVibs, self.nVibs))
            for i in range(walker_size):
                smom += np.outer(self.int_cds[i], self.int_cds[i]) * self.weights
            smom /= np.sum(self.weights)
        return smom

    @staticmethod
    def _calc_inv_sqrt(g_mat):
        w, v = np.linalg.eigh(g_mat)
        inv_root_diag_g = np.diag(1.0 / np.sqrt(w))
        inv_root_g = np.dot(v, np.dot(inv_root_diag_g, v.T))
        return inv_root_g

    def run(self):
        moments = self.first_moments()
        second_moments = self.second_moments()
        gm12 = self._calc_inv_sqrt(self.gmat)
        g_s_g = np.dot(gm12, np.dot(second_moments, gm12))  # g^-1/2 . sm . g^-1/2
        eval, evec = np.linalg.eigh(g_s_g)
        print('Eigenvalues of mass-weighted second moments:')
        print(eval)
        trans_mat = np.dot(evec.T, gm12)

        nms = np.matmul(trans_mat, moments.T).T
        return trans_mat, nms
