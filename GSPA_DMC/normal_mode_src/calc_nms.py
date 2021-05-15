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

    def second_moments(self,moments):
        try:
            smom = np.average(moments[:, :, np.newaxis] * moments[:, np.newaxis, :],
                              axis=0,
                              weights=self.dw)
        except MemoryError:
            print("Memory constraints on second moments matrix. Looping..")
            walker_size = len(self.int_cds)
            vibz = len(self.int_cds.T)
            smom2 = np.zeros((vibz, vibz))
            for i in range(walker_size):
                smom2 += np.outer(moments[i], moments[i]) * self.dw[i]
            smom2 /= np.sum(self.weights)
        return smom

    @staticmethod
    def _calc_inv_sqrt(g_mat):
        w, v = np.linalg.eigh(g_mat)
        inv_root_diag_g = np.diag(1.0 / np.sqrt(w))
        inv_root_g = np.dot(v, np.dot(inv_root_diag_g, v.T))
        return inv_root_g

    def run(self):
        moments = self.first_moments()
        second_moments = self.second_moments(moments)
        # from scipy.linalg import eigh as seigh
        # seval,sevec = seigh(second_moments,self.gmat) # gives you trans_mat.T
        gm12 = self._calc_inv_sqrt(self.gmat)
        g_s_g = np.dot(gm12, np.dot(second_moments, gm12))  # g^-1/2 . sm . g^-1/2
        evals, evecs = np.linalg.eigh(g_s_g)
        print('Eigenvalues of mass-weighted second moments:')
        print(evals)
        trans_mat = np.dot(evecs.T, gm12)

        nms = np.matmul(trans_mat, moments.T).T
        return trans_mat, nms
