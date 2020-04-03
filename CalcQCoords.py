import numpy as np
import scipy.linalg as sla
from .CalcGMat import Gmat
from .internalCoordinates import intCds
from molecularInfo import molecInfo

class CalcQCoords:
    def __init__(self,
                 coords,
                 atoms,
                 weights,
                 intCdFun,
                 tmat=None):
        """
        Take in atoms, coordinates, descendant weights, an internal coordinate
        function to calculate the 'q' coordinates for a DMC wave function.
        If given a tmat, it will skip calculating second moments, and just calculate
        moments and the q coordinates.
        @param coords: ensemble of walkers
        @type coords: np.ndarray
        @param atoms: list of atom strings for G-matrix
        @type atoms: list
        @param weights: descendant weights
        @type weights: np.ndarray
        @param intCdFun: the function that calculates the internal coordinates for the system
        @type intCdFun: intCds classmethod
        @param tmat: if you have a transformation matrix from a previous calc, feed it in here!
        """
        self.coords = coords
        self.atoms = atoms
        self.weights=weights
        self.intCds = intCdFun
        self.nVibs = 3*len(atoms)-6
        self.tmat = tmat

    def _firstMoments(self):
        intz = self.intCds(self.coords)
        dispz = intz - np.average(intz,axis=0,weights=self.weights)
        return dispz

    def _secondMoments(self,dispz):
        """
        Average second moments matrix
        @param dispz: the 'first moments' r - <r>
        @type dispz: np.ndarray
        @return: avgSecondMoments
        """
        try:
            smom = molecInfo.expVal(dispz[:,:,np.newaxis]*dispz[:,np.newaxis,:],weights=dw)
        except MemoryError:
            print("Memory constraints on second moments matrix. Looping..")
            walkerSize = len(self.coords)
            smom = np.zeros((self.nVibs,self.nVibs))
            for i in range(len(walkerSize)):
                smom += np.outer(dispz[i],dispz[i])*self.weights
            smom /= np.sum(self.weights)
        return smom

    @staticmethod
    def _calcinvSqrt(G):
        w, v = np.linalg.eigh(G)
        invRootDiagG = np.diag(1.0 / np.sqrt(w))
        invRootG = np.dot(v, np.dot(invRootDiagG, v.T))
        return invRootG

    def get_Qcoords(self):
        #Calculate average Gmatrix for ensemble of walkers
        if self.tmat is None:
            gmO = Gmat(self.coords,
                     self.atoms,
                     weights=self.weights,
                     intCdFun=self.intCds)
            avgG = gmO.calcGmats()
            #Calculate matrix of second moments
            momz = self._firstMoments()
            mu2avg = self._secondMoments(momz)
            #~linear algebra~
            # evals,evecs = sla.eigh(mu2avg,avgG)
            gm12 = self._calcinvSqrt(avgG)
            mu2avgP = np.dot(gm12, np.dot(mu2avg, gm12))  # mass weights g^-1/2 . sm . g^-1/2
            eval, evec = np.linalg.eigh(mu2avgP)
            Tmat = np.dot(evec.T, gm12)
        else:
            Tmat = self.tmat
        q = np.matmul(Tmat, momz.T).T
        return q