import numpy as np
from .calcGmat import *
from .intCds import *

class CalcQCoords:
    def __init__(self,
                 coords,
                 atoms,
                 weights,
                 intCdFun):
        self.coords = coords
        self.atoms = atoms
        self.weights=weights
        self.intCds = intCdFun

    def _secondMoments(self):
        self.intCds


    def get_Qcoords(self):
        #Calculate average Gmatrix for ensemble of walkers
        gmat = calcGmat.calcGmats(self.coords,
                           self.atoms,
                           self.weights,
                           self.intCds)
        #Calculate matrix of second moments
