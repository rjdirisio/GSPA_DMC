import numpy as np
from researchUtils import Constants

class Gmat:
    def __init__(self,
                 coords,
                 atoms,
                 weights=None,
                 intCdFun=None):
        self.coords = coords
        self.atoms = atoms
        self.weights = weights
        self.intCds = intCdFun

    def calcGmats(self):
        """Returns the stack of internal coordinate g-matrices that cooresponds to the geometry of interest
        @param dw: the weights for the weighted average of the gmatrix. This is used for DMC wave functions.
        Ignore otherwise
        @return:
        """
        if self.weights is not None:
            avg = True
        if len(self.coords.shape) == 2:
            self.coords = np.expand_dims(self.coords,0)
        dx = 1e-4
        mass = Constants.mass(self.atoms,to_AU=True)
        nAtoms = len(self.coords[0])
        nVibs = 3*nAtoms-6
        if not avg:
            gmatz = np.zeros((len(self.coords,nVibs,nVibs)))
        else:
            gnm = np.zeros((nVibs,nVibs))
        for atom in range(nAtoms):
            for coordinate in range(3):
                print(f'dx {atom*3+(coordinate+1)}')
                print(f'atom: {atom}, coordinate {coordinate}')
                delx=np.zeros((self.coords.shape))
                delx[:,atom,coordinate]+=dx #perturbs the x,y,z coordinate of the atom of interest
                coordPlus=self.intCds(self.coords+delx)
                coordMinus=self.intCds(self.coords-delx)
                #For 2pi stuff going from 0 to 2pi
                coordPlus[np.abs(coordPlus - coordMinus) > 1.0] += (-1.0 * 2. * np.pi) * np.sign(coordPlus[np.abs(coordPlus - coordMinus) > 1.0])
                partialderv=(coordPlus-coordMinus)/(2.0*dx) #Finite Diff
                if avg:
                    for i, pd in enumerate(partialderv): #memory constraints. loop...
                        mwpd2 = (partialderv[i, :, np.newaxis] * partialderv[i, np.newaxis, :]) / mass[atom]
                        gnm += mwpd2 * self.weights[i]
                else:
                    mwpd2 = (partialderv[:, :, np.newaxis] * partialderv[:, np.newaxis, :]) / mass[atom]
                    gmatz += mwpd2
        if avg:
            gnm /= np.sum(self.weights)
            return gnm
        else:
            return gmatz