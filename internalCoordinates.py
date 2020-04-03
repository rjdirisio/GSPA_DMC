import numpy as np
from molecularInfo import *
from researchUtils import Constants

class intCds:
    """A class to basically store the internal coordiantes I define for a particular molecule. Uses molecularInfo"""

    @staticmethod
    def _EulerTrimer(xx,trim,outerO,centralO,outerH1,outerH2):
        """
        @param xx: coordinates
        @type np.ndarray
        @param trim: molecularInfo object used to generate attribute
        @type: molecInfo
        @param outerO:
        @param centralO:
        @param outerH1:
        @param outerH2:
        @return:
        """
        OOvec = xx[:, outerO, :] - xx[:,centralO]
        X = OOvec / la.norm(OOvec,axis=1)[:,np.newaxis]
        #always o3o1 cross o3o2
        crs = np.cross(xx[:, 1 - 1] - xx[:, 3 - 1], xx[:, 2 - 1] - xx[:, 3 - 1], axis=1)

        # for symmetrization, axis flips when reflection occurs.
        Z = crs / la.norm(crs, axis=1)[:, np.newaxis]
        Z[len(Z) / 2:] *= -1.0
        Y = np.cross(Z, X, axis=1)
        Z[len(Z) / 2:] *= -1.0
        #end...

        x, y, z = self.H9GetHOHAxis(xx[:, O1 - 1], xx[:, h1 - 1], xx[:, h2 - 1])
        trim.
        exx = np.copy(x)
        x = np.copy(y)
        y = np.copy(z)
        z = np.copy(exx)
        print('lets get weird')
        exX = np.copy(X)
        X = np.copy(Z)
        Z = np.copy(Y)
        Y = np.copy(exX)
        Theta, tanPhi, tanChi = self.eulerMatrix(x, y, z, X, Y, Z)
        return Theta, tanPhi, tanChi

    @classmethod
    def H7O3p_internals(cls,xx,eckRot=True):
        """My definition for H7O3+ internals."""
        """Atom order: O O O H H H H H H H H"""
        nVibs = 24
        if eckRot:
            atmStr = ['O', 'O', 'O'] #just oxygens
            com,rotMs = molRotator.genEckart(geoms=xx[:,[0,1,2]],
                                 refGeom=np.load("refGeoms/H7O3p.npy"),
                                 masses=Constants.masses(atmStr),
                                 planar=True,
                                 retMat=False)
            xx -= com[:,np.newaxis]
            xx = molRotator.rotateGeoms(rotMs=rotMs,
                                   geoms=xx)
        #Internals now
        trim = molecInfo(xx)
        oh8 = xx[:, 8 - 1] - xx[:, 3 - 1]
        oh9 = xx[:, 9 - 1] - xx[:, 3 - 1]
        oh10 = xx[:, 10 - 1] - xx[:, 3 - 1]
        a = np.zeros(len(xx),nVibs)
        a[:,:3] = trim.cartToSpherical(oh8)
        a[:,2] = trim.bondAngle(8-1,3-1,9-1) - trim.bondAngle(8-1,3-1,10-1)
        a[:,3:6] = trim.cartToSpherical(oh9)
        a[:,6:9] = trim.cartToSpherical(oh10)
        a[:,9] = trim.bondLength(1-1,4-1)
        a[:,10] = trim.bondLength(1-1,5-1)
        a[:,11] = trim.bondAngle(5-1,1-1,4-1)
        a[:,12] = trim.bondLength(2-1,6-1)
        a[:,13] = trim.bondLength(2-1,7-1)
        a[:,14] = trim.bondAngle(1-1,3-1,2-1)

        a[:,15:18]
        a[:,18:21]
        a[:,21] = trim.bondLength(1-1,3-1)
        a[:,22] = trim.bondLength(2-1,3-1)
        a[:,23] = trim.bondAngle(1-1,3-1,2-1)
        return internals