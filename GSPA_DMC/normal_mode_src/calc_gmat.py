import numpy as np


class Gmat:
    def __init__(self,
                 coords,
                 masses,
                 num_vibs,
                 dw,
                 coordinate_func,
                 dx=1e-4):
        self.coords = coords
        self.masses = masses
        self.num_vibs = num_vibs
        self.dw = dw
        self.c_func = coordinate_func
        self.dx = dx

    def run(self):
        if len(self.coords.shape) == 2:
            self.coords = [self.coords, self.coords]
            self.dw = [self.dw, self.dw]
        gnm = np.zeros((self.num_vibs, self.num_vibs))
        for atom in range(len(self.masses)):
            for coordinate in range(3):
                print(f'dx {atom * 3 + (coordinate + 1)}')
                print(f'atom: {atom}, coordinate {coordinate}')
                delx = np.zeros(self.coords.shape)
                delx[:, atom, coordinate] += self.dx  # perturbs the x,y,z coordinate of the atom of interest
                coord_plus = self.c_func.get_ints(self.coords + delx)
                coord_minus = self.c_func.get_ints(self.coords - delx)
                # For 2pi stuff going from 0 to 2pi
                coord_plus[np.abs(coord_plus - coord_minus) > 1.0] += \
                    (-1.0 * 2. * np.pi) * np.sign(coord_plus[np.abs(coord_plus - coord_minus) > 1.0])
                partialderv = (coord_plus - coord_minus) / (2.0 * self.dx)  # Finite Diff
                for i, pd in enumerate(partialderv):  # memory constraints. loop...
                    mwpd2 = (partialderv[i, :, np.newaxis] * partialderv[i, np.newaxis, :]) / self.masses[atom]
                    gnm += mwpd2 * self.dw[i]
        print('Finished with G-Matrix, last internal coordinate calculation...')
        final_internals = self.c_func.get_ints(self.coords)
        return gnm / np.sum(self.dw), final_internals
