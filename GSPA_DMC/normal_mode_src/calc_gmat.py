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
        small_mem = False
        #Only use gnm if memory error happens
        gnm = np.zeros((self.num_vibs, self.num_vibs))
        if len(self.coords.shape) == 2:
            print("EXPANDING COORDS")
            self.coords = [self.coords, self.coords]
            self.dw = [self.dw, self.dw]
        try:
            print("TRYING TOT_DERV")
            tot_derv = np.zeros((len(self.dw), self.num_vibs,self.num_vibs))
        except MemoryError:
            print("SWITCHING TO LOW MEMORY")
            small_mem = True
        for atom in range(len(self.masses)):
            for coordinate in range(3):
                print(f'dx {atom * 3 + (coordinate + 1)}')
                print(f'atom: {atom}, coordinate {coordinate}')
                delx = np.zeros(self.coords.shape)
                delx[:, atom, coordinate] += self.dx  # perturbs the x,y,z coordinate of the atom of interest
                coord_plus = self.c_func.get_ints(self.coords + delx)
                coord_minus = self.c_func.get_ints(self.coords - delx)

                # For 2pi stuff going from 0 to 2pi
                problems = len(np.where(np.abs(coord_plus - coord_minus) > 1.0)[0])
                if problems > 0:
                    print("Large derivative in GMAT (most likely from 0-->2pi internals). Fixing...")
                else:
                    del problems
                coord_plus[np.abs(coord_plus - coord_minus) > 1.0] += \
                    (-1.0 * 2. * np.pi) * np.sign(coord_plus[np.abs(coord_plus - coord_minus) > 1.0])
                partialderv = (coord_plus - coord_minus) / (2.0 * self.dx)  # Finite Diff

                if not small_mem:
                    try:
                        mass_weighted_pd = partialderv[:, :, np.newaxis] * partialderv[:, np.newaxis, :] / self.masses[atom]
                        tot_derv += mass_weighted_pd
                        print('stuff')
                    except MemoryError:
                        small_mem=True

                if small_mem:
                    print('SMALL MEMORY CALCS INITIATED')
                    for i, pd in enumerate(partialderv):  # memory constraints. loop...
                        mwpd2 = (partialderv[i, :, np.newaxis] * partialderv[i, np.newaxis, :]) / self.masses[atom]
                        gnm += mwpd2 * self.dw[i]

        print('Finished with G-Matrix, last internal coordinate calculation...')
        final_internals = self.c_func.get_ints(self.coords)
        if small_mem:
            return gnm / np.sum(self.dw), final_internals
        else:
            gmat = np.average(tot_derv, axis=0, weights=self.dw)
            return gmat, final_internals