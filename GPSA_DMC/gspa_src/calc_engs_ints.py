class CalcEngsInts:
    def __init__(self,
                 nms,
                 potentials,
                 dipoles,
                 ham_overlap):
        self.nms = nms
        self.vs = potentials
        self.dips = dipoles
        self.ham = ham_overlap

    def calc_freqs(self):
        q = self.nms
        v = self.vs
        frequencies = []
        return frequencies

    def calc_ints(self):
        q = self.nms
        dip = self.dips
        intensities = []
        return intensities

    def calc_ham(self):
        print('here we gooo')
        overlap = []
        ham = []
        return overlap, ham

    def run(self):
        energies = self.calc_freqs()
        intensities = self.calc_ints()
        if self.ham:
            overlap, ham = self.calc_ham()
            return energies, intensities, overlap, ham
        else:
            return energies, intensities
