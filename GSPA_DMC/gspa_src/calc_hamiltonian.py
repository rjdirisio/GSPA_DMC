class CalcHamOverlap:

    def __init__(self,
                 nms,
                 desc_weights,
                 potentials):
        self.qs = nms
        self.dw = desc_weights
        self.vs = potentials

    def calc_ham(self):
        print('here we gooo')
        overlap = []
        ham = []
        return overlap, ham
