import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


class PlotSpectrum:
    def __init__(self,
                 energies,
                 intensities,
                 plt_energy_range,
                 gauss_width=None,
                 mixed_indcs=None,
                 norm_contrib_fl=None):
        """
        Returns xy data of sticks and of gaussian convolution spectra, which you can then plot with
        :param energies: List of energies of states to be plotted. (optional) Includes states involved in mixing.
        :param intensities: List of intensities of states to be plotted. (optional) Includes states involved in mixing.
        :param plt_energy_range: the window that will be displayed when you plot. the gaussian convolution cares
        about this too
        :param gauss_width: If convoluting stick spectra with gaussians, this is sigma.
        :param mixed_indcs: if there is state mixing, include indices that correspond to states
        :param norm_contrib_fl: path to file to obtain contributions from that, norm intensity in there.
        """

        self.energies = energies
        self.intensities = intensities
        self.e_range = np.sort(plt_energy_range)
        self.gauss_width = gauss_width
        self.mix_idx = mixed_indcs
        self.norm_contrib_fl = norm_contrib_fl
        self._initialize()

    def _initialize(self):
        if self.gauss_width is None:
            self.gauss_time = False
        else:
            self.gauss_time = True
        if self.norm_contrib_fl is not None:
            if '/' in self.norm_contrib_fl:
                self.contrib_pth = self.norm_contrib_fl.rpartition('/')[0]
            else:
                self.contrib_pth = '.'
            self.contrib_xy, self.other_stuff = self._extract_contribs()
        # Get most intense feature in energy range if not given norm_to_ind
        self._norm_intensity()


    def _extract_contribs(self):
        with open(f'{self.norm_contrib_fl}','r') as fll:
            linez = fll.readlines()
            line_num = 0
            contrib_xy = []
            other_stuff = []
            for line in linez:
                if line_num == 0:
                    pass
                elif line_num % 2 == 0 and line_num != 0:
                    other_stuff.append(line)
                else:
                    line = line.split(', ')
                    contrib_xy.append([float(line[0]),float(line[1])])
                line_num+=1
        return np.array(contrib_xy), other_stuff

    def _truncate_data(self):
        """Truncate data to energy range in order to normalize intensity in range to 1"""
        trunc_idx = np.argsort(self.energies)
        trunc_intensities = self.intensities[trunc_idx]
        norm_by = np.amax(trunc_intensities)
        return norm_by

    def _rewrite_contrib(self):
        contribs = open(f"{self.contrib_pth}/contribs_normed.txt", "w")
        contribs.write('E         I\n')
        for num, f_i in enumerate(self.contrib_xy):
            contribs.write(f'{f_i[0]:.3f}, {f_i[1]:.5f}\n ')
            contribs.write(self.other_stuff[num])
        contribs.close()

    def _norm_intensity(self):
        norm_by = self._truncate_data()
        self.intensities = self.intensities / norm_by
        if self.norm_contrib_fl is not None:
            self.contrib_xy[:, 1] = self.contrib_xy[:, 1] / norm_by
            self._rewrite_contrib()

    def _convolute_spectrum(self):
        disc_e = np.linspace(self.e_range[0], self.e_range[1], 5000)
        g = np.zeros(len(disc_e))
        for i in range(len(self.intensities)):
            g += self.intensities[i] * np.exp(
                -np.square(disc_e - self.energies[i]) / (2.0 * np.square(self.gauss_width)))
        g /= np.amax(g)
        return np.array([disc_e, g]).T

    def get_xy_data(self):
        stick_xy = np.column_stack((self.energies, self.intensities))
        if self.gauss_time:
            gauss_xy = self._convolute_spectrum()
            return stick_xy, gauss_xy
        else:
            return stick_xy

    def plot_xy_data(self,
                     stick_xy,
                     savefig_name='spectrum',
                     gauss_xy=None,
                     stick_color='b',
                     mixed_stick_color='r',
                     gauss_color='k',
                     xlabel='Energy (cm$^{-1}$)',
                     ylabel='Rel. Intensity',
                     pdf=False
                     ):
        params = {'text.usetex': False,
                  'mathtext.fontset': 'dejavusans',
                  'font.size': 14}
        plt.rcParams.update(params)
        # Plot sticks
        plt.stem(stick_xy[:, 0], stick_xy[:, 1], f'{stick_color}', markerfmt=" ", basefmt=" ")
        if self.mix_idx is not None:
            plt.stem(stick_xy[self.mix_idx, 0], stick_xy[self.mix_idx, 1], f'{mixed_stick_color}', markerfmt=" ",
                     basefmt=" ")
        if gauss_xy is not None:
            plt.plot(gauss_xy[:, 0], gauss_xy[:, 1], f'{gauss_color}', linewidth=2)
        # Labels
        plt.xlabel(f'{xlabel}')
        plt.ylabel(f'{ylabel}')
        # Save figure
        if '.' in savefig_name:
            splt = savefig_name.split('.')
            savefig_name = splt[0]
        plt.xlim([self.e_range[0],self.e_range[1]])
        if pdf:
            plt.savefig(f'{savefig_name}.pdf', bbox_inches='tight')
        else:
            plt.savefig(f'{savefig_name}.png', dpi=300, bbox_inches='tight')
