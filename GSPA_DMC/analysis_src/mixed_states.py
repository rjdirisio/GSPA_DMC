from scipy.linalg import eigh as seigh
import numpy as np
import os


class MixedStates:
    def __init__(self,
                 res_dir,
                 overlap_mat,
                 ham_mat,
                 assignments,
                 energy_range,
                 dip_matels):
        self.res_dir = res_dir
        self.overlap_mat = overlap_mat
        self.ham_mat = ham_mat
        self.assignments = assignments
        self.mus = dip_matels
        self.energy_range = np.sort(energy_range)
        self._initialize()

    def _initialize(self):
        if not os.path.isdir(self.res_dir):
            print('No res_dir found. creating...')
            os.makedirs(self.res_dir)
        self.transitions = np.diagonal(self.ham_mat)
        self._sort_matrices()
        self._reduced_mat()

    def _sort_matrices(self):
        idx = np.argsort(self.transitions)
        self.srt_assignments = self.assignments[idx]
        self.srt_overlap = self.overlap_mat[idx, :][:, idx]
        self.srt_ham = self.ham_mat[idx, :][:, idx]
        self.srt_mus = self.mus[idx]
        self.srt_transitions = self.transitions[idx]

    def _reduced_mat(self):
        qual = (self.srt_transitions > self.energy_range[0]) * (self.srt_transitions < self.energy_range[1])
        self.chunk_inds = np.where(qual)[0]
        mx = self.chunk_inds[-1]
        mn = self.chunk_inds[0]
        self.chunk_ham = self.srt_ham[mn:mx + 1, mn:mx + 1]
        self.chunk_ov = self.srt_overlap[mn:mx + 1, mn:mx + 1]
        self.chunk_mus = self.srt_mus[mn:mx + 1]
        self.chunk_assn = self.srt_assignments[mn:mx + 1]


    def _diagonalize(self):
        evals, evecs = seigh(self.chunk_ham, self.chunk_ov)
        return evals, evecs

    def _transform_dipoles(self, evecs):
        trans_dips = evecs.dot(self.chunk_mus)
        return trans_dips

    def _write_contributions(self, new_freqs, new_intensities, evecs):
        contribs = open(f"{self.res_dir}/contribs.txt", "w")
        contribs.write('E         I\n')
        evecs_sorted = np.zeros((len(evecs), len(evecs)))
        for nevs in range(len(new_freqs)):
            # Sort by square of eigenvector (abs value)
            state_assign = self.chunk_assn[np.flip(np.argsort(np.square(evecs[:, nevs])))]
            evecs_sorted[:, nevs] = evecs[np.flip(np.argsort(np.square(evecs[:, nevs]))), nevs]
            contribs.write(f'{new_freqs[nevs]:.3f}, {new_intensities[nevs]}\n ')
            for each_state in range(5):  # top 5 states
                contribs.write(
                    f"{state_assign[each_state][0]} {state_assign[each_state][1]}, {evecs_sorted[each_state, nevs]:.3f}")
            contribs.write("\n")
        contribs.close()

    @property
    def get_matrices(self):
        return self.chunk_ov, self.chunk_ham

    def run(self):
        new_freqs, evecs = self._diagonalize()
        transformed_dipoles = self._transform_dipoles(evecs)
        # Replace old intensities with new
        self.srt_mus[self.chunk_inds] = transformed_dipoles
        srt_intensities = np.linalg.norm(self.srt_mus, axis=-1) ** 2
        # Write contributions of zero-order states to mixed states
        self._write_contributions(new_freqs, srt_intensities[self.chunk_inds], evecs)
        # Replace old freqs with new
        self.srt_transitions[self.chunk_inds] = new_freqs

        return self.srt_transitions, srt_intensities, self.chunk_inds
