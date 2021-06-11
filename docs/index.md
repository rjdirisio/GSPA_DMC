# GSPA_DMC

## Installation

GSPA_DMC is available on GitHub, and the process of installation is cloning the repository and then running `pip install` inside the root directory of the cloned project.

This runs the `setup.py` file in the root directory and installs the package so that it is importable anywhere so long as that version of python/conda environment is loaded.

```
git clone https://github.com/rjdirisio/GSPA_DMC.git
cd GSPA_DMC
pip install .
```

## Dependencies

This code requires:

1. NumPy
2. SciPy
3. [pyvibdmc](https://github.com/rjdirisio/pyvibdmc)
4. Matplotlib (Plotting spectra only)


## Getting Started

### Learning about the structure of the package

This package is separated into four distinct but connected parts. The first part of the code produces the `q` 
coordinates, or pseudo normal modes, from the mass-weighted second moments matrix. 
The source code for this portion is in the `GSPA_DMC/GSPA_DMC/normal_mode_src/` directory. 
The second portion of this code is the calculation of the transition energies and intensities 
between the `n=0` and `n=1` or `n=2` states that one calculates.  This portion can also calculate 
the Hamiltonian and overlap matrix of all the states in the constructed basis to then use later on. 
All this is the actual GSPA approximation, and the source for this code is in `GSPA_DMC/GSPA_DMC/gspa_src/`. 
Next, there is a general utilities directory that helps the user perform symmetry operations efficiently and also 
hook the internal coordinates into the code, which is in `GSPA_DMC/GSPA_DMC/utils`.  Finally, the analysis part of 
the code will plot the transitions using stick spectra with Gaussian convolution (optional). This is also where the 
couplings bewteen states are incorporated, and the final spectra coan be plotted as well. The source for this portion 
of the code is in `GSPA_DMC/GSPA_DMC/analysis_src/`

### What is needed to use this package

For the normal mode portion of the code, we require that you pass in a DMC wave function and descendant weights associated with the DMC wave 
function. This takes the form of a `(num_walkers x num_atoms x 3)` NumPy array for the wave function, and a `(num_walkers)` array
for the descendant weights. This is the same input/output structure of what goes in and out of `PyVibDMC`; all 
coordinates must be in atomic units. **NOTE** if you are using `PyVibDMC` to get wave functions, it will return them in Angstroms
if you get them through the `get_wfns` function. You will need to convert the coordinates to Bohr in order to pass into this code.

The internal coordinate function that hooks into the code should take in just the Cartesian coordinates (in Bohr) and output the 
internal coordinates in ()

# How to use the package

While there are many portions of this package, there are only a few classses/functions you should need to use this code. Everything else is "under the hood"!

## Example Run Scripts

### Defining Internal Coordinates and Hooking the Function into the Code

We will use H3O+ as a testing ground for this code. We will start by making a function that defines 3N-6 = 6 internal coordinates for hydronium:

```python
# h3o_internals.py
import numpy as np
from pyvibdmc.analysis import *

def umbrella_angle(cds, center, outer_1, outer_2, outer_3):
    # Calculate the umbrella angle
    cen = cds[:, center]
    # Every walker's xyz coordinate for O
    out_1 = cds[:, outer_1]
    out_2 = cds[:, outer_2]
    out_3 = cds[:, outer_3]
    # get vector pointing along OH bond
    vec_1 = np.divide((out_1 - cen), la.norm(out_1 - cen, axis=1)[:, np.newaxis])  # broadcasting silliness
    vec_2 = np.divide((out_2 - cen), la.norm(out_2 - cen, axis=1)[:, np.newaxis])
    vec_3 = np.divide((out_3 - cen), la.norm(out_3 - cen, axis=1)[:, np.newaxis])
    # vectors between the unit vectors 
    un_12 = vec_2 - vec_1
    un_23 = vec_3 - vec_2
    #Cross product, need arb. def. of which two are going to be 
    # the two that decide if the umbrella is btw 0-90 and 90-180
    line = np.cross(un_12, un_23, axis=1)
    # add normalized vector to O
    spot = line / la.norm(line, axis=1)[:, np.newaxis]
    # Calculate angle between dummy, center, and one of the three outers
    fin_1 = spot
    fin_2 = vec_1
    dotted = (fin_1 * fin_2).sum(axis=1)
    norm_mult = la.norm(fin_1, axis=1) * la.norm(fin_2, axis=1)
    this = np.arccos(dotted / norm_mult)
    return this

def h3o_internals(cds):
    """Returns all coordinates in bohr / radians"""
    # Define AnalyzeWfn object from pyvibdmc 
    # (using pyvibdmc is optional for defining internal coordinates)
    analyzer = AnalyzeWfn(cds)
    # Define 3 OH bond lengths
    roh1 = analyzer.bond_length(3, 0)
    roh2 = analyzer.bond_length(3, 1)
    roh3 = analyzer.bond_length(3, 2)
    # Define 3 HOH bond angles
    hoh1 = analyzer.bond_angle(0, 3, 1)
    hoh2 = analyzer.bond_angle(1, 3, 2)
    hoh3 = analyzer.bond_angle(2, 3, 0)
    # Transform them to not be redundant when H3O is planar
    angle_1 = 2 * hoh1 - hoh2 - hoh3
    angle_2 = hoh2 - hoh3
    # Calculate the umbrella angle for H3O
    umbrella = umbrella_angle(cds, 3, 0, 1, 2)
    h3o = np.array([roh1, roh2, roh3, umbrella, angle_1, angle_2, ]).T
    return h3o

```

The code should return the internal coordinates in shape `(num_walkers x three_n_minus_6)`. See above for a good example.
To have the `GSPA_DMC` code recognize the function `h3o_internals` we use the `InternalCoordinateManager` in the `utils` subpackage:

```python
import GSPA_DMC as gspa
h3o_internals = gspa.InternalCoordinateManager(
    int_function='h3o_internals',
    int_directory='', # Do this if it's in your current directory. Otherwise, specify an absolute or relative path.
    python_file='h3o_internals.py',
    int_names=['ROH_1', 'ROH_2', 'ROH_3', 'Umb','2Th_1-Th2-Th3', 'Th2-Th3'])
```

Now we have a `InternalCoordinateManager` object that we will pass to `NormalModes` for running.

```python
import GSPA_DMC as gspa
nm = gspa.NormalModes(res_dir='test_h3o',
                    atoms=['H', 'H', 'H', 'O'],
                    walkers=cds,
                    descendant_weights=dws,
                    ic_manager=h3o_internals)
```

### Calculating the G-Matrix

Now that we have everything set up, we will now calculate the q coordinates in two steps. The first is a time consuming
part, calculating the G-Matrix through numerical differentiation.  Once the G-Matrix is constructed, we can
use it and the second moments matrix to get the q coordinates.

```python
import GSPA_DMC as gspa
nm = gspa.NormalModes(res_dir='test_h3o',
                    atoms=['H', 'H', 'H', 'O'],
                    walkers=cds,
                    descendant_weights=dws,
                    ic_manager=h3o_internals)
gmat, my_internals = nm.calc_gmat()
#You could run the next step of this code immediately, or save the internals and do it later. See "Calculating the q Coordinates"
np.save('test_h3o/internals.npy", my_internals)
```

 Here, we also specify a results directory called `res_dir`.  
This can be an absolute or relative path to the directory of interest. If the directory doesn't exist yet, it will be created.
 Along with the internal coordinate manager, we also pass the coordinates, the descendant weights, and the atom ordering 
 as a list of one or two letter strings, which is used to calculate the mass (Supports "D" for deuterium). 
 You can also pass in custom `masses`, or if you are struggling with the bohr/angstrom conversion you can pass in the
 `atomic_units=False` argument to `NormalModes` as well, which converts the input to bohr from angstroms.

#### Quickly Projecting $\Psi^2$ onto your Internal Coordinates
`calc_gmat()` also returns the internal coordinates. At this point, you can project these `Psi^2` onto one of these 
internal coordinates to see how they look:

```python
from pyvibdmc.analysis import AnalyzeWfn, Plotter
internals = h3o_internals.get_ints(cds)
int_names = h3o_internals.get_int_names()
for i_num, name in enumerate(int_names):
    histie = AnalyzeWfn.projection_1d(internals[:,i_num],desc_weights=dws)
    Plotter.plt_hist1d(histie,xlabel=name,save_name=name)
```

### Calculating the q Coordinates

Once the G-Matrix is calculated, you can then use it and the internal coordinates to calculate the q coordinates:

```python
gmat = np.load("test_h3o/gmat.npy") # gmat was saved according to res_dir
# Internal coords were NOT saved since it could be a large file.
# Instead, you can manually save them or rerun h3o_internals.get_ints(cds)
my_internals = np.load("test_h3o/internals.npy")  
q_cds = nm.calc_normal_modes(gmat=gmat,
                        internal_coordinates=my_internals,
                        save_nms=True)
```

If `save_nms` is set to `False`, the q coordinates will not be saved (not recommended).

From this run, you will get three files: `res_dir/assignments.txt`, `res_dir/nms.npy`, and `res_dir/trans_mat.npy`,
which correspond to a list of the the linear combinations of mass-weighted internal coordinates that compose the q coordinates,
the q coordinates themselves, and then transformation matrix that takes you from modified internals (r-<r>) to q coordinates.

The q coordinate file is the only essential file for the rest of the code, but the assignments are there to 
see if the q coordinates are coming out clean.

#### Brief but Important Aside on Symmetry in the DMC Wave Function

The assignments that correspond to the q coordinates are sensitive to subtleties in the DMC wave function. For example,
you will not get equal contribution from each of the OH bond lengths in H3O+ for the symmetric stretch if you just use 
a regular, unmodified set of DMC wave functions. To compensate for this, it is good practice to _symmetrize_ the wave function.
This typically entails making copies of the walker of interest when you swap equivalent atoms. In H3O+, you want to make 
sure that it doesn't mater which hydrogen is called "1" or "2" or "3". The projection of $\Psi^2$ onto one of the 
OH bond lengths should look _exactly_ like the other two. The way to do this is swap what is called "1" with "2", 
"2" with "3", and "1" with "3" and so on. GSPA_DMC has a built in `Symmetrize` object that can help with this process:

```python
# new_cds is 2x the size of cds
# Copy, then swap atom 0 with atom 1
new_cds, new_dws = symm.swap_two_atoms(cds, dws, atm_1=0,atm_2=1)
# Swap atom 1 with atom 2
new_cds2, new_dws2 = symm.swap_two_atoms(cds, dws, atm_1=2,atm_2=3)
# Swap atom group [3,4,5] with [6,7,8]
new_cds3, new_dws3 = symm.swap_group(cds, dws, atm_list_1=[3,4,5],atm_list_2=[6,7,8])
# If your atoms are rotated so three of them are in the xy plane, you can reflect about the plane
new_cds4, new_dws4 = symm.reflect_about_xyp(cds, dws)
...
# Combine them all, can now use final_cds and final_dws in the 
final_cds = np.concatenate((cds, new_cds,new_cds2,new_cds3,new_cds4...))
final_dws = np.concatenate((cds, new_dws,new_dws2,new_dws3,new_cds4...))
``` 

### Calculating GSPA Frequencies and Intensities with q Coordinates

Now that you have the q coordinates, you can use them to calculate energies and intensities. For this part of the code,
you need:

1. The q coordinates 
2. The potential energy of each of the walkers `num_walkers`
3. The dipole moment of each walker, `num_walkers x 3`

If you just want to calculate energies, you can pass in a dipole array that is simply zeros or ones (`np.zeros` or `np.ones`) 
and the length of your potential energy array.

```python
import GSPA_DMC as gspa
q_coords = np.load("test_h3o/nms.npy")
my_gspa = gspa.GSPA(res_dir='test_h3o',
               normal_modes=q_coords,
               desc_weights=dws,
               potential_energies=vs,
               dipoles=dips,
               ham_overlap=False)
my_gspa.run()
```

This part of the code generates 5 files if `ham_overlap` is `False`, and 6 files if `ham_overlap` is `True`:

* `res_dir/energies.npz` which has all the transition fundamental, overtone, and combination energies.
   * Keys: `funds`, `overs`, `combos`, with the modes corresponding to the ordering in `assignments.txt`
* `res_dir/intensities.npz` which has all the transition fundamental, overtone, and combination intensities.
   * Keys: `funds`, `overs`, `combos`, with the modes corresponding to the ordering in `assignments.txt`
* `res_dir/assign_order.npz` which gives a numbered list corresponding to which transitions are for which mode.
* `res_dir/all_transitions.txt` which is a plaintext summary of the above 3 `npz` files. 
* `res_dir/red_ham/ov_ham.npz` which has the Hamiltonian and overlap matrix, and dipole matrix elements if one wants to calculate mixed states.
   * Ordering of Hamiltonian: `funds`, `overs`, then `combos`, with the modes corresponding to the ordering in `assignments.txt`
   * If `ham_overlap=False`, then `red_ham` should be empty

**Important Note: The states in `assign_order` and `all_transitions` use a specific notation. There are always 2 numbers
for a state assignment, The first and second number correspond to excitation in that mode. If the number is repeated, like
`4, 4` that means there is two quanta of excitation in mode `4`. If there is a `999` in the second mode, then that means
the mode is only excited in one mode, which is the first number.**

To load the binary `.npz` files:

```python
import numpy as np
a = np.load("res_dir/energies.npz")
# Then use one of the keys to get the numpy array you want
fundamental_transition_energies = a['funds']
```

### Plotting the Stick Spectra

Yay! You made it this far. You now have results you'd like to plot. You can select which subset of frequencies and
intensities to load, but it is recommended to simply combine all transitions to plot:

```python
import GSPA_DMC as gspa
# All the keys for the .npz files
keyz = ['funds', 'overs', 'combos']
np_e = np.load("test_h3o/energies.npz")
np_i = np.load("test_h3o/intensities.npz")
# Combine all energies, intensities
energies = []
intensities = []
for key in keyz:
    energies.append(np_e[key])
    intensities.append(np_i[key])
energies = np.concatenate(energies)
intensities = np.concatenate(intensities)

# Declare PlotSpectrum object
plottie = gspa.PlotSpectrum(energies=energies,
                       intensities=intensities,
                       plt_energy_range=[0, 6000],
                       gauss_width=50,
                       )
# Get stick xy data as well as gaussian convolution xy data
xy_data, gauss_data = plottie.get_xy_data()
# Plot it and save it
plottie.plot_xy_data(stick_xy=xy_data,
                     savefig_name='example.png',
                     gauss_xy=gauss_data,
                     stick_color='b',
                     gauss_color='k',
                     pdf=False)
```

The spectra can be convoluted with Gaussian functions, which is what `gauss_width` controls.  If you just want sticks,
you can simply not declare a `gauss_width` otherwise, this number is used as the `sigma` (standard deviation)
parameter of a Gaussian distribution. 

`savefig_name` can take an absolute path or relative path. You can also save the spectrum as a pdf!

### Mixed States and Hamiltonian Matrices

We can use the Hamiltonian, overlap matrix, and dipole matrix elements we calculated earlier to generate mixed
states with corresponding energies and intensities that are linear combinations of our original basis.

In order to do this, we use the `MixedStates` object

```python
    assignments = np.load('test_h3o/assign_order.npy')
    red_stuff = np.load("test_h3o/red_ham/ov_ham.npz")
    overlap_mat = red_stuff['ov']
    ham_mat = red_stuff['ham']
    dipole_matels = red_stuff['mus']
    my_mix = gspa.MixedStates(res_dir='sample',
                         overlap_mat=overlap_mat,
                         ham_mat=ham_mat,
                         assignments=assignments,
                         energy_range=[3000, 4000],
                         dip_matels=dipole_matels)
    new_freqs, new_intensities, mixed_indcs = my_mix.run()
```

The `assignments` are used to create a `contribs.txt` file which breaks down the mixed states into their component
parts. The file structure of `contribs.txt` goes:


```
Energy, Intensity
 State, Cn;  State, Cn; ...
Energy, Intensity
 State, Cn;  State, Cn; ...
```

Where State corresponds to the numbering in `assignments.txt` and `Cn` is the corresponding coefficient 
of the eigenvectors of the generalized eigenvalue problem used to solve for the new mixed states.

Finally, we can plot the mixed states as a different color on the same plot as the unmixed states.

```python
    plottie = gspa.PlotSpectrum(energies=new_freqs,
                 intensities=new_intensities,
                 plt_energy_range=[0,6000],
                 gauss_width=50,
                 mixed_indcs=mixed_indcs,
                 norm_contrib_fl='sample/contribs.txt',
                 )
    xy_data, gauss_data = plottie.get_xy_data()
    plottie.plot_xy_data(stick_xy=xy_data,
                         savefig_name='example.png',
                         gauss_xy=gauss_data,
                         stick_color='b',
                         mixed_stick_color='r',
                         gauss_color='k',
                         pdf=False)

```

There will be an updated `contribs_normed.txt` file that corresponds to the normalized relative intensities as 
reflected in the spectrum picture.