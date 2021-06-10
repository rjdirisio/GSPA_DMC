# GSPA_DMC

## Installation

GSPA_DMC is available on GitHub, and the process of installation is cloning the repository and then running `pip install` inside the root directory of the cloned project.

This runs the `setup.py` file in the root directory and installs the package so that it is importable anywhere so long as that version of python/conda environment is loaded.

```
git clone https://github.com/rjdirisio/GSPA_DMC.git
cd GSPA_DMC
pip install .
```

## Requirements

This code requires:

1. NumPy
2. SciPy
3. [pyvibdmc](https://github.com/rjdirisio/pyvibdmc)
4. Matplotlib (Plotting spectra only)
5. A Python function that calculates 3n-6 internal coordinates

## Getting Started

### Learning about the structure of the package

This package is separated into four distinct but connected parts. The first part of the code produces the `q` coordinates, or pseudo normal modes, from the mass-weighted second moments matrix. The source code for this portion is in the `GSPA_DMC/GSPA_DMC/normal_mode_src/` directory. The second portion of this code is the calculation of the transition energies and intensities between the `n=0` and `n=1` or `n=2` states that one calculates.  This portion can also calculate the Hamiltonian and overlap matrix of all the states in the constructed basis to then use later on. All this is the actual GSPA approximation, and the source for this code is in `GSPA_DMC/GSPA_DMC/gspa_src/`. Next, there is a general utilities directory that helps the user perform symmetry operations efficiently and also hook the internal coordinates into the code, which is in `GSPA_DMC/GSPA_DMC/utils`.  Finally, the analysis part of the code will plot the transitions using stick spectra with Gaussian convolution (optional). This is also where the couplings bewteen states are incorporated, and the final spectra coan be plotted as well. The source for this portion of the code is in `GSPA_DMC/GSPA_DMC/analysis_src/`

# How to use the package

While there are many portions of this package, there are only a few classses/functions you should need to use this code. Everything else is "under the hood"!

## Example Run Scripts

### Defining Internal Coordinates and Hooking the Function into the Code

We will use H3O+ as a testing ground for this code. We will start by making a function that defines 3N-6 = 6 internal coordinates for hydronium:

```
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
    get vector pointing along OH bond
    vec_1 = np.divide((out_1 - cen), la.norm(out_1 - cen, axis=1)[:, np.newaxis])  # broadcasting silliness
    vec_2 = np.divide((out_2 - cen), la.norm(out_2 - cen, axis=1)[:, np.newaxis])
    vec_3 = np.divide((out_3 - cen), la.norm(out_3 - cen, axis=1)[:, np.newaxis])
    # vectors between the unit vectors 
    un_12 = vec_2 - vec_1
    un_23 = vec_3 - vec_2
    #Cross product, need arb. def. of which two are going to be the two that decide if the umbrella is btw 0-90 and 90-180
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
    # Define AnalyzeWfn object from pyvibdmc (using pyvibdmc is optional for defining internal coordinates)
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

To have the code recognize the function `h3o_internals` we use the `InternalCoordinateManager` in the `utils` subpackage.
