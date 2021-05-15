from pyvibdmc.analysis import *

def umbrella_angle(cds, center, outer_1, outer_2, outer_3):
    # Calculate the umbrella angle
    cen = cds[:, center]
    # Every walker's xyz coordinate for O
    out_1 = cds[:, outer_1]
    out_2 = cds[:, outer_2]
    out_3 = cds[:, outer_3]
    # test = H1-O
    vec_1 = np.divide((out_1 - cen), la.norm(out_1 - cen, axis=1)[:, np.newaxis])  # broadcasting silliness
    vec_2 = np.divide((out_2 - cen), la.norm(out_2 - cen, axis=1)[:, np.newaxis])
    vec_3 = np.divide((out_3 - cen), la.norm(out_3 - cen, axis=1)[:, np.newaxis])
    # vectors between the points along the OH bonds that are 1 unit vector away from the O
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
    analyzer = AnalyzeWfn(cds)
    roh1 = analyzer.bond_length(3, 0)
    roh2 = analyzer.bond_length(3, 1)
    roh3 = analyzer.bond_length(3, 2)
    hoh1 = analyzer.bond_angle(0, 3, 1)
    hoh2 = analyzer.bond_angle(1, 3, 2)
    hoh3 = analyzer.bond_angle(2, 3, 0)
    angle_1 = 2 * hoh1 - hoh2 - hoh3
    angle_2 = hoh2 - hoh3
    umbrella = umbrella_angle(cds, 3, 0, 1, 2)
    h3o = np.array([roh1, roh2, roh3, umbrella, angle_1, angle_2, ]).T
    return h3o
