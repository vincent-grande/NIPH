import numpy as np
import pandas as pd
import gudhi as gd
from pylab import *
from matplotlib import pyplot as plt
import scipy
import scipy.sparse
import sklearn
import sklearn.metrics
import scipy.spatial
import scipy.stats
import seaborn as sns
from scipy import stats
import ot
import os
import multiprocess
from scipy.integrate import quad
from scipy.stats import norm
from scipy import optimize
import numpy as np
import pandas as pd
import gudhi as gd
#from pylab import *
from matplotlib import pyplot as plt
import scipy
import scipy.sparse
import sklearn
import sklearn.metrics
import scipy.spatial
import scipy.stats
import seaborn as sns
from scipy import stats
import ot
import os
import json
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from scipy import optimize
import plotly.express as px

from multiprocess import *
from NIPHSampleGeneration import *

pdf_array = np.array([scipy.stats.norm.pdf(x, 0, 1) for x in np.linspace(-2, 2, 50)])


num_steps = 50
pdf_array = np.array(
    [scipy.stats.norm.pdf(x, 0, 1) for x in np.linspace(-2, 2, num_steps)]
)


def expected_shift_list(
    angle, probing_angle, probing_scaling, actual_scaling, variance=0, dim=1
):
    angle_diff = angle - probing_angle
    if variance == 0:
        return expected_shift_zero_list(
            angle_diff, probing_scaling, actual_scaling, dim=dim
        )
    else:
        # return scipy.integrate.quad(lambda x: expected_shift_zero_list(angle_diff+x,probing_scaling, actual_scaling)* scipy.stats.norm.pdf(x,0,variance),-2*variance,2*variance, epsabs=1.49e-5,epsrel=1.49e-6,limit=30)[0]
        return (
            np.sum(
                np.array(
                    [
                        expected_shift_zero_list(
                            angle_diff + x * variance,
                            probing_scaling,
                            actual_scaling,
                            dim=dim,
                        )
                        * pdf_array[i]
                        for i, x in enumerate(np.linspace(-2, 2, 50))
                    ]
                )
            )
            / num_steps
            * 4
        )


def expected_shift_zero_list(angle_diff, probing_scaling, actual_scaling, dim=1):
    sin_diff_sq = np.sin(angle_diff) ** 2
    cos_diff_sq = np.cos(angle_diff) ** 2
    if dim == 1:
        normal_shift = probing_scaling / np.sqrt(
            1 + (probing_scaling**2 - 1) * cos_diff_sq
        )
        baseline_shift = (
            probing_scaling
            * actual_scaling
            / np.sqrt(1 + (probing_scaling**2 - 1) * sin_diff_sq)
        )
        print("dim 1!")
    elif dim == 0:
        normal_shift = np.sqrt(probing_scaling**2 * cos_diff_sq + sin_diff_sq)
        baseline_shift = actual_scaling
    return np.minimum(normal_shift, baseline_shift)

def city_extracter(city_name, longit):
    with open("secs_residential_"+city_name, 'r') as f:
        sections_residential_cur = json.load(f)
    sections_residential_cur = np.array(sections_residential_cur)
    sections_residential_cur[:,:,1]=sections_residential_cur[:,:,1]*np.cos(longit*np.pi/180)
    length_cur = 0
    section_length_cur = []
    for section in sections_residential_cur:
        length_cur += np.sqrt((section[0][0] - section[1][0])**2 + (section[0][1] - section[1][1])**2)
        section_length_cur.append(length_cur)
    print(length_cur)
    cars_cur = sample_cars(length_cur,sections_residential_cur,section_length_cur, 15000)
    maxs_cur = get_maxs(cars_cur)
    return maxs_cur, np.var(maxs_cur)

def total_angle_costs_list(
    angle, maxima, scaling, parameters, exponent, variance, dim=1
):
    expected_shifts = np.array(
        [
            expected_shift_list(
                angle,
                parameter[0],
                probing_scaling=parameter[1],
                actual_scaling=scaling,
                variance=variance,
                dim=dim,
            )
            for parameter in parameters
        ]
    ).flatten()
    costs = np.abs(expected_shifts - maxima) ** exponent
    return np.sum(costs)

def extract_PCA_direction(points):
    pca = sklearn.decomposition.PCA(n_components=2)
    pca.fit(points)
    return pca.components_[0], np.arctan(pca.components_[0][0]/pca.components_[0][1])*180/np.pi

def city_extracter_pca(city_name, longit):
    with open("secs_residential_"+city_name, 'r') as f:
        sections_residential_cur = json.load(f)
    sections_residential_cur = np.array(sections_residential_cur)
    sections_residential_cur[:,:,1]=sections_residential_cur[:,:,1]*np.cos(longit*np.pi/180)
    length_cur = 0
    section_length_cur = []
    for section in sections_residential_cur:
        length_cur += np.sqrt((section[0][0] - section[1][0])**2 + (section[0][1] - section[1][1])**2)
        section_length_cur.append(length_cur)
    print(length_cur)
    cars_cur = sample_cars(length_cur,sections_residential_cur,section_length_cur, 15000)
    PCA = sklearn.decomposition.PCA().fit(cars_cur)
    explained_variance = PCA.explained_variance_
    print(PCA.explained_variance_)
    return explained_variance[0], explained_variance[0]/explained_variance[1]

def get_maxs(cars, diagram = False, name = "density_diagram"):
    BCs, BC_list, parameters = run_zero_dim_hom(cars)
    BC_weights = []
    for Y in BC_list:
        #print("hi")
        X = BCs
        Z,shift_dest = Compute_death_shifts_zero_dim(X,Y)
        weights = shift_dest*relative_positions(shift_dest)**2
        data = np.concatenate((Z.reshape(BCs[:,1:].shape),weights.reshape(BCs[:,1:].shape)),axis=1)
        BC_weights.append(data.transpose())
    #BC_df = pd.DataFrame(
    #        data, columns=["death", "weight"])
    #sns.kdeplot(BC_df,x="death",weights="weight")
    if diagram:
        density_diagram_zero(BC_weights, save=True, name=name)
    return find_max_list(BC_weights, parameters, true_max=True, threshold=False, weights=False)

def best_angle_fit_list(maxima, parameters, exponent=1, dim=1):
    args_angle = np.linspace(0, np.pi, 90)
    args_variance = np.linspace(0, 1.0, 11)
    args_scaling = np.linspace(1.4, 2.2, 10)
    values = np.zeros((len(args_variance), len(args_angle), len(args_scaling)))
    for i, variance in enumerate(args_variance):
        print(i)
        for j, angle in enumerate(args_angle):
            for k, scale in enumerate(args_scaling):
                values[i, j, k] = total_angle_costs_list(
                    angle,
                    maxima,
                    scaling=scale,
                    exponent=exponent,
                    variance=variance,
                    parameters=parameters,
                    dim=dim,
                )
    min_indices = np.unravel_index(np.argmin(values), values.shape)
    return (
        args_angle[min_indices[1]],
        args_variance[min_indices[0]],
        args_scaling[min_indices[2]],
    )


def best_angle_fit_optimise(maxima, parameters, exponent=4, dim=1):
    def total_costs_now(x):
        return total_angle_costs_list(
            x[0], maxima, x[2], parameters, exponent, x[1], dim=dim
        )

    bounds = [(0, np.pi), (0, 2), (1, 4)]
    optima = optimize.dual_annealing(total_costs_now, bounds).x
    return optima[0], optima[2], optima[1]

def relative_positions(arr):
    sorted_arr = sorted(range(len(arr)), key=lambda x: arr[x])
    return np.array(sorted_arr)

def zero_run_on_points(points):
    parameters = np.array([[i / 10 * np.pi, 1.4] for i in range(10)])
    for scale in np.linspace(1.8, 5.0, 6):
        new_parameters = np.array([[i / 10 * np.pi, scale] for i in range(10)])
        parameters = np.concatenate((parameters, new_parameters), axis=0)
    BCs, BC_list = Compute_Scaled_Deaths_List(
        points, parameters=parameters, verbose=False, dim=0
    )
    BC_weights = []
    for Y in BC_list:
        X = BCs
        Z,shift_dest = Compute_death_shifts_zero_dim(X,Y)
        weights = shift_dest*relative_positions(shift_dest)**2
        data = np.concatenate((Z.reshape(BCs[:,1:].shape),weights.reshape(BCs[:,1:].shape)),axis=1)
        BC_weights.append(data.transpose())
    max_list = find_max_list(BC_weights, parameters, true_max=True, threshold=False)
    return best_angle_fit_optimise(
            max_list, parameters=parameters, exponent=4, dim =0
        )

def run_best_angle_fit(phi):
    # print(phi)
    prob_rad = 2
    prob_angle = 0.1
    points = make_random_angle_spheres(
        100,
        prob_angle,
        prob_rad=prob_rad,
        seed=int(phi * 1000),
        scaling_center=3000,
        phi=phi,
        rectangles=rectangles,
    )
    BCs, BCxs, BCys, BCxys, BC_anti_xys = Compute_Scaled_Deaths(
        points, plain=True, verbose=False
    )
    maxima = Compute_Max_try_2(BCs, BCxs, BCys, BCxys, BC_anti_xys, verbose=False)
    return (phi, best_angle_fit(maxima, exponent=4, account_variance=True))


def run_best_angle_fit_list(params):
    # print(params)
    values = params[0]
    percentage = params[1]
    prob_rad = 2
    phi = values[0]
    scaling = values[1]
    prob_angle = values[2]
    rectangles = False
    percentage_int = int(50 * percentage)
    progressstring = (
        "["
        + percentage_int * "="
        + (50 - percentage_int) * " "
        + "]"
        + " "
        + "%.2f" % (percentage * 100)
        + "%"
    )
    print(progressstring)
    points = make_random_angle_spheres(
        200,
        prob_angle,
        prob_rad=prob_rad,
        seed=int((scaling + phi) * 1000),
        scaling_center=3000,
        phi=phi,
        rectangles=rectangles,
        scaling=scaling,
    )
    parameters = np.array([[i / 10 * np.pi, 1.4] for i in range(10)])
    for scale in np.linspace(1.8, 3.0, 5):
        new_parameters = np.array([[i / 10 * np.pi, scale] for i in range(10)])
        parameters = np.concatenate((parameters, new_parameters), axis=0)

    BCs, BC_list = Compute_Scaled_Deaths_List(
        points, parameters=parameters, verbose=False
    )
    # print(BCs.shape)
    # for BCss in BC_list:
    #    print(BCss.shape)
    maxima = Compute_Maxs(BCs, BC_list, parameters, verbose=False)
    return (
        phi,
        best_angle_fit_optimise(maxima, parameters=parameters, exponent=4),
        values,
    )


def run_get_stds(params):
    # print(params)
    values = params[0]
    percentage = params[1]
    percentage_int = int(50 * percentage)
    progressstring = (
        "["
        + percentage_int * "="
        + (50 - percentage_int) * " "
        + "]"
        + " "
        + "%.2f" % (percentage * 100)
        + "%"
    )
    print(progressstring)
    prob_rad = values[3]
    phi = values[0]
    scaling = values[1]
    prob_angle = values[2]
    rectangles = False
    parameters = np.array([[i / 6 * np.pi, 1.2] for i in range(6)])
    for scale in np.linspace(1.4, 3.4, 5):
        new_parameters = np.array([[i / 6 * np.pi, scale] for i in range(6)])
        parameters = np.concatenate((parameters, new_parameters), axis=0)
        points = make_random_angle_spheres(
            200,
            prob_angle,
            prob_rad=prob_rad,
            seed=int((scaling + phi) * 1000),
            scaling_center=3000,
            phi=phi,
            rectangles=rectangles,
            scaling=scaling,
        )
    BCs, BC_list = Compute_Scaled_Deaths_List(
        points, parameters=parameters, verbose=False
    )
    # print(BCs.shape)
    # for BCss in BC_list:
    #    print(BCss.shape)
    std = np.average(Compute_stds(BCs, BC_list, parameters))
    return (std, values)


def expected_shift(angle, probing_angle, variance=0):
    angle_diff = angle - probing_angle
    if variance == 0:
        return 2 / (np.sqrt(1 + 3 * np.cos(angle_diff) ** 2))
    else:
        return scipy.integrate.quad(
            lambda x: expected_shift_zero(angle_diff + x)
            * scipy.stats.norm.pdf(x, 0, variance),
            -2 * variance,
            2 * variance,
        )[0]
        # return np.sum(np.array([expected_shift_zero(angle_diff+x*variance)* pdf_array[i] for i,x in enumerate(np.linspace(-2,2,50))]))


def expected_shift_zero(angle_diff):
    return 2 / (np.sqrt(1 + 3 * np.cos(angle_diff) ** 2))


def total_angle_costs(angle, maxima, exponent=1, variance=0):
    probing_angles = [i * np.pi / 4 for i in range(4)]
    costs = 0
    for i in range(4):
        costs = (
            costs
            + np.abs(expected_shift(angle, probing_angles[i], variance) - maxima[i])
            ** exponent
        )
    return costs


def best_angle_fit(maxima, exponent=1, account_variance=False):
    if account_variance:
        args_angle = np.linspace(0, np.pi, 90)
        args_variance = np.linspace(0, 1.0, 22)
        values = np.array(
            [
                [
                    total_angle_costs(argument, maxima, exponent, variance)
                    for argument in args_angle
                ]
                for variance in args_variance
            ]
        )
        min_indices = np.unravel_index(np.argmin(values), values.shape)
        return args_angle[min_indices[1]], args_variance[min_indices[0]]
    else:
        arguments = np.linspace(0, np.pi, 1000)
        values = np.array(
            [total_angle_costs(argument, maxima, exponent) for argument in arguments]
        )
    return arguments[np.argmin(values)]


def Compute_Weights(BCs):
    weights = [(barcode[1] - barcode[0]) / np.sqrt(2) for barcode in BCs]
    return weights


def Compute_Weights_mult(BCs, dim=1):
    if dim == 1:
        weights = np.array([(barcode[1] / barcode[0]) for barcode in BCs])
        if 0 in BCs[:, 0]:
            print("Warning: 0 in barcode")
    else:
        weights = np.array([barcode[1] for barcode in BCs])
    return weights


def Compute_mult_shifts(BCs, BCxs, dim=1):
    BCnorm = BCs
    BCnorm[:, 1] = BCnorm[:, 1] / 10
    BCxsnorm = BCxs
    BCxsnorm[:, 1] = BCxsnorm[:, 1] / 10
    Mx = ot.dist(BCnorm, BCxsnorm)
    weightBCs = Compute_Weights(BCs)
    weightBCxs = Compute_Weights(BCxs)
    Gsx = ot.unbalanced.mm_unbalanced(
        weightBCs, weightBCxs, Mx, 1, div="l2", numItermax=10000
    )
    mult_Mx = np.zeros((len(BCs), len(BCxs)))
    for i in range(len(BCs)):
        for j in range(len(BCxs)):
            mult_Mx[i, j] = BCxs[j][1] / BCs[i][1]
    # shiftsX = np.sum(np.multiply(mult_Mx, Gsx), axis = 1)/np.sum(Gsx, axis = 1)
    shiftsX = []
    for i in range(len(BCs)):
        shiftsX.append(mult_Mx[i, np.argmax(Gsx[i, :])])
    return np.array(shiftsX)


def Compute_mult_shifts_zero_dim(BCs, BCxs, dim=0):
    BCnorm = BCs
    BCnorm[:, 1] = BCnorm[:, 1] / 10
    BCxsnorm = BCxs
    BCxsnorm[:, 1] = BCxsnorm[:, 1] / 10
    Mx = ot.dist(BCnorm, BCxsnorm)
    weightBCs = Compute_Weights(BCs)
    weightBCxs = Compute_Weights(BCxs)
    Gsx = ot.unbalanced.mm_unbalanced(
        weightBCs, weightBCxs, Mx, 1, div="l2", numItermax=10000
    )
    mult_Mx = np.zeros((len(BCs), len(BCxs)))
    for i in range(len(BCs)):
        for j in range(len(BCxs)):
            mult_Mx[i, j] = BCxs[j][1] / BCs[i][1]
    # shiftsX = np.sum(np.multiply(mult_Mx, Gsx), axis = 1)/np.sum(Gsx, axis = 1)
    shiftsX = []
    for i in range(len(BCs)):
        shiftsX.append(mult_Mx[i, np.argmax(Gsx[i, :])])
    return np.array(shiftsX)


def Compute_death_shifts(
    BCs, BCxs, weights, weightsx, verbose=False, living_weighting=0
):
    log_deaths = np.log(BCs)
    log_deathsx = np.log(BCxs)
    log_deaths[:, 0] = log_deaths[:, 0] * living_weighting
    log_deathsx[:, 0] = log_deathsx[:, 0] * living_weighting

    Mx = ot.dist(
        log_deaths, log_deathsx
    )  # np.reshape(log_deaths,(len(log_deaths),2)),np.reshape(log_deathsx,(len(log_deathsx),2)))
    # Gsx = ot.unbalanced.mm_unbalanced(np.log(weights), np.log(weightsx),Mx, 1, div='l2')
    # weights = np.ones(weights.shape)
    # weightsx = np.ones(weightsx.shape)
    if verbose:
        print(len(weights))
        print(len(weightsx))
    weights_norm = weights / np.sum(weights)
    weights_norm_x = weightsx / np.sum(weightsx)
    Gsx = ot.emd(weights_norm, weights_norm_x, Mx)
    shifts = np.exp(
        np.divide(Gsx @ log_deathsx[:, 1], np.sum(Gsx, axis=1)) - log_deaths[:, 1]
    )
    return shifts


def Compute_Max_try_2(
    BCs, BCxs, BCys, BC_xys, BC_anti_xys, verbose=False, living_weighting=0
):
    weights = Compute_Weights_mult(BCs)
    weightsx = Compute_Weights_mult(BCxs)
    weightsy = Compute_Weights_mult(BCys)
    weights_xy = Compute_Weights_mult(BC_xys)
    weights_anti_xy = Compute_Weights_mult(BC_anti_xys)
    shiftsX = Compute_death_shifts(BCs, BCxs, weights, weightsx, verbose=verbose)
    shiftsY = Compute_death_shifts(BCs, BCys, weights, weightsy, verbose=verbose)
    shiftsXY = Compute_death_shifts(BCs, BC_xys, weights, weights_xy, verbose=verbose)
    shiftsAntiXY = Compute_death_shifts(
        BCs, BC_anti_xys, weights, weights_anti_xy, verbose=verbose
    )
    BC_shifts = [
        [shiftsX, weights],
        [shiftsXY, weights],
        [shiftsY, weights],
        [shiftsAntiXY, weights],
    ]
    return find_max(BC_shifts, threshold=False)


def Compute_Maxs(original_BCs, Scaled_BCs, parameters, verbose=False):
    original_weights = Compute_Weights_mult(original_BCs)
    BC_shifts = []
    for BCs in Scaled_BCs:
        weights = Compute_Weights_mult(BCs)
        shifts = Compute_death_shifts(
            original_BCs, BCs, original_weights, weights, verbose=verbose
        )
        BC_shifts.append([shifts, original_weights])
    return find_max_list(BC_shifts, parameters, threshold=False)


def Compute_stds(original_BCs, Scaled_BCs, parameters, verbose=False):
    original_weights = Compute_Weights_mult(original_BCs)
    BC_shifts = []
    for BCs in Scaled_BCs:
        weights = Compute_Weights_mult(BCs)
        shifts = Compute_death_shifts(
            original_BCs, BCs, original_weights, weights, verbose=verbose
        )
        BC_shifts.append([shifts, original_weights])
    return find_stds(BC_shifts, parameters)


def Compute_Shifts_try_2(
    BCs, BCxs, BCys, BC_xys, BC_anti_xys, parameters, save=False, verbose=False
):
    weights = Compute_Weights_mult(BCs)
    weightsx = Compute_Weights_mult(BCxs)
    weightsy = Compute_Weights_mult(BCys)
    weights_xy = Compute_Weights_mult(BC_xys)
    weights_anti_xy = Compute_Weights_mult(BC_anti_xys)
    living_weighting = 0.0
    BC_weights = [
        [BCs[:, 1], weights],
        [BCxs[:, 1], weightsx],
        [BCys[:, 1], weightsy],
        [BC_xys[:, 1], weights_xy],
        [BC_anti_xys[:, 1], weights_anti_xy],
    ]
    density_diagram(
        BC_weights,
        ["normal", "0°", "90°", "45°", "135°"],
        [0.1, 6],
        "deaths",
        parameters,
        save,
    )
    shiftsX = Compute_death_shifts(
        BCs, BCxs, weights, weightsx, verbose=verbose, living_weighting=living_weighting
    )
    shiftsY = Compute_death_shifts(
        BCs, BCys, weights, weightsy, verbose=verbose, living_weighting=living_weighting
    )
    shiftsXY = Compute_death_shifts(
        BCs,
        BC_xys,
        weights,
        weights_xy,
        verbose=verbose,
        living_weighting=living_weighting,
    )
    shiftsAntiXY = Compute_death_shifts(
        BCs,
        BC_anti_xys,
        weights,
        weights_anti_xy,
        verbose=verbose,
        living_weighting=living_weighting,
    )
    BC_shifts = [
        [shiftsX, weights],
        [shiftsY, weights],
        [shiftsXY, weights],
        [shiftsAntiXY, weights],
    ]
    density_diagram(
        BC_shifts,
        ["0°", "90°", "45°", "135°"],
        [0.9, 2.1],
        "mult. death shift",
        parameters,
        save,
    )


def rotation_matrix(phi):
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])


def Extract_BCs_from_Points(
    points, max_edge_length_Barcodes=1000000, min_persistance_Barcodes=0.0
):
    Test_Komplex = gd.AlphaComplex(points)
    Rips_simplex_tree_sample = Test_Komplex.create_simplex_tree(
        max_alpha_square=max_edge_length_Barcodes**2
    )
    Barcodes = Rips_simplex_tree_sample.persistence(
        min_persistence=min_persistance_Barcodes
    )
    return Barcodes


def Compute_Scaled_Deaths_List(all_spheres, parameters, dim=1, verbose=False):
    total_Barcodes = []
    for parameter in parameters:
        angle, scaling = parameter
        rotation = rotation_matrix(angle)
        all_spheres_scaled = all_spheres @ rotation @ np.diag([1, scaling]) @ rotation.T
        Barcodes = Extract_BCs_from_Points(all_spheres_scaled)
        total_Barcodes.append(plain_BCs(Barcodes, dim))
    normal_BCs = plain_BCs(Extract_BCs_from_Points(all_spheres), dim)
    return normal_BCs, total_Barcodes


def Compute_Scaled_Deaths(all_spheres, new=False, plain=False, verbose=False):
    rotation = rotation_matrix(np.pi / 4)
    all_spheresx = all_spheres @ np.diag([2, 1])
    all_spheresy = all_spheres @ np.diag([1, 2])
    all_spheres_xy = all_spheres @ rotation @ np.diag([1, 2]) @ rotation.T
    all_spheres_anti_xy = all_spheres @ rotation @ np.diag([2, 1]) @ rotation.T
    Barcodes = Extract_BCs_from_Points(all_spheres)
    Barcodesx = Extract_BCs_from_Points(all_spheresx)
    Barcodesy = Extract_BCs_from_Points(all_spheresy)
    Barcodes_xy = Extract_BCs_from_Points(all_spheres_xy)
    Barcodes_anti_xy = Extract_BCs_from_Points(all_spheres_anti_xy)
    if plain == True:
        return (
            plain_BCs(Barcodes),
            plain_BCs(Barcodesx),
            plain_BCs(Barcodesy),
            plain_BCs(Barcodes_xy),
            plain_BCs(Barcodes_anti_xy),
        )
    normal_barcodes = pick_barcodes(Barcodes)
    normal_barcodesx = pick_barcodes(Barcodesx)
    normal_barcodesy = pick_barcodes(Barcodesy)
    if new == True:
        return Filter_BCs(Barcodes), Filter_BCs(Barcodesx), Filter_BCs(Barcodesy)
    return (
        sort_barcodes(normal_barcodes, num_samples),
        sort_barcodes(normal_barcodesx, num_samples),
        sort_barcodes(normal_barcodesy, num_samples),
    )


def reject_outliers(data, m=1.5):
    data = np.array([x for x in data if not np.isnan(x)])
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def Filter_BCs(BCs):
    return np.array(
        [barcode[1] for barcode in BCs if barcode[0] == 1 and barcode[1][1] != np.inf]
    )


def plain_BCs(Barcodes, dim=1):
    barcodes = []
    for bar in Barcodes:
        if dim == 1:
            if bar[0] == 1 and bar[1][1] != np.inf:
                barcodes.append(np.array(bar[1]) ** 0.5)
        else:
            if bar[0] == 0 and bar[1][1] != np.inf:
                barcodes.append(np.array(bar[1]) ** 0.5)
    return np.array(barcodes)


def pick_barcodes(Barcodes):
    barcodes = []
    for bar in Barcodes:
        if bar[0] == 1 and bar[1][1] != np.inf:
            barcodes.append([bar[1][1] ** 0.5 - bar[1][0] ** 0.5, bar[1][1]])
    return np.array(barcodes)


def sort_barcodes(barcodes, num):
    return barcodes[barcodes[:, 0].argsort()][: int(num * 0.9), 1]


def find_max(BC_weights, true_max=True, threshold=False):
    # kernels = []
    axis_name = "death_shifts"
    # thresh =
    arguments = np.linspace(1.0, 2.0, 1000)
    maxima = []
    for i, BC_data in enumerate(BC_weights):
        BC_df = pd.DataFrame(
            np.array(BC_data).transpose(), columns=[axis_name, "weight"]
        )
        if true_max:
            kernel = scipy.stats.gaussian_kde(
                BC_df[axis_name], weights=BC_df["weight"], bw_method=0.2
            )
            # kernels.append(kernel)
            values = kernel(arguments)
            if threshold:
                cur_max = np.max(values)
                maxima.append(
                    arguments[
                        np.max(
                            [x for x, y in enumerate(values) if y >= thresh * cur_max]
                        )
                    ]
                )
            else:
                maxima.append(arguments[np.argmax(values)])
        else:
            maxima.append(np.average(a=BC_df[axis_name], weights=BC_df["weight"]))
    return maxima


def find_stds(BC_weights, parameters):
    stds = []
    axis_name = "death_shifts"
    maxima = []
    for i, BC_data in enumerate(BC_weights):
        arguments = np.linspace(1.0, parameters[i][1], 1000)
        BC_df = pd.DataFrame(
            np.array(BC_data).transpose(), columns=[axis_name, "weight"]
        )
        kernel = scipy.stats.gaussian_kde(
            BC_df[axis_name], weights=BC_df["weight"], bw_method=0.2
        )
        values = kernel(arguments)
        maximum_pos = arguments[np.argmax(values)]
        maxima.append(maximum_pos)
        data = np.array(BC_data).transpose()
        # print(data.shape)
        values = data[:, 0]
        # print(values)
        weights = data[:, 1]
        mean = maximum_pos
        total_weight = np.sum(weights)
        std_data = np.sqrt(np.sum((values - mean) ** 2 * weights) / total_weight)
        stds.append(std_data)
        # print(((values-mean)**2)*np.array(weights))
    return stds  # np.array(stds)


def find_max_list(BC_weights, parameters, true_max=True, threshold=False, weights = True):
    axis_name = "death_shifts"
    maxima = []
    for i, BC_data in enumerate(BC_weights):
        arguments = np.linspace(1.0, parameters[i][1], 1000)
        BC_df = pd.DataFrame(
            np.array(BC_data).transpose(), columns=[axis_name, "weight"]
        )
        if true_max:
            if weights:
                kernel = scipy.stats.gaussian_kde(
                    BC_df[axis_name], weights=BC_df["weight"], bw_method=0.2
                )
            else:
                kernel = scipy.stats.gaussian_kde(
                    BC_df[axis_name], bw_method=0.2)
            values = kernel(arguments)
            if threshold:
                cur_max = np.max(values)
                maxima.append(
                    arguments[
                        np.max(
                            [x for x, y in enumerate(values) if y >= thresh * cur_max]
                        )
                    ]
                )
            else:
                maxima.append(arguments[np.argmax(values)])
        else:
            maxima.append(np.average(a=BC_df[axis_name], weights=BC_df["weight"]))
    return maxima

def run_zero_dim_hom(points):
    parameters = np.array([[i / 20 * np.pi, 2] for i in range(20)])
    for scale in np.linspace(2.0, 2.0, 0):
        new_parameters = np.array([[i / 10 * np.pi, scale] for i in range(10)])
        parameters = np.concatenate((parameters, new_parameters), axis=0)
    BCs, BC_list = Compute_Scaled_Deaths_List(
        points, parameters=parameters, verbose=False, dim=0
    )
    
    return BCs, BC_list, parameters

def Compute_death_shifts_zero_dim(
    BCs, BCxs, verbose=False
):
    log_deaths = np.log(BCs[:,1:])
    log_deathsx = np.log(BCxs[:,1:])
    Gsx = ot.emd_1d(log_deaths,log_deathsx, p = 0.5)
    shifts = np.exp(
        log_deathsx[:,0]- log_deaths[:,0]
    )
    shift_dest = np.exp(log_deathsx[:,0])
    return shifts,shift_dest