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


def density_diagram_zero(BCs, save=False, name = ""):
    #angle_variance, r_variance, phi, number = parameters
    plt.figure()
    sns.set(rc={"figure.figsize": (12, 8.0)})
    sns.set(font_scale=1)
    sns.set_style("white")
    n = len(BCs)
    for i, BC_data in enumerate(BCs):
        BC_df = pd.DataFrame(
            np.array(BC_data).transpose(), columns=["Multiplicative shift","weight"]
        )
        sns.kdeplot(
            BC_df,
            x="Multiplicative shift",
            #weights="weight",
            bw_method=0.3,
            color=matplotlib.cm.tab20(i),
            linewidth=1,
            label = str(int(360/n*(n-i-1))) + "°"
        )
    plt.legend(loc=2, prop={'size': 8})
    #plt.legend()
    if save:
        plt.savefig(name+".pdf", bbox_inches='tight')
        plt.close()

def density_diagram(BC_weights, legend, limits, axis_name, parameters, save):
    angle_variance, r_variance, phi, number = parameters
    diagram_title = (
        "Phi Variance = "
        + str(angle_variance)
        + ", R Variance = "
        + str(r_variance)
        + ", Phi = "
        + str(phi)
        + "°"
    )
    plt.figure()
    sns.set(rc={"figure.figsize": (12, 6.0)})
    sns.set(font_scale=3.5)
    sns.set_style("white")
    for i, BC_data in enumerate(BC_weights):
        if len(BC_weights) == 5:
            if i == 0:
                i = 4
            else:
                i = i - 1
        BC_df = pd.DataFrame(
            np.array(BC_data).transpose(), columns=[axis_name, "weight"]
        )
        if striped and i == 3:
            sns.kdeplot(
                BC_df,
                x=axis_name,
                weights="weight",
                bw_method=0.2,
                clip=limits,
                color=matplotlib.cm.tab10(i),
                linewidth=10,
                linestyle="--",
            )
        else:
            sns.kdeplot(
                BC_df,
                x=axis_name,
                weights="weight",
                bw_method=0.2,
                clip=limits,
                color=matplotlib.cm.tab10(i),
                linewidth=10,
            )
    plt.legend(legend)
    # plt.title(diagram_title)
    # plt.xlim(limits)
    # if len(BC_weights) == 4:
    #     plt.ylim((0,12))
    # else:
    #     plt.ylim((0,0.4))
    if save:
        zeros = ""
        if number < 10:
            zeros = "0" + zeros
        if number < 100:
            zeros = "0" + zeros
        plt.savefig(
            (str(len(BC_weights)) + file_path + zeros + str(number)).replace(".", "")
            + ".png"
        )
        plt.close()


def draw_points(points):
    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], s=0.1)


def draw_points_2(points):
    plt.figure(figsize=(20, 10))
    plt.scatter(points[:, 0], points[:, 1], s=10)


def make_pretty_diagrams(all_spheres):
    Test_Komplex = gd.AlphaComplex(points=all_spheres)
    Rips_simplex_tree_sample = Test_Komplex.create_simplex_tree(
        max_alpha_square=max_edge_length_Barcodes**2
    )
    Barcodes = Rips_simplex_tree_sample.persistence(
        min_persistence=min_persistance_Barcodes
    )
    normal_barcodes = []
    for bar in Barcodes:
        if bar[0] == 1:
            normal_barcodes.append(bar[1])
    all_spheresx = all_spheres @ np.diag([2, 1])
    all_spheresy = all_spheres @ np.diag([1, 2])
    Test_Komplexx = gd.AlphaComplex(points=all_spheresx)
    Rips_simplex_tree_samplex = Test_Komplexx.create_simplex_tree(
        max_alpha_square=max_edge_length_Barcodes**2
    )
    Barcodesx = Rips_simplex_tree_samplex.persistence(
        min_persistence=min_persistance_Barcodes
    )
    normal_barcodesx = []
    for bar in Barcodesx:
        if bar[0] == 1:
            normal_barcodesx.append(bar[1])
    Test_Komplexy = gd.AlphaComplex(points=all_spheresy)
    Rips_simplex_tree_sampley = Test_Komplexy.create_simplex_tree(
        max_alpha_square=max_edge_length_Barcodes**2
    )
    Barcodesy = Rips_simplex_tree_sampley.persistence(
        min_persistence=min_persistance_Barcodes
    )
    normal_barcodesy = []
    for bar in Barcodesy:
        if bar[0] == 1:
            normal_barcodesy.append(bar[1])
    print(normal_barcodesy)
    fig_per = plt.figure("Persistence Diagram", figsize=(10, 10))
    ax_per = plt.axes()
    ax_per.scatter(
        [x[0] for x in normal_barcodes],
        [x[1] for x in normal_barcodes],
        color=matplotlib.cm.tab10(0),
    )
    ax_per.scatter(
        [x[0] for x in normal_barcodesx],
        [x[1] for x in normal_barcodesx],
        color=matplotlib.cm.tab10(1),
    )
    ax_per.scatter(
        [x[0] for x in normal_barcodesy],
        [x[1] for x in normal_barcodesy],
        color=matplotlib.cm.tab10(2),
    )

    fig0 = plt.figure("3D Image of Points Used", figsize=(10, 10))
    ax0 = plt.axes()
    ax0.scatter(all_spheres[:, 0], all_spheres[:, 1], s=0.3)
    figdensities = plt.figure("Density of persistence", figsize=(10, 10))
    axdensities = plt.axes()
    sns.set_style("whitegrid")
    sns.kdeplot(np.array([x[1] for x in normal_barcodes]), bw_method=0.5)
    sns.kdeplot(np.array([x[1] for x in normal_barcodesx]), bw_method=0.5)
    sns.kdeplot(np.array([x[1] for x in normal_barcodesy]), bw_method=0.5)
    sns.kdeplot(
        reject_outliers(
            np.arctan(
                (np.array(normal_barcodesx) - np.array(normal_barcodes))
                / (np.array(normal_barcodesy) - np.array(normal_barcodes))
            )
        ),
        bw_method=0.5,
    )
