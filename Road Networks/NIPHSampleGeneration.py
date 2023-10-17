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

max_edge_length_Barcodes = 80
min_persistance_Barcodes = 0.1

def make_points_grid(x_spacing,y_spacing,noise_level, num_points_row = 30,phi = 0):
    points_grid = np.array([rotation_matrix(phi)@(np.array([x,y])+np.random.normal(0,noise_level,2)) for x in np.linspace(-x_spacing,x_spacing,num_points_row) for y in np.linspace(-y_spacing,y_spacing,num_points_row)])
    return points_grid

def cut_out(points,xmin,xmax,ymin,ymax):
    return points[np.where((points[:,0]>xmin) & (points[:,0]<xmax) & (points[:,1]>ymin) & (points[:,1]<ymax))]

def sample_ell(center, radius, n_per_sphere, scaling=[1, 1], phi=0):
    r = radius
    ndim = center.size
    x = np.random.normal(size=(n_per_sphere, ndim))
    ssq = np.sqrt(np.sum(x**2, axis=1))
    p = ((x.transpose() / ssq).transpose()) @ np.diag(scaling) * r
    p = p @ np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    p = p + center
    return p

def sample_cars(total_length,sections, section_lengths, car_density):
    car_coords = []
    for i in range(int(total_length*car_density)):
        place = np.random.rand()*total_length
        section_id = next(x[0] for x in enumerate(section_lengths) if x[1] > place)
        section = sections[section_id]
        car_coords.append(section[0]+(section[1]-section[0])*np.random.rand())
    return np.array(car_coords)

def ellipsis(center, radius, n_per_sphere, scaling=[1, 1], phi=0):
    return minmax_landmark_sampling(
        sample_ell(center, radius, 4 * n_per_sphere, scaling, phi), n_per_sphere
    )[0]


def rectangle(center, radius, n_per_line, scaling=[1, 1], phi=0):
    total = np.sum(scaling)
    a = sampleline(
        scaling, scaling * np.array([-1, 1]), int(n_per_line * scaling[0] / total / 2)
    )
    b = sampleline(
        scaling * np.array([1, -1]),
        scaling * np.array([-1, -1]),
        int(n_per_line * scaling[0] / total / 2),
    )
    c = sampleline(
        scaling, scaling * np.array([1, -1]), int(n_per_line * scaling[1] / total / 2)
    )
    d = sampleline(
        scaling * np.array([-1, 1]),
        scaling * np.array([-1, -1]),
        int(n_per_line * scaling[1] / total / 2),
    )
    rect = np.concatenate((a, b, c, d), axis=0)
    rect = rect * radius
    rect = rect @ np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    rect = rect + center
    return rect


def sampleline(start, end, n_per_line):
    x = np.random.rand(n_per_line, 1)
    p = x * (end - start).transpose() + start
    return p


def make_random_angle_spheres(
    num_samples,
    prob,
    seed,
    prob_rad=0,
    scaling_center=150,
    phi=np.pi / 4,
    rectangles=False,
    scaling=2,
    pts_per_sphere=100,
):
    np.random.seed(seed)
    all_spheres = np.array([[], []]).transpose()
    centers = np.random.rand(num_samples, 2) * scaling_center
    radii = (
        np.abs(np.random.rand(num_samples)) * prob_rad
        + 0.2
        + (1 - np.minimum(1, prob_rad)) * 0.8
    )
    for i in range(num_samples):
        psi = phi + np.random.normal(0, prob)
        if rectangles:
            new_sphere = rectangle(
                centers[i],
                radii[i],
                int(np.floor(pts_per_sphere * radii[i])),
                [scaling, 1],
                phi=psi,
            )
        else:
            new_sphere = ellipsis(
                centers[i],
                radii[i],
                int(np.floor(pts_per_sphere * radii[i])),
                [scaling, 1],
                phi=psi,
            )
        all_spheres = np.concatenate((all_spheres, new_sphere), axis=0)
    return all_spheres


def make_random_angle_spheres_rect(
    num_samples,
    prob,
    seed,
    prob_rad=0,
    scaling_center=150,
    phi=np.pi / 4,
    rectangles=False,
    scaling=2,
):
    np.random.seed(seed)
    all_spheres = np.array([[], []]).transpose()
    centers = np.random.rand(num_samples, 2) * (scaling_center, scaling_center / 2)
    radii = (
        np.abs(np.random.rand(num_samples)) * prob_rad
        + 0.2
        + (1 - np.minimum(1, prob_rad)) * 0.8
    )
    for i in range(num_samples):
        psi = phi + np.random.normal(0, prob)
        if rectangles:
            new_sphere = rectangle(
                centers[i],
                radii[i],
                int(np.floor(pts_per_sphere * radii[i])),
                [scaling, 1],
                phi=psi,
            )
        else:
            new_sphere = ellipsis(
                centers[i],
                radii[i],
                int(np.floor(pts_per_sphere * radii[i])),
                [scaling, 1],
                phi=psi,
            )
        all_spheres = np.concatenate((all_spheres, new_sphere), axis=0)
    return all_spheres


def minmax_landmark_sampling(points, n_landmarks):
    n_points = points.shape[0]
    landmarks = np.zeros((n_landmarks, points.shape[1]))
    landmarks_index = np.zeros((n_landmarks, points.shape[1]))
    new_index = np.random.randint(n_points)
    landmarks[0] = points[new_index]
    landmarks_index[0] = new_index
    dists = sklearn.metrics.pairwise_distances(points, landmarks[0].reshape(1, -1))
    for i in range(1, n_landmarks):
        distances = scipy.spatial.distance.cdist(points, landmarks[0:i])
        min_distances = np.min(distances, axis=1)
        new_index = np.argmax(min_distances)
        landmarks_index[i] = new_index
        landmarks[i] = points[new_index]
    return landmarks, landmarks_index


def sparsify(points, sparsity):
    psparse = list(points.copy())
    length = len(psparse)
    for i in range(int(floor(length * sparsity))):
        n = np.random.randint(0, len(psparse))
        psparse.pop(n)
    psparse = np.array(psparse)
    return psparse
