"""
Anisotropic Gaussian Mixture Models For 3D Object Localization

AGMM is very similar to regular Gaussian Mixture Models
except that we add an adjustment step after each iteration of the classic
EM algorithm. In effect, this unsupervised learning method does not
require the number of clusters to be specified beforehand. The steps are
implemented as follows:

Initilization step:
    -Use K-means to initialize clusters
    -Fit Gaussian with known singular values of covariance matrix

Expectation step:
    -Update point membership
    -Redistribute/delete singleton clusters

Maxization step:
    -Update mean and covariance for each cluster
    -Fit covariance "loosely" since the aim is to not fit to outliers

Adjustment:
    -Split clusters that have poorly fitted distributions
    -Combine clusters that overlap closely

Stopping condition:
    -Stop when nothing changes in the expectation step
"""

import numpy as np
import math
from os.path import join
import sklearn
import sys
from sklearn import mixture
from sklearn import cluster
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.stats import gaussian_kde
import scipy
from matplotlib.patches import Ellipse

WC_PATH = '/Users/ryan/Desktop/ir/wc/here_door_detections.wc'
PRINT_INTERVAL = 1
SINGULAR_VALUES = [0.3, 0.02, 0.02]
MAX_STEPS = 5
MIN_PROBABILITY = 1e-4

def import_coords(wc_path):
    wc = []
    with open(wc_path, 'r') as f: 
        for i in f.readlines():
            wc.append(map(float, i.split()))
    return np.array(wc)

def initialize_means(coords):
    km = cluster.KMeans(n_clusters=22)
    km.fit(coords)
    return km.cluster_centers_

def view_step_coords(original, centers):
    plt.scatter(original[:,0], original[:,1], s=15, edgecolors='none')
    plt.scatter(centers[:,0], centers[:,1], c='r')
    plt.show()

def view_step_ccs(original, ccs):
    plt.scatter(original[:,0], original[:,1], s=15, edgecolors='none', alpha=.5)
    centers = []
    for cc in ccs:
        centers.append(cc.mean)
        plot_cov_ellipse(cc.C[:2,:2], cc.mean[:2], nstd=2, fill=False, color='black')
    centers = np.array(centers)
    plt.scatter(centers[:,0], centers[:,1], c='r')

    plt.show()

def redistribute_singletons(ccs, delete=False):
    # find singleton clusters
    to_distribute = []
    for cc in ccs:
        if len(cc.members) == 0:
            print "[ERROR] cluster had 0 members."
            ccs.remove(cc)
        elif len(cc.members) == 1:
            to_distribute.append(cc.members[0])
            ccs.remove(cc)

    if not delete:
        # redistribute singletons to other clusters
        for point in to_distribute:
            closest = 0
            closest_dist = np.linalg.norm(ccs[0].mean-point,2)
            for j in range(1, len(ccs)):
                # for first step, use same metric as init step
                dist = np.linalg.norm(ccs[j].mean-point,2)
                if dist < closest_dist:
                    closest = j
                    closest_dist = dist
            ccs[closest].members.append(point)

# use to initialize all ccs after the initialization step
def initialize_clusters(coords, centers, singular_values):
    # init empty clusters
    ccs = []
    for center in centers:
        ccs.append(ClusterCenter(center))

    # assign each sample to a cluster naively
    for i in range(len(coords)):
        closest = 0
        closest_dist = np.linalg.norm(centers[0]-coords[i],2)
        for j in range(1, len(centers)):
            # for first step, use same metric as init step
            dist = np.linalg.norm(centers[j]-coords[i],2)
            if dist < closest_dist:
                closest = j
                closest_dist = dist
        ccs[closest].members.append(coords[i])

    # can't create a covariance matrix on one point,
    # so redistribute
    redistribute_singletons(ccs, delete=False)

    # fit gaussian and compute loss with assumed 
    # singular values of covariance matrix
    for cc in ccs:
        cov_mat = np.cov(np.array(cc.members).T)
        u,s,v = np.linalg.svd(cov_mat, full_matrices=True)
        fitted_cov = np.dot(u, np.dot(np.diag(singular_values), v))
        cc.C = fitted_cov
        model = scipy.stats.multivariate_normal(cc.mean, cc.C)
        prob_sum = 0
        for p in cc.members:
            prob_sum += model.pdf(p)
        cc.loss = prob_sum / len(cc.members)

    return ccs

def stopping_condition(coords, ccs, curr_step, max_step):
    if curr_step == max_step:
        return True
    return False

def update_expectation(ccs):
    for orig in ccs:
        to_del = []
        orig_model = scipy.stats.multivariate_normal(orig.mean, orig.C)
        for i in range(len(orig.members)):
            orig_prob = orig_model.pdf(orig.members[i])
            for cc in ccs:
                if cc is orig:
                    continue
                new_prob = scipy.stats.multivariate_normal(cc.mean, cc.C).pdf(orig.members[i])
                if new_prob > orig_prob:
                    to_del.append(i)
                    cc.members.append(orig.members[i])
        for d in to_del[::-1]:
            del orig.members[d]
    redistribute_singletons(ccs, delete=True)

def update_maximization(ccs, singular_values):
    for cc in ccs:
        cov_mat = np.cov(np.array(cc.members).T)
        u,s,v = np.linalg.svd(cov_mat, full_matrices=True)
        fitted_cov = np.dot(u, np.dot(np.diag(singular_values), v))
        cc.C = fitted_cov
        model = scipy.stats.multivariate_normal(cc.mean, cc.C)
        prob_sum = 0
        for p in cc.members:
            prob_sum += model.pdf(p)
        cc.loss = prob_sum / len(cc.members)

def update_adjustment(ccs):
    #TODO
    return

def run():
    # setup
    coords = import_coords(WC_PATH)

    # initialize means
    centers = initialize_means(coords)

    # move centers into ClusterCenter class
    ccs = initialize_clusters(coords, centers, SINGULAR_VALUES)

    # view initialize step
    view_step_ccs(coords, ccs)

    step = 0
    # while stopping condition is not satisfied
    while not stopping_condition(coords, ccs, step, MAX_STEPS):
        # EXPECTATION:
        # - update point membership
        update_expectation(ccs)

        # MAXIMIZATION:
        # - update mean and covariance
        update_maximization(ccs, SINGULAR_VALUES)

        # ADJUSTMENT:
        # - split & combine clusters based on gain/loss
        update_adjustment(ccs)
        
        step += 1
        if step % PRINT_INTERVAL == 0:
            print "{} steps complete".format(step)
        # view current step
        view_step_ccs(coords, ccs)

        sys.exit(1)
        

class ClusterCenter:
    def __init__(self, coord, neighbors=None, covariance_matrix=None, loss=None):
        self.mean = coord
        if neighbors is None:
            self.members = []
        else:
            self.members = neighbors
        self.C = covariance_matrix
        self.loss = loss    


def plot_cov_ellipse(cov, pos, nstd=1, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip             

if __name__ == '__main__':
    run()

