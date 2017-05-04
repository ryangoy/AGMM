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


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
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

mean = [0.0,0.0]
cov = [[5.0, -4.0,],
       [-4.0, 5.0]]

u,s,v = np.linalg.svd(cov, full_matrices=True)

g = [[4, 0],
     [0, 1]]

fitted_cov = np.dot(u, np.dot(g, v))
# fitted_cov = g

print s
x,y = np.random.multivariate_normal(mean=mean, cov=cov, size=1000).T

model = scipy.stats.multivariate_normal(mean, fitted_cov)
prob_sum = 0
min_proba = 1
for xi, yi in zip(x,y):
	proba = model.pdf([xi,yi])
	if proba < min_proba:
		min_proba = proba
	prob_sum += proba
print "min probability is {}".format(min_proba)
print "average probability: {}".format(prob_sum / len(x))



plot_cov_ellipse(fitted_cov, mean, nstd=2, fill=False, color='black')


plt.scatter(x, y, alpha=.2)
plt.xlim(-20, 20)
plt.ylim(-20, 20)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()