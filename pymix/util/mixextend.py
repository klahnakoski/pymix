## encoding: utf-8
################################################################################
#
#       This file is part of the Modified Python Mixture Package, original
#       source code is from https://svn.code.sf.net/p/pymix/code.  Also see
#       http://www.pymix.org/pymix/index.php?n=PyMix.Download
#
#       Changes made by: Kyle Lahnakoski (kyle@lahnakoski.com)
#
################################################################################


from math import exp, log, sqrt

import numpy as np
import scipy
from scipy import stats, special
from pyLibrary.maths import Math


def get_normalized_posterior_matrix(data):
    logsum = np.log(np.sum(np.exp(data), axis=0))
    result = data - logsum
    return result, np.sum(logsum)


def substract_matrix(a, b):
    result = np.subtract(a, b)
    return result


def add_matrix(a, b):
    result = np.add(a, b)
    return result

def get_log_normal_inverse_gamma_prior_density(mu_p, kappa, dof, scale, cmu, csigma):
    output = [0] * len(cmu)
    for i in range(len(cmu)):
        output[i] = Math.log(pow((pow(csigma[i], 2.0)), (- (dof + 2.0) / 2.0)) * exp(-scale / (2.0 * pow(csigma[i], 2.0))))
        output[i] += Math.log(gsl_ran_gaussian_pdf(cmu[i] - mu_p, sqrt(pow(csigma[i], 2.0) / kappa)))
    return output


def wrap_gsl_ran_gaussian_pdf(loc, scale, x):
    output = stats.norm(loc, scale).pdf(x)
    return output


def gsl_ran_gaussian_pdf(dx, scale):
    output = stats.norm(0.0, scale).pdf(dx)
    return output


# def wrap_gsl_dirichlet_pdf(alpha, x):
#     return exp(wrap_gsl_dirichlet_lnpdf(alpha, x))


def wrap_gsl_dirichlet_lnpdf(alpha, x):
    if hasattr(x[0], "__iter__"):
        output = [wrap_gsl_dirichlet_lnpdf(alpha, xi) for xi in x]
    else:
        output = Math.log(special.gamma(sum(alpha))) - np.sum(np.log(special.gamma(alpha))) + np.sum(np.log([xi ** (ai - 1.0) for xi, ai in zip(x, alpha)]))

    return output


def wrap_gsl_sf_lngamma(alpha):
    output = Math.log(special.gamma(alpha))
    return output
