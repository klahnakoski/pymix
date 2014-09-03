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
#
#       This file is part of the Python Mixture Package
#
#       file:    mixture.py
#       author: Benjamin Georgi
#
#       Copyright (C) 2004-2009 Benjamin Georgi
#       Copyright (C) 2004-2009 Max-Planck-Institut fuer Molekulare Genetik,
#                               Berlin
#
#       Contact: georgi@molgen.mpg.de
#
#       This library is free software; you can redistribute it and/or
#       modify it under the terms of the GNU Library General Public
#       License as published by the Free Software Foundation; either
#       version 2 of the License, or (at your option) any later version.
#
#       This library is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#       Library General Public License for more details.
#
#       You should have received a copy of the GNU Library General Public
#       License along with this library; if not, write to the Free
#       Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
################################################################################


from math import exp, sqrt, log

import numpy as np
import scipy
from scipy import stats, special


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


def wrap_gsl_dirichlet_sample(alpha, n):
    result = scipy.random.dirichlet(alpha, n)
    return result


def get_log_normal_inverse_gamma_prior_density(mu_p, kappa, dof, scale, cmu, csigma):
    output = [0]*len(cmu)
    for i in range(len(cmu)):
        output[i] = log(pow((pow(csigma[i], 2.0) ), (- (dof + 2.0) / 2.0)) * exp(-scale / (2.0 * pow(csigma[i], 2.0))))
        output[i] += log(gsl_ran_gaussian_pdf(cmu[i] - mu_p, sqrt(pow(csigma[i], 2.0) / kappa)))
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
        output = log(special.gamma(sum(alpha))) - np.sum(np.log(special.gamma(alpha))) + np.sum(np.log([xi ** (ai - 1.0) for xi, ai in zip(x, alpha)]))

    return output


def wrap_gsl_sf_lngamma(alpha):
    output = log(special.gamma(alpha))
    return output
