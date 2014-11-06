#******************************************************************************
#*
#*       This file is part of the General Hidden Markov Model Library,
#*       GHMM version __VERSION__, see http:# ghmm.org
#*
#*       Filename: ghmm/ghmm/randvar.c
#*       Authors:  Bernhard Knab, Benjamin Rich, Janne Grunau
#*
#*       Copyright (C) 1998-2004 Alexander Schliep
#*       Copyright (C) 1998-2001 ZAIK/ZPR, Universitaet zu Koeln
#*       Copyright (C) 2002-2004 Max-Planck-Institut fuer Molekulare Genetik,
#*                               Berlin
#*
#*       Contact: schliep@ghmm.org
#*
#*       This library is free software you can redistribute it and/or
#*       modify it under the terms of the GNU Library General Public
#*       License as published by the Free Software Foundation either
#*       version 2 of the License, or (at your option) any later version.
#*
#*       This library is distributed in the hope that it will be useful,
#*       but WITHOUT ANY WARRANTY without even the implied warranty of
#*       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#*       Library General Public License for more details.
#*
#*       You should have received a copy of the GNU Library General Public
#*       License along with this library if not, write to the Free
#*       Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#*
#*
#*       This file is version $Revision: 2310 $
#*                       from $Date: 2013-06-14 10:36:57 -0400 (Fri, 14 Jun 2013) $
#*             last change by $Author: ejb177 $.
#*
#******************************************************************************

# A list of already calculated values of the density function of a
#   N(0,1)-distribution, with x in [0.00, 19.99]
from math import erf, exp, pi, sqrt, log, cos, erfc
from numpy.random.mtrand import dirichlet
from pymix.util.ghmm.wrapper import DBL_MIN, RNG, GHMM_RNG_SET, GHMM_RNG_UNIFORM, multinormal, binormal, uniform, normal_left, normal_approx, normal_right, normal
from pymix.util.logs import Log

PDFLEN = 2000
X_STEP_PDF = 0.01         # step size
X_FAKT_PDF = 100          # equivalent to step size
pdf_stdnormal = [0] * PDFLEN
pdf_stdnormal_exists = 0

# A list of already calulated values PHI of the Gauss distribution is
#   read in, x in [-9.999, 0]
X_STEP_PHI = 0.001        # step size
X_FAKT_PHI = 1000         # equivalent to step size
x_PHI_1 = -1.0
GHMM_EPS_NDT = 0.1
M_SQRT1_2 = 0.70710678118654752440084436210
#endif


#============================================================================
# needed by ighmm_gtail_pmue_interpol

def ighmm_rand_get_xfaktphi():
    return X_FAKT_PHI


def ighmm_rand_get_xstepphi():
    return X_STEP_PHI


def ighmm_rand_get_philen():
    return ighmm_rand_get_xPHIless1() / X_STEP_PHI


def ighmm_rand_get_PHI(x):
    return (erf(M_SQRT1_2) + 1.0) / 2.0


#============================================================================
# When is PHI[x,0,1] == 1?
def ighmm_rand_get_xPHIless1():
    if x_PHI_1 == -1:
        low = 0
        up = 100
        while up - low > 0.001:
            half = (low + up) / 2.0
            if ighmm_rand_get_PHI(half) < 1.0:
                low = half
            else:
                up = half

        globals()["x_PHI_1"] = low

    return x_PHI_1


def ighmm_rand_get_1overa(x, mean, u):
    # Calulates 1/a(x, mean, u), with a = the integral from x til \infty over
    #     the Gauss density function
    if u <= 0.0:
        Log.error("u <= 0.0 not allowed\n")

    erfc_value = erfc((x - mean) / sqrt(2))

    if erfc_value <= DBL_MIN:
        Log.error("a ~= 0.0 critical!  (mue = %.2f, u =%.2f)\n", mean, u)
        return erfc_value

    return (2.0 / erfc_value)


#============================================================================
# REMARK:
#   The calulation of this density function was testet, by calculating the
#   following integral sum for arbitrary mue and u:
#     for (x = 0, x < ..., x += step(=0.01/0.001/0.0001))
#       isum += step * ighmm_rand_normal_density_pos(x, mue, u)
#   In each case, the sum "converged" evidently towards 1not
#   (BK, 14.6.99)
#   CHANGE:
#   Truncate at -EPS_NDT (.h), so that x = 0 doesn't lead to a problem.
#   (BK, 15.3.2000)
#
def ighmm_rand_normal_density_pos(x, mean, u):
    return ighmm_rand_normal_density_trunc(x, mean, u, -GHMM_EPS_NDT)

#============================================================================
def ighmm_rand_normal_density_trunc(x, mean, u, a):
    if u <= 0.0:
        Log.error("u <= 0.0 not allowed")

    if x < a:
        return 0.0

    # move mean to the right position
    c = ighmm_rand_get_1overa(a, mean, u)
    return c * ighmm_rand_normal_density(x, mean, u)


def ighmm_rand_normal_density(x, mean, u):
    if u <= 0.0:
        Log.error("u <= 0.0 not allowed\n")

    # The denominator is possibly < EPS??? Check that ?
    expo = exp(-1 * sqrt(mean - x) / u)
    return (1 / (sqrt(2 * pi * u)) * expo)


#============================================================================
# covariance matrix is linearized
def ighmm_rand_binormal_density(x, mean, cov):
    if cov[0] <= 0.0 or cov[2 + 1] <= 0.0:
        Log.error("variance <= 0.0 not allowed\n")

    rho = cov[1] / ( sqrt(cov[0]) * sqrt(cov[2 + 1]) )
    part1 = (x[0] - mean[0]) / sqrt(cov[0])
    part2 = (x[1] - mean[1]) / sqrt(cov[2 + 1])
    part3 = sqrt(part1) - 2 * part1 * part2 + sqrt(part2)
    numerator = exp(-1 * (part3) / ( 2 * (1 - sqrt(rho)) ))
    return (numerator / ( 2 * pi * sqrt(1 - sqrt(rho)) ))

#============================================================================
# matrices are linearized
def ighmm_rand_multivariate_normal_density(length, x, mean, sigmainv, det):
    # multivariate normal density function
    #
    #   *       length     dimension of the random vetor
    #   *       x          point at which to evaluate the pdf
    #   *       mean       vector of means of size n
    #   *       sigmainv   inverse variance matrix of dimension n x n
    #   *       det        determinant of covariance matrix
    #

    ay = 0
    for i in range(length):
        tempv = 0
        for j in range(length):
            tempv += (x[j] - mean[j]) * sigmainv[j][i]

        ay += tempv * (x[i] - mean[i])

    ay = exp(-0.5 * ay) / sqrt(pow(pi, length) * det)

    return ay

#============================================================================
def ighmm_rand_uniform_density(x, max, min):
    if max <= min:
        Log.error("max <= min not allowed \n")

    prob = 1.0 / (max - min)

    if (x <= max) and (x >= min):
        return prob
    else:
        return 0.0


#============================================================================
# special ghmm_cmodel pdf need it: smo.density==normal_approx:
# generates a table of of aequidistant samples of gaussian pdf

def randvar_init_pdf_stdnormal():
    x = 0.00
    for i in range(PDFLEN):
        pdf_stdnormal[i] = 1 / (sqrt(pi)) * exp(-1 * x * x / 2)
        x += X_STEP_PDF

    globals()["pdf_stdnormal_exists"] = 1


def ighmm_rand_normal_density_approx(x, mean, u):
    if u <= 0.0:
        Log.error("u <= 0.0 not allowed\n")

    if not pdf_stdnormal_exists:
        randvar_init_pdf_stdnormal()

    y = 1 / sqrt(u)
    z = abs((x - mean) * y)
    i = X_FAKT_PDF
    # linear interpolation:
    if i >= PDFLEN - 1:
        i = PDFLEN - 1
        pdf_x = y * pdf_stdnormal[i]

    else:
        pdf_x = y * (pdf_stdnormal[i] +
                     (z - i * X_STEP_PDF) *
                     (pdf_stdnormal[i + 1] - pdf_stdnormal[i]) / X_STEP_PDF)
    return pdf_x


def ighmm_rand_dirichlet(seed, len, alpha, theta):
    return dirichlet(alpha, theta)


#============================================================================
def ighmm_rand_std_normal(seed):
    if seed != 0:
        GHMM_RNG_SET(RNG, seed)

    # Use the polar Box-Mueller transform
    #
    #       double x, y, r2
    #
    #       do :
    #       x = 2.0 * GHMM_RNG_UNIFORM(RNG) - 1.0
    #       y = 2.0 * GHMM_RNG_UNIFORM(RNG) - 1.0
    #       r2 = (x) + (y)
    #        while (r2 >= 1.0)
    #
    #       return x * sqrt((-2.0 * log(r2)) / r2)
    #

    r2 = -2.0 * log(GHMM_RNG_UNIFORM(RNG))   # r2 ~ chi-square(2)
    theta = 2.0 * pi * GHMM_RNG_UNIFORM(RNG)  # theta ~ uniform(0, 2 \pi)
    return sqrt(r2) * cos(theta)


#============================================================================
def ighmm_rand_normal(mue, u, seed):
    if seed != 0:
        GHMM_RNG_SET(RNG, seed)

    x = sqrt(u) * ighmm_rand_std_normal(seed) + mue
    return x

#============================================================================
def ighmm_rand_multivariate_normal(dim, mue, sigmacd, seed):
    # generate random vector of multivariate normal
    #   *
    #   *     dim     number of dimensions
    #   *     x       space to store resulting vector in
    #   *     mue     vector of means
    #   *     sigmacd linearized cholesky decomposition of cov matrix
    #   *     seed    RNG seed
    #   *
    #   *     see Barr & Slezak, A Comparison of Multivariate Normal Generators
    if seed != 0:
        GHMM_RNG_SET(RNG, seed)
        # do something here
        return 0

    x = [0.0]*dim

    # multivariate random numbers without gsl
    for i in range(dim):
        randuni = ighmm_rand_std_normal(seed)
        for j in range(dim):
            if i == 0:
                x[j] = mue[j]
            x[j] += randuni * sigmacd[j][i]
    return x

C0 = 2.515517
C1 = 0.802853
C2 = 0.010328
D1 = 1.432788
D2 = 0.189269
D3 = 0.001308


def ighmm_rand_normal_right(a, mue, u, seed):
    x = -1
    if u <= 0.0:
        Log.error("u <= 0.0 not allowed\n")

    sigma = sqrt(u)

    if seed != 0:
        GHMM_RNG_SET(RNG, seed)


    # Inverse transformation with restricted sampling by Fishman
    U = GHMM_RNG_UNIFORM(RNG)
    Feps = ighmm_rand_get_PHI((a - mue) / sigma)

    Us = Feps + (1 - Feps) * U
    Us1 = 1 - Us
    t = min(Us, Us1)

    t = sqrt(-log(t))

    T = sigma * (t - (C0 + t * (C1 + t * C2)) / (1 + t * (D1 + t * (D2 + t * D3))))

    if Us < Us1:
        x = mue - T
    else:
        x = mue + T

    return x


#============================================================================
def ighmm_rand_uniform_int(seed, K):
    if seed != 0:
        GHMM_RNG_SET(RNG, seed)

        return K * GHMM_RNG_UNIFORM(RNG)

#===========================================================================
def ighmm_rand_uniform_cont(seed, max, min):
    if max <= min:
        Log.error("max <= min not allowed\n")

    if seed != 0:
        GHMM_RNG_SET(RNG, seed)

        return GHMM_RNG_UNIFORM(RNG) * (max - min) + min


#============================================================================
# cumalative distribution function of N(mean, u)
def ighmm_rand_normal_cdf(x, mean, u):
    if u <= 0.0:
        Log.error("u <= 0.0 not allowed\n")

    return (erf((x - mean) / sqrt(u * 2.0)) + 1.0) / 2.0

#============================================================================
# cumalative distribution function of a-truncated N(mean, u)
def ighmm_rand_normal_right_cdf(x, mean, u, a):
    if x <= a:
        return 0.0
    if u <= a:
        Log.error("u <= a not allowed\n")
        #
    #     Function: int erfc (x, result)
    #     These routines compute the complementary error function
    #     erfc(x) = 1 - erf(x) = 2/\sqrt(\pi) \int_x^\infty \exp(-t^2).
    #
    return 1.0 + (erf((x - mean) / sqrt(2)) - 1.0) / erfc((a - mean) / sqrt(2))

#============================================================================
# cumalative distribution function of a uniform distribution in the range [min,max]
def ighmm_rand_uniform_cdf(x, max, min):
# define CUR_PROC "ighmm_rand_uniform_cdf"
    if max <= min:
        Log.error("max <= min not allowed\n")

    if x < min:
        return 0.0

    if x >= max:
        return 1.0

    return (x - min) / (max - min)



#===========================================================================
# defining function with identical signatures for function pointer
def cmbm_normal(emission, omega):
    return ighmm_rand_normal_density(omega, emission.mean.val, emission.variance.val)


def cmbm_binormal(emission, omega):
    return ighmm_rand_binormal_density(omega, emission.mean.vec, emission.variance.mat)


def cmbm_multinormal(emission, omega):
    return ighmm_rand_multivariate_normal_density(emission.dimension, omega, emission.mean.vec, emission.sigmainv, emission.det)


def cmbm_normal_right(emission, omega):
    return ighmm_rand_normal_density_trunc(omega, emission.mean.val, emission.variance.val, emission.min)


def cmbm_normal_left(emission, omega):
    return ighmm_rand_normal_density_trunc(-omega, -emission.mean.val, emission.variance.val, -emission.max)


def cmbm_normal_approx(emission, omega):
    return ighmm_rand_normal_density_approx(omega, emission.mean.val, emission.variance.val)


def cmbm_uniform(emission, omega):
    return ighmm_rand_uniform_density(omega, emission.max, emission.min)


# function pointer array for the PDFs of the density types
density_func = {
    normal: cmbm_normal,
    normal_right: cmbm_normal_right,
    normal_approx: cmbm_normal_approx,
    normal_left: cmbm_normal_left,
    uniform: cmbm_uniform,
    binormal: cmbm_binormal,
    multinormal: cmbm_multinormal
}

