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
from math import erf, exp, pi, cos, erfc, sqrt
from numpy.random.mtrand import dirichlet
from pyLibrary.maths import Math
from pymix.util.ghmm import random_mt
from pymix.util.ghmm.wrapper import DBL_MIN
from pymix.util.logs import Log

def sqr(x):
    return x*x


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


def ighmm_rand_get_1overa(x, mean, variance):
    # Calulates 1/a(x, mean, u), with a = the integral from x til \infty over
    #     the Gauss density function
    if variance <= 0.0:
        Log.error("u <= 0.0 not allowed\n")

    erfc_value = erfc((x - mean) / sqrt(variance*2))

    if erfc_value <= DBL_MIN:
        Log.error("a ~= 0.0 critical!  (mue = %.2f, u =%.2f)\n", mean, variance)
        return erfc_value

    return 2.0 / erfc_value
