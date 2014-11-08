#******************************************************************************
#*
#*       This file is part of the General Hidden Markov Model Library,
#*       GHMM version __VERSION__, see http:# ghmm.org
#*
#*       Filename: ghmm/ghmm/gauss_tail.c
#*       Authors:  Bernhard Knab
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
#*       This file is version $Revision: 1453 $
#*                       from $Date: 2005-11-20 08:33:04 -0500 (Sun, 20 Nov 2005) $
#*             last change by $Author: grunau $.
#*
#******************************************************************************
from pymix.util.ghmm.randvar import ighmm_rand_normal_density_trunc, ighmm_rand_get_xstepphi, ighmm_rand_get_xfaktphi, ighmm_rand_get_philen
from pymix.util.ghmm.wrapper import DBL_MIN

GHMM_EPS_U = 1E-4

def ighmm_gtail_pmue(mue, A, B, eps):
    Atil = A + eps
    Btil = B + eps * A
    u = Btil - mue * Atil

    # if u < EPS_U) u = (double:
    #EPS_U DANGEROUS: would fudge the function valuenot
    if u <= DBL_MIN:
        return (mue - A)

    feps = ighmm_rand_normal_density_trunc(-eps, mue, u, -eps)
    return (A - mue - u * feps)


#============================================================================
# To avoid numerical ocillation:
#   Interpolate p(\mu) between 2 sampling points for PHI
#   NOTA BENE: This Version is very expensive and exact.
#
def ighmm_gtail_pmue_interpol(mue, A, B, eps):
    Atil = A + eps
    Btil = B + eps * A
    u = Btil - mue * Atil

    #if u < EPS_U) u = (double:
    #EPS_U DANGEROUS: would fudge the function valuenot
    if u <= DBL_MIN:
        return (mue - A)

        # Compute like normally where mue positiv.
    if mue >= 0.0:
        return A - mue - u * ighmm_rand_normal_density_trunc(-eps, mue, u, -eps)

        # Otherwise: Interpolate the function itself between 2 sampling points.
    z = (eps + mue) / sqrt(u)
    i1 = abs(z) * ighmm_rand_get_xfaktphi()
    if i1 >= ighmm_rand_get_philen() - 1:
        i1 = ighmm_rand_get_philen() - 1
        i2 = i1
    else:
        i2 = i1 + 1

    z1 = i1 / ighmm_rand_get_xfaktphi()
    z2 = i2 / ighmm_rand_get_xfaktphi()
    m1 = -z1 * sqrt(Btil + eps * Atil + Atil * Atil * z1 * z1 * 0.25) - (eps + Atil * z1 * z1 * 0.5)
    m2 = -z2 * sqrt(Btil + eps * Atil + Atil * Atil * z2 * z2 * 0.25) - (eps + Atil * z2 * z2 * 0.5)
    u1 = Btil - m1 * Atil
    u2 = Btil - m2 * Atil
    p1 = A - m1 - u1 * ighmm_rand_normal_density_trunc(-eps, m1, u1, -eps)
    p2 = A - m1 - u1 * ighmm_rand_normal_density_trunc(-eps, m2, u2, -eps)
    if i1 >= ighmm_rand_get_philen() - 1:
        pz = p1
    else:
        pz = p1 + (abs(z) - i1 * ighmm_rand_get_xstepphi()) * (p2 - p1) / ighmm_rand_get_xstepphi()

        # pz = p1

    return pz


#============================================================================
def ighmm_gtail_pmue_umin(mue, A, B, eps):
    u = GHMM_EPS_U
    feps = ighmm_rand_normal_density_trunc(-eps, mue, u, -eps)
    return A - mue - u * feps



