# *****************************************************************************
#
#        This file is part of the General Hidden Markov Model Library,
#        GHMM version __VERSION__, see http:# ghmm.org
#
#        Filename: ghmm/ghmm/sviterbi.c
#        Authors:  Bernhard Knab, Benjamin Georgi
#
#        Copyright (C) 1998-2004 Alexander Schliep
#        Copyright (C) 1998-2001 ZAIK/ZPR, Universitaet zu Koeln
#        Copyright (C) 2002-2004 Max-Planck-Institut fuer Molekulare Genetik,
#                                Berlin
#
#        Contact: schliep@ghmm.org
#
#        This library is free software you can redistribute it and/or
#        modify it under the terms of the GNU Library General Public
#        License as published by the Free Software Foundation either
#        version 2 of the License, or (at your option) any later version.
#
#        This library is distributed in the hope that it will be useful,
#        but WITHOUT ANY WARRANTY without even the implied warranty of
#        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#        Library General Public License for more details.
#
#        You should have received a copy of the GNU Library General Public
#        License along with this library if not, write to the Free
#        Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
#
#        This file is version $Revision: 2267 $
#                        from $Date: 2009-04-24 11:01:58 -0400 (Fri, 24 Apr 2009) $
#              last change by $Author: grunau $.
#
# *****************************************************************************

from pyLibrary.maths import Math
from pymix.util.ghmm.wrapper import ARRAY_CALLOC, DBL_MAX, ighmm_dmatrix_stat_alloc
from pymix.util.logs import Log


class local_store_t:
    def __init__(self, smo, T):
        self.log_b = ighmm_dmatrix_stat_alloc(smo.N, T)

        self.phi = ARRAY_CALLOC(smo.N)
        self.phi_new = ARRAY_CALLOC(smo.N)
        self.psi = [[0]*smo.N for i in range(T)]


def sviterbi_precompute(smo, O, T, v):
    # Precomputing of Math.log(b_j(O_t))
    for t in range(T):
        for j in range(smo.N):
            cb = smo.s[j].calc_b(O[t])
            if cb == 0.0:
            # DBL_EPSILON ?
                v.log_b[j][t] = -DBL_MAX
            else:
                v.log_b[j][t] = Math.log(cb)


def ghmm_cmodel_viterbi(smo, O, T):

    v = local_store_t(smo, T)

    state_seq = ARRAY_CALLOC(T)
    # Precomputing of Math.log(bj(ot))
    sviterbi_precompute(smo, O, T, v)

    # Initialization for  t = 0
    for j in range(smo.N):
        if smo.s[j].pi == 0.0 or v.log_b[j][0] == -DBL_MAX:
            v.phi[j] = -DBL_MAX
        else:
            v.phi[j] = Math.log(smo.s[j].pi) + v.log_b[j][0]


    # Recursion
    for t in range(1, T):
        if smo.cos == 1:
            osc = 0
        else:
            if not smo.class_change.get_class:
                Log.error("get_class not initialized\n")

            #printf("1: cos = %d, k = %d, t = %d\n",smo.cos,smo.class_change.k,t)
            osc = smo.class_change.get_class(smo, O, smo.class_change.k, t - 1)
            if osc >= smo.cos:
                Log.error("get_class returned index %d but model has only %d classes not \n", osc, smo.cos)

        for j in range(smo.N):
            # find maximum
            # max_phi = phi[i] + log_in_a[j][i] ...
            max_value = -DBL_MAX
            v.psi[t][j] = -1
            for i in range(smo.N):
                if v.phi[i] > -DBL_MAX and Math.log(smo.s[j].in_a[osc][i]) > -DBL_MAX:
                    value = v.phi[i] + Math.log(smo.s[j].in_a[osc][i])
                    if value > max_value:
                        max_value = value
                        v.psi[t][j] = i

            # no maximum found (state is never reached or b(O[t]) = 0
            if max_value == -DBL_MAX or v.log_b[j][t] == -DBL_MAX:
                v.phi_new[j] = -DBL_MAX
            else:
                v.phi_new[j] = max_value + v.log_b[j][t]

        # replace old phi with new phi
        for j in range(smo.N):
            v.phi[j] = v.phi_new[j]

    # Termination
    max_value = -DBL_MAX
    state_seq[T - 1] = -1
    for j in range(smo.N):
        if v.phi[j] != -DBL_MAX and v.phi[j] > max_value:
            max_value = v.phi[j]
            state_seq[T - 1] = j

    if max_value == -DBL_MAX:
        Log.error("sequence can't be build from model, no backtracking possible")

    log_p = max_value
    # Backtracking
    for t in reversed(range(T - 1)):
        state_seq[t] = v.psi[t + 1][state_seq[t + 1]]

    return state_seq, log_p
