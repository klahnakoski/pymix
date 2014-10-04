#******************************************************************************
#*
#*       This file is part of the General Hidden Markov Model Library,
#*       GHMM version __VERSION__, see http:# ghmm.org
#*
#*       Filename: ghmm/ghmm/sviterbi.c
#*       Authors:  Bernhard Knab, Benjamin Georgi
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
#*       This file is version $Revision: 2267 $
#*                       from $Date: 2009-04-24 11:01:58 -0400 (Fri, 24 Apr 2009) $
#*             last change by $Author: grunau $.
#*
#******************************************************************************

from math import log
from util.ghmm.wrapper import ARRAY_CALLOC, ighmm_dmatrix_alloc, DBL_MAX, ighmm_dmatrix_stat_alloc
from vendor.pyLibrary.env.logs import Log


class local_store_t:
    def __init__(self, smo, T):
        self.log_b = ighmm_dmatrix_stat_alloc(smo.N, T)

        self.phi = ARRAY_CALLOC(smo.N)
        self.phi_new = ARRAY_CALLOC(smo.N)
        self.psi = ighmm_dmatrix_alloc(T, smo.N)


def sviterbi_precompute(smo, O, T, v):
    # Precomputing of log(b_j(O_t))
    for t in range(0, T):
        for j in range(0, smo.N):
            cb = smo.s[j].calc_b(O[t][0])
            if cb == 0.0:
            # DBL_EPSILON ?
                v.log_b[j][t] = -DBL_MAX
            else:
                v.log_b[j][t] = log(cb)


def ghmm_cmodel_viterbi(smo, O, T):
    state_seq = None

    # T is length of sequence divide by dimension to represent the number of time points
    T /= smo.dim

    v = local_store_t(smo, T)

    state_seq = ARRAY_CALLOC(T)
    # Precomputing of log(bj(ot))
    sviterbi_precompute(smo, O, T, v)

    # Initialization for  t = 0
    for j in range(0, smo.N):
        if smo.s[j].pi == 0.0 or v.log_b[j][0] == -DBL_MAX:
            v.phi[j] = -DBL_MAX
        else:
            v.phi[j] = log(smo.s[j].pi) + v.log_b[j][0]


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

        for j in range(0, smo.N):
            # find maximum
            # max_phi = phi[i] + log_in_a[j][i] ...
            max_value = -DBL_MAX
            v.psi[t][j] = -1
            for i in range(0, smo.s[j].in_states):
                if (v.phi[smo.s[j].in_id[i]] > -DBL_MAX and
                        log(smo.s[j].in_a[osc][i]) > -DBL_MAX):
                    value = v.phi[smo.s[j].in_id[i]] + log(smo.s[j].in_a[osc][i])
                    if value > max_value:
                        max_value = value
                        v.psi[t][j] = smo.s[j].in_id[i]



            # no maximum found (state is never reached or b(O[t]) = 0
            if max_value == -DBL_MAX or v.log_b[j][t] == -DBL_MAX:
                v.phi_new[j] = -DBL_MAX
            else:
                v.phi_new[j] = max_value + v.log_b[j][t]
                # for j in range( 0,  smo.N):
            # replace old phi with new phi
        for j in range(0, smo.N):
            v.phi[j] = v.phi_new[j]

    # Termination
    max_value = -DBL_MAX
    state_seq[T - 1] = -1
    for j in range(0, smo.N):
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
