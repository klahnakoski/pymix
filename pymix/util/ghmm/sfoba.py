#******************************************************************************
#*
#*       This file is part of the General Hidden Markov Model Library,
#*       GHMM version __VERSION__, see http:# ghmm.org
#*
#*       Filename: ghmm/ghmm/sfoba.c
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
#*       This file is version $Revision: 2243 $
#*                       from $Date: 2008-11-05 08:05:05 -0500 (Wed, 05 Nov 2008) $
#*             last change by $Author: christoph_mpg $.
#*
#******************************************************************************
from math import log
from util.ghmm.types import kSilentStates
from util.ghmm.wrapper import DBL_MIN, ARRAY_CALLOC, ighmm_cmatrix_stat_alloc, GHMM_EPS_PREC
from vendor.pyLibrary.env.logs import Log


LOWER_SCALE_BOUND = 3.4811068399043105e-57


def sfoba_initforward(smo, alpha_1, omega, scale, b):
    scale[0] = 0.0
    if b == None:
        for i in range(0, smo.N):
            alpha_1[i] = smo.s[i].pi * smo.s[1].calc_b(omega)
            scale[0] += alpha_1[i]
    else:
        for i in range(0, smo.N):
            alpha_1[i] = smo.s[i].pi * b[i][smo.M]
            scale[0] += alpha_1[i]

    if scale[0] > DBL_MIN:
        c_0 = 1 / scale[0]
        for i in range(0, smo.N):
            alpha_1[i] *= c_0


def sfoba_stepforward(s, alpha_t, osc, b_omega):
    value = 0.0
    for i in range(0, s.in_states):
        id = s.in_id[i]
        value += s.in_a[osc][i] * alpha_t[id]

    value *= b_omega             # b_omega outside the sum
    return value
    # sfoba_stepforward


def ghmm_cmodel_forward(smo, O, T, b, alpha, scale):
    t = 0
    osc = 0

    # T is length of sequence divide by dimension to represent the number of time points
    T /= smo.dim
    # calculate alpha and scale for t = 0
    if b == None:
        sfoba_initforward(smo, alpha[0], O, scale, None)
    else:
        sfoba_initforward(smo, alpha[0], O, scale, b[0])

    if scale[0] <= DBL_MIN:
        Log.error(" means f(O[0], mue, u) << 0, first symbol very unlikely")

    log_p = log(scale[0])

    if smo.cos == 1:
        osc = 0
    else:
        if not smo.class_change.get_class:
            Log.error("get_class not initialized\n")

        # printf("1: cos = %d, k = %d, t = %d\n",smo.cos,smo.class_change.k,t)
        osc = smo.class_change.get_class(smo, O, smo.class_change.k, t)
        if osc >= smo.cos:
            Log.error("get_class returned index %d but model has only %d classes not \n", osc, smo.cos)

    for t in range(1, T):
        scale[t] = 0.0
        # b not calculated yet
        if b == None:
            for i in range(0, smo.N):
                alpha[t][i] = sfoba_stepforward(smo.s[i], alpha[t - 1], osc, smo.s[i].calc_b( O[t]))
                scale[t] += alpha[t][i]

        # b precalculated
        else:
            for i in range(0, smo.N):
                alpha[t][i] = sfoba_stepforward(smo.s[i], alpha[t - 1], osc, b[t][i][smo.M])
                scale[t] += alpha[t][i]

        if scale[t] <= DBL_MIN:        #
            Log.error(" seq. can't be build")

        c_t = 1 / scale[t]
        # scale alpha
        for i in range(0, smo.N):
            alpha[t][i] *= c_t
            # summation of log(c[t]) for calculation of log( P(O|lambda) )
        log_p -= log(c_t)

        if smo.cos == 1:
            osc = 0

        else:
            if not smo.class_change.get_class:
                Log.error("get_class not initialized\n")

            # printf("1: cos = %d, k = %d, t = %d\n",smo.cos,smo.class_change.k,t)
            osc = smo.class_change.get_class(smo, O, smo.class_change.k, t)
            if osc >= smo.cos:
                Log.error("get_class returned index %d but model has only %d classes not \n", osc, smo.cos)
    return log_p


def ghmm_cmodel_backward(smo, O, T, b, beta, scale):
    # T is length of sequence divide by dimension to represent the number of time points
    T /= smo.dim

    beta_tmp = ARRAY_CALLOC(smo.N)

    for t in range(0, T):
        # try differenent bounds here in case of problems
        #       like beta[t] = NaN
        if scale[t] < LOWER_SCALE_BOUND:
            Log.error("error")

    # initialize
    c_t = 1 / scale[T - 1]
    for i in range(0, smo.N):
        beta[T - 1][i] = 1
        beta_tmp[i] = c_t

    # Backward Step for t = T-2, ..., 0
    # beta_tmp: Vector for storage of scaled beta in one time step

    if smo.cos == 1:
        osc = 0

    else:
        if not smo.class_change.get_class:
            Log.error("get_class not initialized\n")

        osc = smo.class_change.get_class(smo, O, smo.class_change.k, T - 2)
        # printf("osc(%d) = %d\n",T-2,osc)
        if osc >= smo.cos:
            Log.error("get_class returned index %d but model has only %d classes not \n", osc, smo.cos)

    for t in reversed(range(0, T - 1)):
        if b == None:
            for i in range(0, smo.N):
                sum = 0.0
                for j in range(0, smo.s[i].out_states):
                    j_id = smo.s[i].out_id[j]
                    sum += smo.s[i].out_a[osc][j] * smo.s[j_id].calc_b(O[t+1]) * beta_tmp[j_id]

                beta[t][i] = sum

        else:
            for i in range(0, smo.N):
                sum = 0.0
                for j in range(0, smo.s[i].out_states):
                    j_id = smo.s[i].out_id[j]
                    sum += smo.s[i].out_a[osc][j] * b[t + 1][j_id][smo.M] * beta_tmp[j_id]

                    #printf("  smo.s[%d].out_a[%d][%d] * b[%d] * beta_tmp[%d] = %f * %f *
                    #            %f\n",i,osc,j,t+1,j_id,smo.s[i].out_a[osc][j], b[t + 1][j_id][smo.M], beta_tmp[j_id])

                beta[t][i] = sum
                # printf(" .   beta[%d][%d] = %f\n",t,i,beta[t][i])

        c_t = 1 / scale[t]
        for i in range(0, smo.N):
            beta_tmp[i] = beta[t][i] * c_t

        if smo.cos == 1:
            osc = 0

        else:
            if not smo.class_change.get_class:
                Log.error("get_class not initialized\n")

            # if t=1 the next iteration will be the last
            if t >= 1:
                osc = smo.class_change.get_class(smo, O, smo.class_change.k, t - 1)
                # printf("osc(%d) = %d\n",t-1,osc)
                if osc >= smo.cos:
                    Log.error("get_class returned index %d but model has only %d classes not \n", osc, smo.cos)


def ghmm_cmodel_logp(smo, O, T):
    alpha = ighmm_cmatrix_stat_alloc(T, smo.N)
    scale = ARRAY_CALLOC(T)
    # run forward alg.
    return ghmm_cmodel_forward(smo, O, T, None, alpha, scale)


def ghmm_cmodel_logp_joint(mo, O, len, S, slen):
    state_pos = 0
    pos = 0
    osc = 0
    dim = mo.dim

    prevstate = state = S[0]
    log_p = log(mo.s[state].pi)
    if not (mo.model_type & kSilentStates) or 1: # XXX not mo.silent[state]  :
        log_p += log(mo.s[state].calc_b(O[pos]))
        pos += 1

    for state_pos in range(1, len):
        state = S[state_pos]
        for j in range(0, mo.s[state].in_states):
            if prevstate == mo.s[state].in_id[j]:
                break
        if mo.cos > 1:
            if not mo.class_change.get_class:
                Log.error("get_class not initialized")

            osc = mo.class_change.get_class(mo, O, mo.class_change.k, pos)
            if osc >= mo.cos:
                Log.error("get_class returned index %d but model has only %d classes!", osc, mo.cos)

        if (j == mo.s[state].in_states or abs(mo.s[state].in_a[osc][j]) < GHMM_EPS_PREC):
            Log.error("Sequence can't be built. There is no transition from state %d to %d.", prevstate, state)

        log_p += log(mo.s[state].in_a[osc][j])

        if not (mo.model_type & kSilentStates) or 1: # XXX !mo.silent[state]
            log_p += log(mo.s[state].calc_b(O[pos]))
            pos += 1

        prevstate = state

    if pos < len:
        Log.note("state sequence too shortnot  processed only %d symbols", pos)
    if state_pos < slen:
        Log.note("sequence too shortnot  visited only %d states", state_pos)

    return log_p
