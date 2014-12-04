# *****************************************************************************
#
#        This file is part of the General Hidden Markov Model Library,
#        GHMM version __VERSION__, see http:# ghmm.org
#
#        Filename: ghmm/ghmm/viterbi.c
#        Authors:  Wasinee Rungsarityotin, Benjamin Georgi
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
#        This file is version $Revision: 1991 $
#                        from $Date: 2007-12-05 11:39:04 -0500 (Wed, 05 Dec 2007) $
#              last change by $Author: grunau $.
#
# *****************************************************************************
from pyLibrary.maths import Math
from pymix.util.ghmm.types import kSilentStates
from pymix.util.ghmm.wrapper import ARRAY_CALLOC, ighmm_dmatrix_stat_alloc, ighmm_cmatrix_alloc, DBL_MAX
from pymix.util.logs import Log


class viterbi_alloc:
    def __init__(self, mo, len):
        # Allocate the log_in_a's . individal lenghts
        self.log_in_a = ARRAY_CALLOC(mo.N)
        for j in range(mo.N):
            self.log_in_a[j] = ARRAY_CALLOC(mo.N)

        self.log_b = ighmm_cmatrix_alloc(mo.N, mo.M)
        self.phi = ARRAY_CALLOC(mo.N)
        self.phi_new = ARRAY_CALLOC(mo.N)
        self.psi = ighmm_dmatrix_stat_alloc(len, mo.N)
        self.path_len = ARRAY_CALLOC(mo.N)
        self.topo_order_length = 0
        self.topo_order = ARRAY_CALLOC(mo.N)


def Viterbi_precompute(mo, o, len, v):
    # Precomputing the Math.log(a_ij)
    for j in range(mo.N):
        for i in range(mo.N):
            if mo.s[j].in_a[i] == 0.0:        # DBL_EPSILON ?
                v.log_in_a[j][i] = +1 # Not used any further in the calculations
            else:
                v.log_in_a[j][i] = Math.log(mo.s[j].in_a[i])



    # Precomputing the Math.log(bj(ot))
    for j in range(mo.N):
        for t in range(mo.M):
            if mo.s[j].b[t] == 0.0:    # DBL_EPSILON ?
                v.log_b[j][t] = +1
            else:
                v.log_b[j][t] = Math.log(mo.s[j].b[t])


def viterbi_silent(mo, t, v):
    for topocount in range(mo.topo_order_length):
        St = mo.topo_order[topocount]
        if mo.silent[St]:    # Silent states
            # Determine the maximum
            # max_phi = phi[i] + log_in_a[j][i] ...
            max_value = -DBL_MAX
            max_id = -1
            for i in range(mo.N):
                if v.phi[i] != +1 and v.log_in_a[St][i] != +1:
                    value = v.phi[i] + v.log_in_a[St][i]
                    if value > max_value:
                        max_value = value
                        max_id = i



            # No maximum found (is, state never reached)
            #               or the output O[t] = 0.0:
            if max_id < 0:
                v.phi[St] = 1
            else:
                v.phi[St] = max_value
                v.psi[t][St] = max_id
                v.path_len[St] = v.path_len[max_id] + 1


#  Return the viterbi path of the sequence.
def ghmm_dmodel_viterbi(mo, o, len):
    # for silent states: initializing path length with a multiple
    #       of the sequence length
    #       and sort the silent states topological
    if mo.model_type & kSilentStates:
        mo.order_topological()


    # Allocate the matrices log_in_a, log_b,Vektor phi, phi_new, Matrix psi
    v = viterbi_alloc(mo, len)

    plen = ARRAY_CALLOC(mo.N)

    # Precomputing the Math.log(a_ij) and Math.log(bj(ot))
    Viterbi_precompute(mo, o, len, v)

    # Initialization, that is t = 0
    for j in range(mo.N):
        if mo.s[j].pi == 0.0 or v.log_b[j][o[0]] == +1: # instead of 0, DBL_EPS.?
            v.phi[j] = +1
        else:
            v.phi[j] = Math.log(mo.s[j].pi) + v.log_b[j][o[0]]
            v.path_len[j] = 1

    if mo.model_type & kSilentStates:  # could go into silent state at t=0
        viterbi_silent(mo, 0, v)


    # t > 0
    for t in range(1, len):
        for j in range(mo.N):
            # initialization of phi, psi
            v.phi_new[j] = +1
            v.psi[t][j] = -1

        for St in range(mo.N):
            # Determine the maximum
            # max_phi = phi[i] + log_in_a[j][i] ...
            if not (mo.model_type & kSilentStates) or not mo.silent[St]:
                max_value = -DBL_MAX
                max_id = -1
                for i in range(mo.N):
                    if v.phi[i] != +1 and v.log_in_a[St][i] != +1:
                        value = v.phi[i] + v.log_in_a[St][i]
                        if value > max_value:
                            max_value = value
                            max_id = i


                # No maximum found (is, state never reached)
                #                   or the output O[t] = 0.0:
                if max_id >= 0 and v.log_b[St][o[t]] != +1:
                    v.phi_new[St] = max_value + v.log_b[St][o[t]]
                    v.psi[t][St] = max_id
                    plen[St] = v.path_len[max_id] + 1


                    # complete time step for emitting states

        # Exchange pointers
        (v.phi, v.phi_new) = (v.phi_new, v.phi)
        (plen, v.path_len) = (v.path_len, plen)

        # complete time step for silent states
        if mo.model_type & kSilentStates:
            viterbi_silent(mo, t, v)

            # Next observation , increment time-step

    # Termination - find end state
    max_value = -DBL_MAX
    end_state = -1
    for j in range(mo.N):
        if v.phi[j] != +1 and v.phi[j] > max_value:
            max_value = v.phi[j]
            end_state = j

    if end_state < 0:
        Log.error("Sequence can't be generated from the model!")

    log_p = max_value
    len_path = v.path_len[end_state]

    # allocating state_seq array
    # state_seq = ARRAY_CALLOC(len_path+1)
    state_seq = ARRAY_CALLOC(len_path)
    t = len - 1
    state_seq_index = len_path - 1
    # state_seq[len_path] = -1
    state_seq[state_seq_index] = end_state
    state_seq_index -= 1
    prev_state = end_state

    # backtrace is simple if the path length is known
    for state_seq_index in reversed(range(len_path - 1)):
        next_state = v.psi[t][prev_state]
        state_seq[state_seq_index] = prev_state = next_state
        if not (mo.model_type & kSilentStates) or not mo.silent[next_state]:
            t -= 1

    if t > 0:
        Log.error("state_seq_index = %d, t = %d", state_seq_index, t)

    return state_seq, log_p


def ghmm_dmodel_viterbi_logp(mo, o, len, state_seq):
    vpath, log_p = ghmm_dmodel_viterbi(mo, o, len)
    return log_p
