# *****************************************************************************
#
#        This file is part of the General Hidden Markov Model Library,
#        GHMM version __VERSION__, see http:# ghmm.org
#
#        Filename: ghmm/ghmm/model.c
#        Authors:  Benhard Knab, Bernd Wichern, Benjamin Georgi, Alexander Schliep
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
#        This file is version $Revision: 2304 $
#                        from $Date: 2013-05-31 13:48:13 -0400 (Fri, 31 May 2013) $
#              last change by $Author: ejb177 $.
#
# *****************************************************************************
from pymix.util.ghmm import random_mt
from pymix.util.ghmm.dmodel import ghmm_dmodel
from pymix.util.ghmm.sequences import sequence
from pymix.util.ghmm.dstate import ghmm_dstate
from pymix.util.ghmm.types import kDiscreteHMM, kBackgroundDistributions, kSilentStates, kNoBackgroundDistribution, kTiedEmissions, kUntied, kHigherOrderEmissions, kLabeledStates, kLeftRight, kPairHMM, kTransitionClasses
from pymix.util.ghmm.wrapper import ARRAY_CALLOC, ARRAY_MALLOC, ARRAY_REALLOC, GHMM_EPS_PREC, sequence_max_symbol, GHMM_MAX_SEQ_LEN, ghmm_xmlfile_parse, ighmm_cvector_normalize
from pymix.util.logs import Log

DONE = 0
NOTVISITED = 1
VISITED = 2


def model_copy_vectors(mo, index, a_matrix, b_matrix, pi, fix):
    mo.s[index].pi = pi[index]
    mo.s[index].fix = fix[index]
    for i in range(0, mo.M):
        mo.s[index].b[i] = b_matrix[index][i]
    for i in range(0, mo.N):
        mo.s[index].out_a[i] = a_matrix[index][i]
        mo.s[index].in_a[i] = a_matrix[i][index]


def ghmm_dmodel_read(filename):
    pass


def ghmm_dmodel_direct_read(s, multip):
    pass

#============================================================================
# Produces models from given sequences
def ghmm_dmodel_from_sequence(sq):
    mo = ARRAY_CALLOC(sq.seq_number)
    max_symb = sequence_max_symbol(sq)
    for i in range(0, sq.seq_number):
        mo[i] = ghmm_dmodel_generate_from_sequence(sq.seq[i], sq.seq_len[i], max_symb + 1)
    return mo



def ghmm_dmodel_copy(mo):
    m2 = ghmm_dmodel(mo.M, mo.N, mo.model_type, [0]*mo.N, [0]*mo.N)
    m2.s = [ghmm_dstate() for i in range(mo.N)]

    if mo.model_type & kSilentStates:
        m2.silent = ARRAY_CALLOC(mo.N)
    if mo.model_type & kTiedEmissions:
        m2.tied_to = ARRAY_CALLOC(mo.N)
    if mo.model_type & kBackgroundDistributions:
        m2.background_id = ARRAY_CALLOC(mo.N)
        m2.bp = mo.bp

    if mo.model_type & kHigherOrderEmissions:
        m2.order = ARRAY_CALLOC(mo.N)
    if mo.model_type & kLabeledStates:
        m2.label = ARRAY_CALLOC(mo.N)

    m2.pow_lookup = ARRAY_MALLOC(mo.maxorder + 2)

    for i in range(0, mo.N):
        if mo.model_type & kHigherOrderEmissions:
            size = pow(mo.M, mo.order[i] + 1)
        else:
            size = mo.M

        nachf = mo.N
        vorg = mo.N

        m2.s[i].out_a = ARRAY_CALLOC(nachf)
        m2.s[i].in_a = ARRAY_CALLOC(vorg)
        m2.s[i].b = ARRAY_CALLOC(size)

        # copy the values
        for j in range(0, nachf):
            m2.s[i].out_a[j] = mo.s[i].out_a[j]

        for j in range(0, vorg):
            m2.s[i].in_a[j] = mo.s[i].in_a[j]

        # copy all b values for higher order states
        for m in range(0, size):
            m2.s[i].b[m] = mo.s[i].b[m]

        m2.s[i].pi = mo.s[i].pi
        m2.s[i].fix = mo.s[i].fix
        if mo.model_type & kSilentStates:
            m2.silent[i] = mo.silent[i]
        if mo.model_type & kTiedEmissions:
            m2.tied_to[i] = mo.tied_to[i]
        if mo.model_type & kLabeledStates:
            m2.label[i] = mo.label[i]
        if mo.model_type & kHigherOrderEmissions:
            m2.order[i] = mo.order[i]
        if mo.model_type & kBackgroundDistributions:
            m2.background_id[i] = mo.background_id[i]

    m2.N = mo.N
    m2.M = mo.M
    m2.prior = mo.prior

    m2.model_type = mo.model_type
    # not necessary but the history is at least initialised
    m2.emission_history = mo.emission_history
    return m2

#============================================================================
def ghmm_dmodel_check(mo):
    imag = 0

    # The sum of the Pi[i]'s is 1
    sum_ = 0.0
    for i in range(0, mo.N):
        sum_ += mo.s[i].pi

    if abs(sum_ - 1.0) >= GHMM_EPS_PREC:
        Log.error("sum_ Pi[i] != 1.0")

    # check each state
    for i in range(mo.N):
        sum_ = sum(mo.s[i].out_a)

        if sum_ == 0.0:
            Log.warning("sum of s[%d].out_a[*] = 0.0 (assumed final state but %d transitions)", i, mo.N)

        if abs(sum_ - 1.0) >= GHMM_EPS_PREC:
            Log.error("sum of s[%d].out_a[*] = %f != 1.0", i, sum_)

        # Sum the a[i][j]'s : normalized in transitions
        sum_ = mo.s[i].pi
        for j in range(mo.N):
            sum_ += mo.s[i].in_a[j]

        if abs(sum_) <= GHMM_EPS_PREC:
            imag = 1
            Log.error("state %d can't be reached", i)


        # Sum the b[j]'s: normalized emission probs
        sum_ = 0.0
        for j in range(mo.M):
            sum_ += mo.s[i].b[j]

        if imag:
            # not reachable states
            if (abs(sum_ + mo.M) >= GHMM_EPS_PREC):
                Log.error("state %d can't be reached but is not set as non-reachale state", i)

        elif (mo.model_type & kSilentStates) and mo.silent[i]:
            # silent states
            if sum_ != 0.0:
                Log.error("state %d is silent but has a non-zero emission probability", i)
        else:
            # normal states
            if abs(sum_ - 1.0) >= GHMM_EPS_PREC:
                Log.error("sum s[%d].b[*] = %f != 1.0", i, sum_)


def ghmm_dmodel_check_compatibility(mo, model_number):
    for i in range(1, model_number):
        if -1 == ghmm_dmodel_check_compatibel_models(mo[0], mo[i]):
            return -1

    return 0


def ghmm_dmodel_check_compatibel_models(mo, m2):
    if mo.N != m2.N:
        Log.error("different number of states (%d != %d)\n", mo.N, m2.N)

    if mo.M != m2.M:
        Log.error("different number of possible outputs (%d != %d)\n", mo.M, m2.M)

    for i in range(mo.N):
        if mo.N != m2.N:
            Log.error("different number of outstates (%d != %d) in state %d.\n", mo.N, m2.N, i)

    return 0


def ghmm_dmodel_generate_from_sequence(seq, seq_len, anz_symb):
    Log.error("in_id has been removed, do not know how that affects this")

    mo = ghmm_dmodel(seq_len, anz_symb)

    # All models generated from sequences have to be LeftRight-models
    mo.model_type = kLeftRight

    # Allocate memory for all vectors
    mo.s = ARRAY_CALLOC(mo.N)
    for i in range(0, mo.N):
        if i == 0:
            mo.s[i] = ghmm_dstate(mo.M, 0, 1)
        elif i == mo.N - 1:  # End state
            mo.s[i] = ghmm_dstate(mo.M, 1, 0)
        else:                      # others
            mo.s[i] = ghmm_dstate(mo.M, 1, 1)


    # Allocate states with the right values, the initial state and the end
    #     state extra
    for i in range(1, mo.N - 1):
        s = mo.s[i]
        s.pi = 0.0
        s.b[seq[i]] = 1.0         # others stay 0
        s.out_id = i + 1
        s.in_id = i - 1
        s.out_a = s.in_a = 1.0


    # Initial state
    s = mo.s[0]
    s.pi = 1.0
    s.b[seq[0]] = 1.0
    s.out_id = 1
    s.out_a = 1.0
    # No in_id and in_a

    # End state
    s = mo.s[mo.N - 1]
    s.pi = 0.0
    s.b[seq[mo.N - 1]] = 1.0   # All other b's stay zero
    s.in_id = mo.N - 2
    s.in_a = 1.0
    # No out_id and out_a

    ghmm_dmodel_check(mo)

    return mo


def get_random_output(mo, i, position):
    sum_ = 0.0

    p = random_mt.float23()

    for m in range(0, mo.M):
        # get the right index for higher order emission models
        e_index = mo.get_emission_index(i, m, position)

        # get the probability, exit, if the index is -1
        if -1 != e_index:
            sum_ += mo.s[i].b[e_index]
            if sum_ >= p:
                break

        else:
            Log.error("State has order %d, but in the history are only %d emissions.\n", mo.order[i], position)

    if mo.M == m:
        Log.error("no valid output choosen. Are the Probabilities correct? sum: %g, p: %g\n", sum_, p)

    return m


def ghmm_dmodel_generate_sequences(mo, seed, global_len, seq_number, Tmax):
    n = 0

    sq = sequence(seq_number)

    # allocating additional fields for the state sequence in the sequence class
    sq.states = ARRAY_CALLOC(seq_number)
    sq.states_len = ARRAY_CALLOC(seq_number)

    # A specific length of the sequences isn't given. As a model should have
    #     an end state, the konstant MAX_SEQ_LEN is used.
    if len <= 0:
        len = GHMM_MAX_SEQ_LEN

    if seed > 0:
        random_mt.set_seed( seed)


    # initialize the emission history
    mo.emission_history = 0

    while n < seq_number:
        sq.seq[n] = ARRAY_CALLOC(len)

        # for silent models we have to allocate for the maximal possible number
        #       of lables and states
        if mo.model_type & kSilentStates:
            sq.states[n] = ARRAY_CALLOC(len * mo.N)
        else:
            sq.states[n] = ARRAY_CALLOC(len)

        pos = label_pos = 0

        # Get a random initial state i
        p = random_mt.float23()
        sum_ = 0.0
        for state in range(mo.N):
            sum_ += mo.s[state].pi
            if sum_ >= p:
                break

        while pos < len:
            # save the state path and label
            sq.states[n][label_pos] = state
            label_pos += 1

            # Get a random output m if the state is not a silent state
            if not (mo.model_type & kSilentStates) or not (mo.silent[state]):
                m = get_random_output(mo, state, pos)
                mo.update_emission_history(m)
                sq.seq[n][pos] = m
                pos += 1


            # get next state
            p = random_mt.float23()
            if pos < mo.maxorder:
                max_sum = 0.0
                for j in range(0, mo.N):
                    if not (mo.model_type & kHigherOrderEmissions) or mo.order[j] <= pos:
                        max_sum += mo.s[state].out_a[j]

                if abs(max_sum) < GHMM_EPS_PREC:
                    Log.error("No possible transition from state %d "
                              "since all successor states require more history "
                              "than seen up to position: %d.",
                        state, pos)

                p *= max_sum

            sum_ = 0.0
            for j in range(0, mo.N):
                if not (mo.model_type & kHigherOrderEmissions) or mo.order[j] <= pos:
                    sum_ += mo.s[state].out_a[j]
                    if sum_ >= p:
                        break

            if sum_ == 0.0:
                Log.note("final state (%d) reached at position %d of sequence %d", state, pos, n)
                break

            state = j
            # while pos < len:
        # realocate state path and label sequence to actual size
        if mo.model_type & kSilentStates:
            sq.states[n] = ARRAY_REALLOC(sq.states[n], label_pos)

        sq.seq_len[n] = pos
        sq.states_len[n] = label_pos
        n += 1
        # while  n < seq_number :
    return sq


#============================================================================

def ghmm_dmodel_likelihood(mo, sq):
    log_p = 0.0
    for i in range(sq.seq_number):
        log_p_i = mo.logp(sq.seq[i], sq.seq_len[i])
        if log_p_i != +1:
            log_p += log_p_i
        else:
            Log.error("sequence[%d] can't be build.", i)


def ghmm_dmodel_get_transition(mo, i, j):
    if mo.s and mo.s[i].out_a and mo.s[j].in_a:
        return mo.s[i].out_a[j]
    return 0.0


def ghmm_dmodel_check_transition(mo, i, j):
    if mo.s and mo.s[i].out_a and mo.s[j].in_a:
        if mo.s[i].out_a[j] > 0.0:
            return 1
    return 0


def ghmm_dmodel_set_transition(mo, i, j, prob):
    if mo.s and mo.s[i].out_a and mo.s[j].in_a:
        mo.s[i].out_a[j] = prob
        mo.s[j].in_a[i] = prob


def ghmm_dmodel_direct_clean(mo_d, check):
    pass


def ghmm_dmodel_direct_check_data(mo_d, check):
#define CUR_PROC "ghmm_dmodel_direct_check_data"
    if check.r_a != mo_d.N or check.c_a != mo_d.N:
        Log.error("Incompatible dim. A (%d X %d) and N (%d)\n", check.r_a, check.c_a, mo_d.N)

    if check.r_b != mo_d.N or check.c_b != mo_d.M:
        Log.error("Incompatible dim. B (%d X %d) and N X M (%d X %d)\n", check.r_b, check.c_b, mo_d.N, mo_d.M)

    if check.len_pi != mo_d.N:
        Log.error("Incompatible dim. Pi (%d) and N (%d)\n", check.len_pi, mo_d.N)

    if check.len_fix != mo_d.N:
        Log.error("Incompatible dim. fix_state (%d) and N (%d)\n", check.len_fix, mo_d.N)


#============================================================================
# XXX symmetric not implemented yet
def ghmm_dmodel_prob_distance(m0, m, maxT, symmetric, verbose):
    STEPS = 40
    d = 0.0
    if verbose:
        step_width = maxT / 40
        steps = STEPS

    else:                          # else just one
        step_width = maxT

    d1 = ARRAY_CALLOC(steps)

    mo1 = m0
    mo2 = m

    for k in range(2):    # Two passes for the symmetric case
        # seed = 0 . no reseeding. Call  ghmm_rng_timeseed(RNG) externally
        seq0 = ghmm_dmodel_generate_sequences(mo1, 0, maxT + 1, 1, maxT + 1)

        if seq0 == None:
            Log.error(" generate_sequences failed not ")

        if seq0.seq_len[0] < maxT:      # There is an absorbing state

            # NOTA BENE: Assumpting the model delivers an explicit end state,
            #         the condition of a fix initial state is removed.

            # For now check that Pi puts all weight on state
            #
            #         t = 0
            #         for i in range( 0,  mo1.N):
            #         if mo1.s[i].pi > 0.001:
            #         t+=1
            #
            #         if t > 1:
            #         GHMM_LOG(LCONVERTED, "ERROR: No proper left-to-right model. Multiple start states")
            #         goto STOP
            #

            left_to_right = 1
            total = seq0.seq_len[0]

            while total <= maxT:

                # create a additional sequences at once
                a = (maxT - total) / (total / seq0.seq_number) + 1
                # printf("total=%d generating %d", total, a)
                tmp = ghmm_dmodel_generate_sequences(mo1, 0, 0, a, a)
                if tmp == None:
                    Log.error(" generate_sequences failed not ")

                seq.add(tmp)

                total = 0
                for i in range(0, seq0.seq_number):
                    total += seq0.seq_len[i]

        if left_to_right:
            for i, t in enumerate(range(step_width, maxT + step_width, step_width)):
                index = 0
                total = seq0.seq_len[0]

                # Determine how many sequences we need to get a total of t
                #           and adjust length of last sequence to obtain total of
                #           exactly t

                while total < t:
                    index += 1
                    total += seq0.seq_len[index]

                true_len = seq0.seq_len[index]
                true_number = seq0.seq_number

                if (total - t) > 0:
                    seq0.seq_len[index] = total - t
                seq0.seq_number = index

                p0 = ghmm_dmodel_likelihood(mo1, seq0)
                if p0 == +1 or p0 == -1:     # error!
                    Log.error("problem: ghmm_dmodel_likelihood failed not ")

                p = ghmm_dmodel_likelihood(mo2, seq0)
                if p == +1 or p == -1:       # what shall we do now?
                    Log.error("problem: ghmm_dmodel_likelihood failed not ")

                d = 1.0 / t * (p0 - p)

                if symmetric:
                    if k == 0:
                        # save d
                        d1[i] = d
                    else:
                        # calculate d
                        d = 0.5 * (d1[i] + d)
                if verbose and (not symmetric or k == 1):
                    Log.note("%d\t%f\t%f\t%f\n", t, p0, p, d)

                seq0.seq_len[index] = true_len
                seq0.seq_number = true_number



        else:

            true_len = seq0.seq_len[0]

            for i, t in enumerate(range(step_width, maxT + step_width, step_width)):
                seq0.seq_len[0] = t

                p0 = ghmm_dmodel_likelihood(mo1, seq0)
                # printf("   P(O|m1) = %f\n",p0)
                if p0 == +1:
                    Log.error("seq0 can't be build from mo1not ")

                p = ghmm_dmodel_likelihood(mo2, seq0)
                # printf("   P(O|m2) = %f\n",p)
                if p == +1:          # what shall we do now?
                    Log.error("problem: seq0 can't be build from mo2not ")

                d = (1.0 / t) * (p0 - p)

                if symmetric:
                    if k == 0:
                        # save d
                        d1[i] = d
                    else:
                        # calculate d
                        d = 0.5 * (d1[i] + d)

                if verbose and (not symmetric or k == 1):
                    Log.note("%d\t%f\t%f\t%f\n", t, p0, p, d)

            seq0.seq_len[0] = true_len

        if symmetric:
            mo1 = m
            mo2 = m0

        else:
            break

    return d


def ghmm_dstate_clean(my_state):
    pass


#==========================  Labeled HMMs  ================================

def ghmm_dmodel_label_generate_sequences(mo, seed, global_len, seq_number, Tmax):
    n = 0
    sq = sequence(seq_number)

    # allocating additional fields for the state sequence in the sequence class
    sq.states = ARRAY_CALLOC(seq_number)
    sq.states_len = ARRAY_CALLOC(seq_number)

    # allocating additional fields for the labels in the sequence class
    sq.state_labels = ARRAY_CALLOC(seq_number)
    sq.state_labels_len = ARRAY_CALLOC(seq_number)

    # A specific length of the sequences isn't given. As a model should have
    #     an end state, the konstant MAX_SEQ_LEN is used.
    if len <= 0:
        len = GHMM_MAX_SEQ_LEN

    if seed > 0:
        random_mt.set_seed( seed)


    # initialize the emission history
    mo.emission_history = 0

    while n < seq_number:
        sq.seq[n] = ARRAY_CALLOC(len)

        # for silent models we have to allocate for the maximal possible number
        #       of lables and states
        if mo.model_type & kSilentStates:
            sq.states[n] = ARRAY_CALLOC(len * mo.N)
            sq.state_labels[n] = ARRAY_CALLOC(len * mo.N)

        else:
            sq.states[n] = ARRAY_CALLOC(len)
            sq.state_labels[n] = ARRAY_CALLOC(len)

        pos = label_pos = 0

        # Get a random initial state i
        p = random_mt.float23()
        sum_ = 0.0
        for state in range(mo.N):
            sum_ += mo.s[state].pi
            if sum_ >= p:
                break

        while pos < len:
            # save the state path and label
            sq.states[n][label_pos] = state
            sq.state_labels[n][label_pos] = mo.label[state]
            label_pos += 1

            # Get a random output m if the state is not a silent state
            if not (mo.model_type & kSilentStates) or not (mo.silent[state]):
                m = get_random_output(mo, state, pos)
                mo.update_emission_history(m)
                sq.seq[n][pos] = m
                pos += 1


            # get next state
            p = random_mt.float23()
            if pos < mo.maxorder:
                max_sum = 0.0
                for j in range(0, mo.N):
                    if not (mo.model_type & kHigherOrderEmissions) or mo.order[j] < pos:
                        max_sum += mo.s[state].out_a[j]

                if abs(max_sum) < GHMM_EPS_PREC:
                    Log.error("No possible transition from state %d since all successor states require more history than seen up to position: %d.", state, pos)

                p *= max_sum

            sum_ = 0.0
            for j in range(0, mo.N):
                if not (mo.model_type & kHigherOrderEmissions) or mo.order[j] < pos:
                    sum_ += mo.s[state].out_a[j]
                    if sum_ >= p:
                        break

            if sum_ == 0.0:
                Log.note("final state (%d) reached at position %d of sequence %d", state, pos, n)
                break

            state = j
            # while pos < len:
        # realocate state path and label sequence to actual size
        if mo.model_type & kSilentStates:
            sq.states[n] = ARRAY_REALLOC(sq.states[n], label_pos)
            sq.state_labels[n] = ARRAY_REALLOC(sq.state_labels[n], label_pos)

        sq.seq_len[n] = pos
        sq.states_len[n] = label_pos
        sq.state_labels_len[n] = label_pos
        n += 1
        # while  n < seq_number :
    return (sq)


# Scales the output and transitions probs of all states in a given model
def ghmm_dmodel_normalize(mo):
    pi_sum = 0.0
    i_id = 0
    size = 1

    for i in range(mo.N):
        if mo.s[i].pi >= 0.0:
            pi_sum += mo.s[i].pi
        else:
            mo.s[i].pi = 0.0

        # check model_type before using state order
        if mo.model_type & kHigherOrderEmissions:
            size = pow(mo, mo.M, mo.order[i])

        # normalize transition probabilities
        ighmm_cvector_normalize(mo.s[i].out_a, 0, mo.N)

        # for every outgoing probability update the corrosponding incoming probability
        for j in range(0, mo.N):
            mo.s[j].in_a[i] = mo.s[i].out_a[j]

        # normalize emission probabilities, but not for silent states
        if not ((mo.model_type & kSilentStates) and mo.silent[i]):
            for m in range(size):
                ighmm_cvector_normalize(mo.s[i].b, m * mo.M, mo.M)

    for i in range(0, mo.N):
        mo.s[i].pi /= pi_sum


def ghmm_dmodel_add_noise(mo, level, seed):
    if level > 1.0:
        level = 1.0

    for i in range(0, mo.N):
        for j in range(0, mo.N):
            # add noise only to out_a, in_a is updated on normalisation
            mo.s[i].out_a[j] *= (1 - level) + (random_mt.float23() * 2 * level)

        if mo.model_type & kHigherOrderEmissions:
            size = pow(mo, mo.M, mo.order[i])
        for hist in range(size):
            for h in range(hist * mo.M, hist * mo.M + mo.M):
                mo.s[i].b[h] *= (1 - level) + (random_mt.float23() * 2 * level)

    ghmm_dmodel_normalize(mo)


def ghmm_dstate_transition_add(s, start, dest, prob):
    # resize the arrays
    s[start].out_a.insert(dest, prob)
    s[dest].in_a.insert(start, prob)


def ghmm_dstate_transition_del(s, start, dest):
    del s[start].out_a[dest]
    del s[dest].in_a[start]



#
#   Allocates a new ghmm_dbackground class and assigs the arguments to
#   the respective fields. Note: The arguments need allocation outside of this
#   function.
#
#   @return     :               0 on success, -1 on error
#   @param mo   :               one model
#   @param cur  :               a id of a state
#   @param times:               number of times the state cur is at least evaluated
#
def ghmm_dmodel_duration_apply(mo, cur, times):
    failed = 0

    if mo.model_type & kSilentStates:
        Log.error("Sorry, apply_duration doesn't support silent states yet\n")

    last = mo.N
    mo.N += times - 1

    mo.s = ARRAY_REALLOC(mo.s, mo.N)
    if mo.model_type & kSilentStates:
        mo.silent = ARRAY_REALLOC(mo.silent, mo.N)
        mo.topo_order = ARRAY_REALLOC(mo.topo_order, mo.N)

    if mo.model_type & kTiedEmissions:
        mo.tied_to = ARRAY_REALLOC(mo.tied_to, mo.N)
    if mo.model_type & kBackgroundDistributions:
        mo.background_id = ARRAY_REALLOC(mo.background_id, mo.N)

    size = pow(mo, mo.M, mo.order[cur] + 1)
    for i in range(last, mo.N):
        # set the new state
        mo.s[i].pi = 0.0
        mo.order[i] = mo.order[cur]
        mo.s[i].fix = mo.s[cur].fix
        mo.label[i] = mo.label[cur]
        mo.s[i].in_a = []
        mo.s[i].out_a = []

        mo.s[i].b = ARRAY_MALLOC(size)
        for j in range(0, size):
            mo.s[i].b[j] = mo.s[cur].b[j]

        if mo.model_type & kSilentStates:
            mo.silent[i] = mo.silent[cur]
            # XXX what to do with topo_order
            #         mo.topo_order[i] = ????????????

        if mo.model_type & kTiedEmissions:
            # XXX is there a clean solution for tied states?
            #         what if the current state is a tie group leader?
            #         the last added state should probably become
            #         the new tie group leader
            mo.tied_to[i] = kUntied
        if mo.model_type & kBackgroundDistributions:
            mo.background_id[i] = mo.background_id[cur]


    # move the outgoing transitions to the last state
    while len(mo.s[cur].out_a) > 0:
        if 0 == cur:
            ghmm_dstate_transition_add(mo.s, mo.N - 1, mo.N - 1, 0)
            ghmm_dstate_transition_del(mo.s, cur, 0)
        else:
            ghmm_dstate_transition_add(mo.s, mo.N - 1, 0, 0)
            ghmm_dstate_transition_del(mo.s, cur, 0)


    # set the linear transitions through all added states
    ghmm_dstate_transition_add(mo.s, cur, last, 1.0)
    for i in range(last + 1, mo.N):
        ghmm_dstate_transition_add(mo.s, i - 1, i, 1.0)

    ghmm_dmodel_normalize(mo)


def ghmm_dbackground_alloc(n, m, orders, B):
    ptbackground = ARRAY_CALLOC(1)

    # initialize name
    ptbackground.name = ARRAY_CALLOC(n)
    for i in range(n):
        ptbackground.name[i] = None

    ptbackground.n = n
    ptbackground.m = m
    if orders:
        ptbackground.order = orders
    if B:
        ptbackground.b = B

    return ptbackground


def ghmm_dbackground_copy(bg):
    new_order = ARRAY_MALLOC(bg.n)
    new_b = ARRAY_CALLOC(bg.n)

    for i in range(0, bg.n):
        new_order[i] = bg.order[i]
        b_i_len = pow(bg.m, bg.order[i] + 1)
        new_b[i] = ARRAY_CALLOC(b_i_len)
        for j in range(0, b_i_len):
            new_b[i][j] = bg.b[i][j]

    tmp = ghmm_dbackground_alloc(bg.n, bg.m, new_order, new_b)

    for i in range(0, bg.n):
        tmp.name[i] = bg.name[i]

    return tmp


def ghmm_dmodel_background_apply(mo, background_weight):
    if not (mo.model_type & kBackgroundDistributions):
        Log.error("Error: No background distributions")

    for i in range(mo.N):
        if mo.background_id[i] != kNoBackgroundDistribution:
            if mo.model_type & kHigherOrderEmissions:
                if mo.order[i] != mo.bp.order[mo.background_id[i]]:
                    Log.error("State (%d) and background order (%d) do not match in state %d. Background_id = %d",
                        mo.order[i],
                        mo.bp.order[mo.background_id[i]], i,
                        mo.background_id[i])
                    # XXX Cache in ghmm_dbackground
                size = pow(mo.M, mo.order[i] + 1)
                for j in range(size):
                    mo.s[i].b[j] = (1.0 - background_weight[i]) * mo.s[i].b[j] + background_weight[i] * mo.bp.b[mo.background_id[i]][j]
            else:
                if mo.bp.order[mo.background_id[i]] != 0:
                    Log.error("Error: State and background order do not match\n")
                    return -1

                for j in range(mo.M):
                    mo.s[i].b[j] = (1.0 - background_weight[i]) * mo.s[i].b[j] + background_weight[i] * mo.bp.b[mo.background_id[i]][j]


def ghmm_dmodel_get_uniform_background(mo, sq):
    n = 0
    sum_ = 0.0

    if not (mo.model_type & kBackgroundDistributions):
        Log.error("Error: Model has no background distribution")

    mo.bp = None
    mo.background_id = ARRAY_MALLOC(mo.N)

    # create a background distribution for each state
    for i in range(0, mo.N):
        mo.background_id[i] = mo.order[i]


    # allocate
    mo.bp = ARRAY_CALLOC(1)
    mo.bp.order = ARRAY_CALLOC(mo.maxorder)

    # set number of distributions
    mo.bp.n = mo.maxorder

    # set br.order
    for i in range(0, mo.N):
        if mo.background_id[i] != kNoBackgroundDistribution:
            mo.bp.order[mo.background_id[i]] = mo.order[i]

    # allocate and initialize br.b with zeros
    mo.bp.b = ARRAY_CALLOC(mo.bp.n)

    for i in range(0, mo.bp.n):
        mo.bp.b[i] = ARRAY_MALLOC(pow(mo, mo.M, mo.bp.order[i] + 1))

    for i in range(0, mo.bp.n):

        # find a state with the current order
        for j in range(0, mo.N):
            if mo.bp.order[i] == mo.order[j]:
                break

        # initialize with ones as psoudocounts
        size = pow(mo, mo.M, mo.bp.order[n] + 1)
        for m in range(0, size):
            mo.bp.b[i][m] = 1.0

        for n in range(0, sq.seq_number):

            for t in range(0, mo.bp.order[i]):
                mo.update_emission_history(sq.seq[n][t])

            for t in range(mo.bp.order[i], sq.seq_len[n]):

                e_index = mo.get_emission_index(j, sq.seq[n][t], t)
                if -1 != e_index:
                    mo.bp.b[i][e_index] += 1



        # normalise
        size = pow(mo, mo.M, mo.bp.order[n])
        for h in range(0, size, mo.M):
            for m in range(h, h + mo.M):
                sum_ += mo.bp.b[i][m]
            for m in range(h, h + mo.M):
                mo.bp.b[i][m] /= sum_


def ghmm_dmodel_distance(mo, m2):
    number = 0
    distance = 0.0

    # PI
    for i in range(mo.N):
        tmp = mo.s[i].pi - m2.s[i].pi
        distance += tmp * tmp
        number += 1

    for i in range(mo.N):
        # A
        for j in range(mo.N):
            tmp = mo.s[i].out_a[j] - m2.s[i].out_a[j]
            distance += tmp * tmp
            number += 1

        # B
        for j in range(pow(mo.M, mo.order[i] + 1)):
            tmp = mo.s[i].b[j] - m2.s[i].b[j]
            distance += tmp * tmp
            number += 1

    return distance / number


def ghmm_dmodel_xml_read(filename, mo_number):
    f = ghmm_xmlfile_parse(filename)
    if not f and (f.modelType & kDiscreteHMM and not (f.modelType & (kPairHMM | kTransitionClasses))):
        Log.error("wrong model type, model in file is not a plain discrete model")

    mo_number = f.noModels
    mo = f.model.d
    return mo


def ghmm_dmodel_xml_write(mo, file, mo_number):
    pass
