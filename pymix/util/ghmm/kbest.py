# *****************************************************************************
#
#        This file is part of the General Hidden Markov Model Library,
#        GHMM version __VERSION__, see http:# ghmm.org
#
#        Filename: ghmm/ghmm/kbest.c
#        Authors:  Anyess von Bock, Alexander Riemer, Janne Grunau
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
#        This file is version $Revision: 2264 $
#                        from $Date: 2009-04-22 11:13:40 -0400 (Wed, 22 Apr 2009) $
#              last change by $Author: grunau $.
#
# *****************************************************************************
from math import exp
from pyLibrary.maths import Math
from pymix.util.ghmm.wrapper import ARRAY_MALLOC, ARRAY_CALLOC, ARRAY_REALLOC, ighmm_cvector_log_sum
from pymix.util.logs import Log

KBEST_THRESHOLD = -3.50655789732
#  Math.log(0.03) => threshold: 3% of most probable partial hypothesis
KBEST_EPS = 1E-15


#============================================================================

#  Data type for single linked list of hypotheses.
#
class hypo_List:
    def __init__(self):
        self.hyp_c = None                # < hypothesis
        self.refcount = 0             # < counter of the links to this hypothesis
        self.chosen = None
        self.gamma_states = None
        self.gamma_a = None
        self.gamma_id = None
        self.next = None  # < next list element
        self.parent = None # < parent hypothesis


#============================================================================
#
#  Builds logarithmic transition matrix from the states' in_a values
#  the row for each state is the logarithmic version of the state's in_a
#  @return transition matrix with logarithmic values, 1.0 if a[i,j] = 0
#  @param s:           array of all states of the model
#  @param N:           number of states in the model
#
def kbest_buildLogMatrix(s, N):
    # create & initialize matrix:
    log_a = ARRAY_MALLOC(N)
    for i in range(0, N):
        log_a[i] = ARRAY_MALLOC(N)
        for j in range(N):
            log_a[i][j] = Math.log(s[i].in_a[j])

    return log_a


#============================================================================
def ghmm_dmodel_label_kbest(mo, o_seq, seq_len, k):
    num_labels = 0

    # logarithmized transition matrix A, Math.log(a(i,j)) => log_a[i*N+j],
    #     1.0 for zero probability
    # double **log_a

    # matrix of hypotheses, holds for every position in the sequence a list
    #     of hypotheses
    # hypoList **h
    # hypoList *hP

    # vectors for rows in the matrices
    # int *hypothesis

    # pointer & prob. of the k most probable hypotheses for each state
    #     - matrices of dimensions #states x k:  argm(i,l) => argmaxs[i*k+l]
    # double *maxima
    # hypoList **argmaxs

    # pointer to & probability of most probable hypothesis in a certain state
    # hypoList *argmax
    # double sum

    # break if sequence empty or k<1:
    if seq_len <= 0 or k <= 0:
        return None

    h = [None for i in range(seq_len)]

    # 1. Initialization (extend empty hypothesis to #labels hypotheses of
    #         length 1):

    # get number of labels (= maximum label + 1)
    #     and number of states with those labels
    states_wlabel = ARRAY_CALLOC(mo.N)
    label_max_out = ARRAY_CALLOC(mo.N)
    for i in range(0, mo.N):
        c = mo.label[i]
        states_wlabel[c] += 1
        if c > num_labels:
            num_labels = c
        if mo.N > label_max_out[c]:
            label_max_out[c] = mo.N

    # add one to the maximum label to get the number of labels
    num_labels += 1
    states_wlabel = ARRAY_REALLOC(states_wlabel, num_labels)
    label_max_out = ARRAY_REALLOC(label_max_out, num_labels)

    # initialize h:
    hP = h[0]
    for i in range(0, mo.N):
        if mo.s[i].pi > KBEST_EPS:
            # printf("Found State %d with initial probability %f\n", i, mo.s[i].pi)
            exists = 0
            while hP != None:
                if hP.hyp_c == mo.label[i]:
                    # add entry to the gamma list
                    g_nr = hP.gamma_states
                    hP.gamma_id[g_nr] = i
                    hP.gamma_a[g_nr] = Math.log(mo.s[i].pi) + Math.log(mo.s[i].b[mo.get_emission_index(i, o_seq[0], 0)])
                    hP.gamma_states = g_nr + 1
                    exists = 1
                    break

                else:
                    hP = hP.next

            if not exists:
                h[0] = ighmm_hlist_insert(None, mo.label[i], None)
                # initiallize gamma-array with safe size (number of states) and add the first entry
                h[0].gamma_a = ARRAY_MALLOC(states_wlabel[mo.label[i]])
                h[0].gamma_id = ARRAY_MALLOC(states_wlabel[mo.label[i]])
                h[0].gamma_id[0] = i
                h[0].gamma_a[0] = Math.log(mo.s[i].pi) + Math.log(mo.s[i].b[mo.get_emission_index(i, o_seq[0], 0)])
                h[0].gamma_states = 1
                h[0].chosen = 1

            hP = h[0]


    # reallocating the gamma list to the real size
    hP = h[0]
    while hP != None:
        hP.gamma_a=ARRAY_REALLOC(hP.gamma_a, hP.gamma_states)
        hP.gamma_id=ARRAY_REALLOC(hP.gamma_id, hP.gamma_states)
        hP = hP.next


    # calculate transition matrix with logarithmic values:
    log_a = kbest_buildLogMatrix(mo.s, mo.N)

    # initialize temporary arrays:
    maxima = ARRAY_MALLOC(mo.N * k)                             # for each state save k
    argmaxs = ARRAY_MALLOC(mo.N * k)


    # Main loop: Cycle through the sequence:
    for t in range(1, seq_len):

        # put o_seq[t-1] in emission history:
        mo.update_emission_history(o_seq[t - 1])

        # 2. Propagate hypotheses forward and update gamma:
        no_oldHyps, h[t] = ighmm_hlist_prop_forward(mo, h[t - 1], h[t], num_labels, states_wlabel, label_max_out)
        # printf("t = %d (%d), no of old hypotheses = %d\n", t, seq_len, no_oldHyps)

        # calculate new gamma:
        hP = h[t]
        # cycle through list of hypotheses
        while hP != None:

            for i in range(0, hP.gamma_states):
                # if hypothesis hP ends with label of state i:
                #           gamma(i,c):= Math.log(sum(exp(a(j,i)*exp(oldgamma(j,old_c)))))
                #           + Math.log(b[i](o_seq[t]))
                #           else: gamma(i,c):= -INF (represented by 1.0)
                i_id = hP.gamma_id[i]
                hP.gamma_a[i] = ighmm_log_gamma_sum(log_a[i_id], mo.s[i_id], hP.parent)
                b_index = mo.get_emission_index(i_id, o_seq[t], t)
                if b_index < 0:
                    hP.gamma_a[i] = 1.0
                    if mo.order[i_id] > t:
                        continue
                    else:
                        Log.note("i_id: %d, o_seq[%d]=%d\ninvalid emission index!\n", i_id, t, o_seq[t])
                else:
                    try:
                        p = mo.s[i_id].b[b_index]
                        if p == 0.0:
                            hP.gamma_a[i] = -float("inf")
                        else:
                            hP.gamma_a[i] += Math.log(p)
                    except Exception, e:
                        Log.error("", e)
                    #printf("%g = %g\n", Math.log(mo.s[i_id].b[b_index]), hP.gamma_a[i])
                if hP.gamma_a[i] > 0.0:
                    Log.error("gamma too large. ghmm_dl_kbest failed\n")

            hP = hP.next

        # 3. Choose the k most probable hypotheses for each state and discard all
        #           hypotheses that were not chosen:

        # initialize temporary arrays:
        for i in range(0, mo.N * k):
            maxima[i] = 1.0
            argmaxs[i] = None

        # cycle through hypotheses & calculate the k most probable hypotheses for
        #       each state:
        # THIS IS MANAGING A SORTED LIST OF CANDIDATES, WITH LENGTH k
        hP = h[t]
        while hP != None:
            for i in range(0, hP.gamma_states):
                i_id = hP.gamma_id[i]
                if hP.gamma_a[i] > KBEST_EPS:
                    continue
                    # find first best hypothesis that is worse than current hypothesis:
                for l in range(k):
                    if maxima[i_id * k + l] >= KBEST_EPS or maxima[i_id * k + l] <= hP.gamma_a[i]:
                        break
                else:
                    l = k

                if l < k:
                    # for each m>l: m'th best hypothesis becomes (m+1)'th best
                    for m in reversed(range(l + 1, k)):
                        argmaxs[i_id * k + m] = argmaxs[i_id * k + m - 1]
                        maxima[i_id * k + m] = maxima[i_id * k + m - 1]

                    # save new l'th best hypothesis:
                    maxima[i_id * k + l] = hP.gamma_a[i]
                    argmaxs[i_id * k + l] = hP

            hP = hP.next

        # set 'chosen' for all hypotheses from argmaxs array:
        for i in range(0, mo.N * k):
            # only choose hypotheses whose prob. is at least threshold*max_prob
            if (maxima[i] != 1.0 and maxima[i] >= KBEST_THRESHOLD + maxima[(i % mo.N) * k]):
                argmaxs[i].chosen = 1

        # remove hypotheses that were not chosen from the lists:
        # remove all hypotheses till the first chosen one
        while h[t] != None and 0 == h[t].chosen:
            h[t] = ighmm_hlist_remove(h[t])
            # remove all other not chosen hypotheses
        if not h[t]:
            Log.error("No chosen hypothesis. ghmm_dl_kbest failed\n")

        hP = h[t]
        while hP.next != None:
            if 1 == hP.next.chosen:
                hP = hP.next
            else:
                hP.next = ighmm_hlist_remove(hP.next)


    # 4. Save the hypothesis with the highest probability over all states:
    hP = h[seq_len - 1]
    argmax = None
    log_p = 1.0                 # log_p will store log of maximum summed probability
    while hP != None:
        # sum probabilities for each hypothesis over all states:
        sum = ighmm_cvector_log_sum(hP.gamma_a, hP.gamma_states)
        # and select maximum sum
        if sum < KBEST_EPS and (log_p == 1.0 or sum > log_p):
            log_p = sum
            argmax = hP

        hP = hP.next


    # found a valid path?
    if log_p < KBEST_EPS:
        # yes: extract chosen hypothesis:
        hypothesis = ARRAY_MALLOC(seq_len)
        for i in reversed(range(seq_len)):
            hypothesis[i] = argmax.hyp_c
            argmax = argmax.parent
    else:
        # no: return 1.0 representing -INF and an empty hypothesis
        hypothesis = None

    # dispose of calculation matrices:
    hP = h[seq_len - 1]
    while hP != None:
        hP = ighmm_hlist_remove(hP)

    return hypothesis, log_p


#================ utility functions ========================================
# inserts new hypothesis into list at position indicated by pointer plist
def ighmm_hlist_insert(plist, newhyp, parlist):
    newlist = hypo_List()
    newlist.hyp_c = newhyp
    if parlist:
        parlist.refcount += 1
    newlist.parent = parlist
    newlist.next = plist

    return newlist


#============================================================================
# removes hypothesis at position indicated by pointer plist from the list
#   removes recursively parent hypothesis with refcount==0
def ighmm_hlist_remove(plist):
    tempPtr = plist.next

    if plist.parent:
        plist.parent.refcount -= 1
        if 0 == plist.parent.refcount:
            plist.parent = ighmm_hlist_remove(plist.parent)
    return tempPtr


#============================================================================
def ighmm_hlist_prop_forward(mo, h, hplus, labels, nr_s, max_out):
    no_oldHyps = 0
    newHyps = 0
    hP = h
    created = ARRAY_MALLOC(labels)

    # extend the all hypotheses with the labels of out_states
    #     of all states in the hypotesis
    while hP != None:

        # lookup table for labels, created[i]!=0 iff the current hypotheses
        #       was propagated forward with label i
        for c in range(0, labels):
            created[c] = None

        # extend the current hypothesis and add all states which may have
        #       probability greater null
        for i in range(0, hP.gamma_states):
            # skip impossible states
            if hP.gamma_a[i] == 1.0:
                continue
            i_id = hP.gamma_id[i]
            for j in range(0, mo.N):
                c = mo.label[j]

                # create a new hypothesis with label c
                if not created[c]:
                    hplus = ighmm_hlist_insert(hplus, c, hP)
                    created[c] = hplus
                    # initiallize gamma-array with safe size (number of states
                    hplus.gamma_id = ARRAY_MALLOC(min(nr_s[c], hP.gamma_states * max_out[hP.hyp_c]))
                    hplus.gamma_id[0] = j
                    hplus.gamma_states = 1
                    newHyps += 1

                # add a new gamma state to the existing hypothesis with c
                else:
                    g_nr = created[c].gamma_states
                    # search for state j_id in the gamma list
                    for k in range(0, g_nr):
                        if j == created[c].gamma_id[k]:
                            break
                        # add the state to the gamma list
                    else:
                        created[c].gamma_id[g_nr] = j
                        created[c].gamma_states = g_nr + 1

        # reallocating gamma-array to the correct size
        for c in range(0, labels):
            if created[c]:
                created[c].gamma_a = ARRAY_CALLOC(created[c].gamma_states)
                created[c].gamma_id = ARRAY_REALLOC(created[c].gamma_id, created[c].gamma_states)
                created[c] = None

        hP = hP.next
        no_oldHyps += 1

    return no_oldHyps, hplus


#============================================================================
#
#   Calculates the logarithm of sum(exp(log_a[j,a_pos])+exp(log_gamma[j,g_pos]))
#   which corresponds to the logarithm of the sum of a[j,a_pos]*gamma[j,g_pos]
#   @return ighmm_log_sum for products of a row from gamma and a row from matrix A
#   @param log_a:      row of the transition matrix with logarithmic values (1.0 for Math.log(0))
#   @param s:          ghmm_dstate whose gamma-value is calculated
#   @param parent:     a pointer to the parent hypothesis
#
def ighmm_log_gamma_sum(log_a, s, parent):
    max = 1.0
    argmax = 0

    # shortcut for the trivial case
    if parent.gamma_states == 1:
        return parent.gamma_a[0] + log_a[parent.gamma_id[0]]

    logP = ARRAY_MALLOC(len(s.in_a))

    # calculate logs of a[k,l]*gamma[k,hi] as sums of logs and find maximum:
    for j in range(len(s.in_a)):
        # search for state j_id in the gamma list
        for k in range(0, parent.gamma_states):
            if parent.gamma_id[k] == j:
                break
        if k == parent.gamma_states:
            logP[j] = 1.0
        else:
            logP[j] = log_a[j] + parent.gamma_a[k]
            if max == 1.0 or (logP[j] > max and logP[j] != 1.0):
                max = logP[j]
                argmax = j

    # calculate max+Math.log(1+sum[j!=argmax exp(logP[j]-max)])
    result = 1.0
    for j in range(len(s.in_a)):
        if j != argmax and logP[j] != 1.0:
            result += exp(logP[j] - max)

    result = Math.log(result)
    result += max
    return result
