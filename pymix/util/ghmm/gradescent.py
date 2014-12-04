# *****************************************************************************
#
#        This file is part of the General Hidden Markov Model Library,
#        GHMM version __VERSION__, see http:# ghmm.org
#
#        Filename: ghmm/ghmm/gradescent.c
#        Authors:  Janne Grunau, Alexander Riemer
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
#        This file is version $Revision: 2262 $
#                        from $Date: 2009-04-22 09:44:25 -0400 (Wed, 22 Apr 2009) $
#              last change by $Author: grunau $.
#
# *****************************************************************************
from pymix.util.ghmm.model import ghmm_dmodel_copy
from pymix.util.ghmm.reestimate import ighmm_reestimate_alloc_matvek
from pymix.util.ghmm.types import kTiedEmissions
from pymix.util.ghmm.wrapper import ARRAY_CALLOC, GHMM_EPS_PREC
from pymix.util.logs import Log


#  allocates memory for m and n matrices:
def gradient_descent_galloc(mo):
    matrix_b = ARRAY_CALLOC(mo.N)
    for i in range(mo.N):
        matrix_b[i] = ARRAY_CALLOC(pow(mo.M, mo.order[i] + 1))

    # matrix_a(i,j) = matrix_a[i*mo.N+j]
    matrix_a = ARRAY_CALLOC(mo.N * mo.N)

    # allocate memory for matrix_pi
    matrix_pi = ARRAY_CALLOC(mo.N)
    return matrix_b, matrix_a, matrix_pi


def ghmm_dmodel_label_gradient_expectations(mo, alpha, beta, scale, seq, seq_len, matrix_b, matrix_a, vec_pi):
    # initialise matrices with zeros
    for i in range(mo.N):
        for j in range(mo.N):
            matrix_a[i * mo.N + j] = 0
        size = pow(mo.M, mo.order[i] + 1)
        for h in range(size):
            matrix_b[i][h] = 0

    for t in range(seq_len):

        # sum products of forward and backward variables over all states:
        foba_sum = 0.0
        for i in range(mo.N):
            foba_sum += alpha[t][i] * beta[t][i]
        if GHMM_EPS_PREC > abs(foba_sum):
            Log.error("gradescent_compute_expect: foba_sum (%g) smaller as EPS_PREC (%g). t = %d.\n", foba_sum, GHMM_EPS_PREC, t)

        for i in range(mo.N):

            # compute gamma implicit
            gamma = alpha[t][i] * beta[t][i] / foba_sum

            # n_pi is easiest: n_pi(i) = gamma(0,i)
            if 0 == t:
                vec_pi[i] = gamma

            # n_b(i,c) = sum[t, hist(t)=c | gamma(t,i)] / sum[t | gamma(t,i)]
            e_index = mo.get_emission_index(i, seq[t], t)
            if -1 != e_index:
                matrix_b[i][e_index] += gamma


        # updating history, xi needs the right e_index for the next state
        mo.update_emission_history(seq[t])

        for i in range(mo.N):
            # n_a(i,j) = sum[t=0..T-2 | xi(t,i,j)] / sum[t=0..T-2 | gamma(t,i)]
            # compute xi only till the state before the last
            for j in range(mo.N):
                if t >= seq_len - 1:
                    break

                # compute xi implicit
                xi = 0
                e_index = mo.get_emission_index(j, seq[t + 1], t + 1)
                if e_index != -1:
                    xi = alpha[t][i] * beta[t + 1][j] * mo.s[i].out_a[j] * mo.s[j].b[e_index] / (scale[t + 1] * foba_sum)

                matrix_a[i * mo.N + j] += xi


def compute_performance(mo, sq):
    # log P[O | lambda, labeling] as computed by the forward algorithm
    # sum over log P (calculated by forward_label) for all sequences
    #     used to compute the performance of the training
    log_p_sum = 0.0

    # loop over all sequences
    for k in range(sq.seq_number):
        success = 0
        seq_len = sq.seq_len[k]

        log_p = mo.label_logp(sq.seq[k], sq.state_labels[k], seq_len)
        log_p_sum += log_p

        log_p = mo.logp(sq.seq[k], seq_len)
        log_p_sum -= log_p


    # return log_p_sum in success or +1.0 a probality of 0.0 on error
    return log_p_sum


#   Trains the model with a set of annotated sequences using gradient descent.
#   Model must not have silent states. (iteration)
#   @return            0/-1 success/error
#   @param mo:         pointer to a ghmm_dmodel
#   @param sq:         class of annotated sequences
#   @param eta:        training parameter for gradient descent
#
def gradient_descent_onestep(mo, sq, eta):
    # log P[O | lambda, labeling] as computed by the forward algorithm

    # allocate memory for the parameters used for reestimation
    m_b, m_a, m_pi = gradient_descent_galloc(mo)
    n_b, n_a, n_pi, = gradient_descent_galloc(mo)

    # loop over all sequences
    for k in range(sq.seq_number):
        seq_len = sq.seq_len[k]

        alpha, beta, scale = ighmm_reestimate_alloc_matvek(seq_len, mo.N)

        # calculate forward and backward variables without labels:
        log_p = mo.forward(sq.seq[k], seq_len, alpha, scale)
        mo.backward(sq.seq[k], seq_len, beta, scale)

        # compute n matrices (labels):
        ghmm_dmodel_label_gradient_expectations(mo, alpha, beta, scale, sq.seq[k], seq_len, m_b, m_a, m_pi)

        # calculate forward and backward variables with labels:
        log_p = mo.label_forward(sq.seq[k], sq.state_labels[k], seq_len, alpha, scale)

        log_p = mo.label_backward(sq.seq[k], sq.state_labels[k], seq_len, beta, scale)


        # compute m matrices (labels):
        ghmm_dmodel_label_gradient_expectations(mo, alpha, beta, scale, sq.seq[k], seq_len, m_b, m_a, m_pi)

        # reestimate model parameters:
        # PI
        pi_sum = 0
        #  update
        for i in range(mo.N):
            if mo.s[i].pi > 0.0:
                gradient = eta * (m_pi[i] - n_pi[i])
                if mo.s[i].pi + gradient > GHMM_EPS_PREC:
                    mo.s[i].pi += gradient
                else:
                    mo.s[i].pi = GHMM_EPS_PREC


            # sum over new PI vector
            pi_sum += mo.s[i].pi

        if pi_sum < GHMM_EPS_PREC:
            # never get here
            Log.error("Training ruined the model. You lose.\n")

        #  normalise
        for i in range(mo.N):
            mo.s[i].pi /= pi_sum

        # A
        for i in range(mo.N):
            a_row_sum = 0
            # update
            for j in range(mo.N):
                gradient = eta * (m_a[i * mo.N + j] - n_a[i * mo.N + j]) / (seq_len - 1)
                if mo.s[i].out_a[j] + gradient > GHMM_EPS_PREC:
                    mo.s[i].out_a[j] += gradient
                else:
                    mo.s[i].out_a[j] = GHMM_EPS_PREC

                # sum over rows of new A matrix
                a_row_sum += mo.s[i].out_a[j]

            if a_row_sum < GHMM_EPS_PREC:
                # never get here
                Log.error("Training ruined the model. You lose.\n")

            # normalise
            for j in range(mo.N):
                mo.s[i].out_a[j] /= a_row_sum
                mo.s[j].in_a[i] = mo.s[i].out_a[j]




        # B
        for i in range(mo.N):
            # don't update fix states
            if mo.s[i].fix:
                continue

            # update
            size = pow(mo.M, mo.order[i])
            for h in range(size):
                b_block_sum = 0
                for g in range(mo.M):
                    hist = h * mo.M + g
                    gradient = eta * (m_b[i][hist] - n_b[i][hist]) / seq_len
                    # printf("gradient[%d][%d] = %g, m_b = %g, n_b = %g\n"
                    #             , i, hist, gradient, m_b[i][hist], n_b[i][hist])
                    if gradient + mo.s[i].b[hist] > GHMM_EPS_PREC:
                        mo.s[i].b[hist] += gradient
                    else:
                        mo.s[i].b[hist] = GHMM_EPS_PREC

                    # sum over M-length blocks of new B matrix
                    b_block_sum += mo.s[i].b[hist]

                if b_block_sum < GHMM_EPS_PREC:
                    # never get here
                    Log.error("Training ruined the model. You lose.\n")

                # normalise
                for g in range(mo.M):
                    hist = h * mo.M + g
                    mo.s[i].b[hist] /= b_block_sum

        # restore "tied_to" property
        if mo.model_type & kTiedEmissions:
            mo.update_tie_groups()


#-
#
#   Trains the ghmm_dmodel with a set of annotated sequences till convergence using
#   gradient descent.
#   Model must not have silent states. (checked in Python wrapper)
#   @return            trained model/None pointer success/error
#   @param mo:         pointer to a ghmm_dmodel
#   @param sq:         class of annotated sequences
#   @param eta:        intial parameter eta (rate)
#   @param no_steps    number of training steps
#
def ghmm_dmodel_label_gradient_descent(mo, sq, eta, no_steps):
    runs = 0
    last = ghmm_dmodel_copy(mo)
    last_perf = compute_performance(last, sq)

    while eta > GHMM_EPS_PREC and runs < no_steps:
        runs += 1
        gradient_descent_onestep(mo, sq, eta)

        cur_perf = compute_performance(mo, sq)

        if last_perf < cur_perf:
            # if model is degenerated, lower eta and try again
            if cur_perf > 0.0:
                Log.note("current performance = %g", cur_perf)
                mo = last.copy()
                eta *= .5

            else:
                # Improvement insignificant, assume convergence
                if abs(last_perf - cur_perf) < cur_perf * (-1e-8):
                    Log.note("convergence after %d steps.", runs)
                    return 0

                if runs < 175 or 0 == runs % 50:
                    Log.note("Performance: %g\t improvement: %g\t step %d", cur_perf, cur_perf - last_perf, runs)

                # significant improvement, next iteration
                last = ghmm_dmodel_copy(mo)
                last_perf = cur_perf
                eta *= 1.07

        # no improvement
        else:
            if runs < 175 or 0 == runs % 50:
                Log.note("Performance: %g\t NOT improvment: %g\t step %d", cur_perf, cur_perf - last_perf, runs)


            # try another training step
            runs += 1
            eta *= .85
            gradient_descent_onestep(mo, sq, eta)

            cur_perf = compute_performance(mo, sq)
            Log.note("Performance: %g\t ?Improvement: %g\t step %d", cur_perf, cur_perf - last_perf, runs)

            # improvement, save and proceed with next iteration
            if last_perf < cur_perf and cur_perf < 0.0:
                last = mo.copy()
                last_perf = cur_perf

            # still no improvement, revert to saved model
            else:
                runs -= 1
                mo = last.copy()
                eta *= .9

    return mo
