# *****************************************************************************
#
#        This file is part of the General Hidden Markov Model Library,
#        GHMM version __VERSION__, see http:# ghmm.org
#
#        Filename: ghmm/ghmm/sreestimate.c
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
#        This file is version $Revision: 2242 $
#                        from $Date: 2008-11-05 08:04:26 -0500 (Wed, 05 Nov 2008) $
#              last change by $Author: christoph_mpg $.
#
# *****************************************************************************
from math import sqrt
from pymix.distributions.normal_right import NormalRight
from pymix.util.ghmm.gauss_tail import ighmm_gtail_pmue_umin
from pymix.util.ghmm.matrixop import ighmm_determinant, ighmm_inverse
from pymix.util.ghmm.randvar import GHMM_EPS_NDT, ighmm_rand_get_xPHIless1, sqr
from pymix.util.ghmm.root_finder import ghmm_zbrent_AB
from pymix.util.ghmm.types import kMultivariate
from pymix.util.ghmm.wrapper import ARRAY_CALLOC, ighmm_cmatrix_stat_alloc, DBL_MIN, GHMM_EPS_PREC, DBL_MAX, matrix_alloc
from pymix.util.logs import Log
from pyLibrary.maths import Math

MCI = 0
GHMM_MAX_ITER_BW = 500
GHMM_EPS_ITER_BW = 0.0001
GHMM_EPS_U = 1E-4

C_PHI = None
CC_PHI = None


class local_store_t():
    def __init__(self):
        self.cos = None  # int
        self.pi_num = None  # double *
        self.pi_denom = None  # double
        self.a_num = None  # double ***
        self.a_denom = None  # double **
        self.c_num = None  # double **
        self.c_denom = None  # double *
        self.mue_num = None  # double ***
        self.u_num = None  # double ***
        self.mue_u_denom = None  # double **       # mue-denom. = u-denom. for sym. normal density
        self.sum_gt_otot = None  # double **       # for truncated normal density
        self.t = None  # local_store_


def sreestimate_alloc(smo):
    r = local_store_t()
    r.cos = smo.cos
    r.pi_num = ARRAY_CALLOC(smo.N)
    r.a_num = ARRAY_CALLOC(smo.N)
    for i in range(smo.N):
        r.a_num[i] = ighmm_cmatrix_stat_alloc(smo.cos, smo.N)

    r.a_denom = ighmm_cmatrix_stat_alloc(smo.N, smo.cos)

    # For sacke of simplicity, a NXmax(M) is being allocated,
    #  even though not all emissins components are of size max(M)
    r.c_denom = ARRAY_CALLOC(smo.N)
    r.c_num = ighmm_cmatrix_stat_alloc(smo.N, smo.M)
    r.mue_num = ARRAY_CALLOC(smo.N)

    for i in range(smo.N):
        r.mue_num[i] = ighmm_cmatrix_stat_alloc(smo.s[i].M, smo.dim)

    r.u_num = ARRAY_CALLOC(smo.N)
    for i in range(smo.N):
        r.u_num[i] = ARRAY_CALLOC(smo.s[i].M)
        for j in range(smo.s[i].M):
            r.u_num[i][j] = matrix_alloc(smo.dim, smo.dim)

    r.mue_u_denom = ighmm_cmatrix_stat_alloc(smo.N, smo.M)
    r.sum_gt_otot = ighmm_cmatrix_stat_alloc(smo.N, smo.M)
    return r

def sreestimate_init(r, smo):
    r.pi_denom = 0.0
    for i in range(smo.N):
        r.pi_num[i] = 0.0
        for osc in range(smo.cos):
            r.a_denom[i][osc] = 0.0
            for j in range(smo.N):
                r.a_num[i][osc][j] = 0.0

        r.c_denom[i] = 0.0
        for m in range(smo.s[i].M):
            for j in range(smo.dim):
                r.mue_num[i][m][j] = 0.0

            for j in range(smo.dim):
                for k in range(smo.dim):
                    r.u_num[i][m][j][k] = 0.0

            r.c_num[i][m] = 0.0
            r.mue_u_denom[i][m] = 0.0
            r.sum_gt_otot[i][m] = 0.0


def sreestimate_alloc_matvek(T, N, M):
    alpha = ighmm_cmatrix_stat_alloc(T, N)
    beta = ighmm_cmatrix_stat_alloc(T, N)

    scale = ARRAY_CALLOC(T)
    # 3-dim. matrix for b[t][i][m] with m = 1..M(not ):
    b = ARRAY_CALLOC(T)
    for t in range(T):
        b[t] = ighmm_cmatrix_stat_alloc(N, M + 1)

    return alpha, beta, scale, b


def sreestimate_precompute_b(smo, O, T, b):
# define CUR_PROC "sreestimate_precompute_b"
    # save sum (c_im * b_im(O_t))  in b[t][i][smo.M]
    for t in range(T):
        for i in range(smo.N):
            b[t][i][smo.M] = 0.0
        # save c_im * b_im(O_t)  directly in  b[t][i][m]
    for t in range(T):
        for i in range(smo.N):
            for m in range(smo.s[i].M):
                b[t][i][m] = smo.s[i].calc_cmbm(m, O[t])
                b[t][i][smo.s[i].M] += b[t][i][m]


def sreestimate_setlambda(r, smo):
    a_factor_i = 0.0
    c_factor_i = 0.0
    if r.pi_denom <= DBL_MIN:
        Log.error("pi: denominator == 0.0not \n")
    pi_factor = 1 / r.pi_denom

    for i in range(smo.N):
        # Pi
        smo.s[i].pi = r.pi_num[i] * pi_factor

        # A
        for osc in range(smo.cos):
            # note: denom. might be 0 never reached state?
            a_denom_pos = 1

            if r.a_denom[i][osc] <= DBL_MIN:
                a_denom_pos = 0
            else:
                a_factor_i = 1 / r.a_denom[i][osc]

            a_num_pos = 0

            for j in range(smo.N):
                # TEST: denom. < numerator
                if (r.a_denom[i][osc] - r.a_num[i][osc][j]) < -GHMM_EPS_PREC:
                    smo.s[i].out_a[osc][j] = 1.0
                elif a_denom_pos:
                    smo.s[i].out_a[osc][j] = r.a_num[i][osc][j] * a_factor_i
                else:
                    continue
                if r.a_num[i][osc][j] > 0.0:
                    a_num_pos = 1

                smo.s[j].in_a[osc][i] = smo.s[i].out_a[osc][j]

        # if fix, continue to next state
        if smo.s[i].fix:
            continue

        # C, Mue und U


        c_denom_pos = 1
        if r.c_denom[i] <= DBL_MIN:     # < EPS_PREC ?
            c_denom_pos = 0

        else:
            c_factor_i = 1 / r.c_denom[i]

        c_num_pos = 0
        fix_w = 1.0
        unfix_w = 0.0
        fix_flag = 0

        for m in range(smo.s[i].M):
            # if fixed continue to next component
            if smo.s[i].e[m].fixed:
                #printf("state %d, component %d is fixed not \n",i,m)
                fix_w = fix_w - smo.s[i].c[m]
                fix_flag = 1           # we have to normalize weights . fix flag is set to one
                continue


            # TEST: denom. < numerator
            if (r.c_denom[i] - r.c_num[i][m]) < 0.0:     # < -EPS_PREC ?
                smo.s[i].c[m] = 1.0
            elif c_denom_pos:
                # c_denom == 0: no change in c_im (?)
                smo.s[i].c[m] = r.c_num[i][m] * c_factor_i

            if r.c_num[i][m] > 0.0:
                c_num_pos = 1

            unfix_w = unfix_w + smo.s[i].c[m]

            if abs(r.mue_u_denom[i][m]) <= DBL_MIN:
            # < EPS_PREC ?
                pass
            else:
                # set mue_im
                if smo.model_type & kMultivariate:
                    for d in range(smo.s[i].e[m].dimension):
                        smo.s[i].e[m].mean[d] = r.mue_num[i][m][d] / r.mue_u_denom[i][m]
                else:
                    smo.s[i].e[m].mean = r.mue_num[i][m][0] / r.mue_u_denom[i][m]

            # TEST: u_denom == 0.0 ?
            if abs(r.mue_u_denom[i][m]) <= DBL_MIN:     # < EPS_PREC ?
                pass
                # smo.s[i].u[m]  unchangednot

            else:
                if smo.model_type & kMultivariate:
                    for d1 in range(smo.s[i].e[m].dimension):
                        for d2 in range(smo.s[i].e[m].dimension):
                            u_im = r.u_num[i][m][d1][d2] / r.mue_u_denom[i][m]
                            if abs(u_im) <= GHMM_EPS_U:
                                if u_im < 0:
                                    u_im = -1.0 * GHMM_EPS_U
                                else:
                                    u_im = GHMM_EPS_U

                            smo.s[i].e[m].variance[d1][d2] = u_im

                    # update the inverse and the determinant of covariance matrix
                    smo.s[i].e[m].sigma_det = ighmm_determinant(smo.s[i].e[m].variance)
                    smo.s[i].e[m].sigma_inv = ighmm_inverse(smo.s[i].e[m].variance)

                else:
                    u_im = r.u_num[i][m][0][0] / r.mue_u_denom[i][m]
                    if u_im <= GHMM_EPS_U:
                        u_im = GHMM_EPS_U
                    smo.s[i].e[m].variance = u_im



            # modification for truncated normal density:
            #         1-dim optimization for mue, calculate u directly
            #         note: if denom == 0 -. mue and u not recalculated above
            if isinstance(smo.s[i].e[m], NormalRight) and abs(r.mue_u_denom[i][m]) > DBL_MIN:
                A = smo.s[i].e[m].mean
                B = r.sum_gt_otot[i][m] / r.mue_u_denom[i][m]

                # A^2 ~ B . search max at border of EPS_U
                if B - A * A < GHMM_EPS_U:
                    mue_left = -GHMM_EPS_NDT  # attention: only works if  EPS_NDT > EPS_U not
                    mue_right = A

                    if (ighmm_gtail_pmue_umin(mue_left, A, B, GHMM_EPS_NDT) > 0.0 and ighmm_gtail_pmue_umin(mue_right, A, B, GHMM_EPS_NDT) > 0.0) or (
                                ighmm_gtail_pmue_umin(mue_left, A, B, GHMM_EPS_NDT) < 0.0 and ighmm_gtail_pmue_umin(mue_right, A, B, GHMM_EPS_NDT) < 0.0):
                        Log.note(
                            "umin:fl:%.3f\tfr:%.3f\t left %.3f\t right %3f\t A %.3f\t B %.3f\n",
                            ighmm_gtail_pmue_umin(mue_left, A, B, GHMM_EPS_NDT),
                            ighmm_gtail_pmue_umin(mue_right, A, B, GHMM_EPS_NDT), mue_left, mue_right, A, B
                        )

                    mue_im = ghmm_zbrent_AB(ighmm_gtail_pmue_umin, mue_left, mue_right, ACC, A, B, GHMM_EPS_NDT)
                    u_im = GHMM_EPS_U

                else:
                    Atil = A + GHMM_EPS_NDT
                    Btil = B + GHMM_EPS_NDT * A
                    mue_left = (-C_PHI * sqrt(Btil + GHMM_EPS_NDT * Atil + CC_PHI * sqrt(Atil) / 4.0) - CC_PHI * Atil / 2.0 - GHMM_EPS_NDT) * 0.99
                    mue_right = A
                    if A < Btil *  NormalRight(-GHMM_EPS_NDT, 0, -GHMM_EPS_NDT).linear_pdf(Btil):
                        mue_right = min(GHMM_EPS_NDT, mue_right)
                    else:
                        mue_left = max(-GHMM_EPS_NDT, mue_left)
                    mue_im = ghmm_zbrent_AB(ighmm_gtail_pmue_interpol, mue_left, mue_right, ACC, A, B, GHMM_EPS_NDT)
                    u_im = Btil - mue_im * Atil

                # set modified values of mue and u
                smo.s[i].e[m].mean = mue_im
                if u_im < GHMM_EPS_U:
                    u_im = GHMM_EPS_U
                smo.s[i].e[m].variance = u_im
                # end modifikation truncated density

                # for (m ..)
                # adjusting weights for fixed mixture components if necessary
        if fix_flag == 1:
            for m in range(smo.s[i].M):
                if smo.s[i].e[m].fixed == 0:
                    smo.s[i].c[m] = (smo.s[i].c[m] * fix_w) / unfix_w
                    # for (i = 0 .. < smo.N)


def sreestimate_one_step(smo, r, seq_number, T, O, seq_w):
    T_k = 0
    T_k_max = 0
    log_p = 0.0
    valid_parameter = valid_logp = 0

    #scan for max T_k: alloc of alpha, beta, scale and b only once
    T_k_max = Math.max(*T)
    alpha, beta, scale, b = sreestimate_alloc_matvek(T_k_max, smo.N, smo.M)

    # loop over all sequences
    for k in range(seq_number):
        # Test: ignore sequences with very small weights
        # if seq_w[k] < 0.0001:
        #       continue
        #
        # seq. is used for calculation of log_p
        valid_logp += 1
        T_k = T[k] / smo.dim
        # precompute output densities
        sreestimate_precompute_b(smo, O[k], T_k, b)

        if smo.cos > 1:
            smo.class_change.k = k

        log_p_k = smo.forward(O[k], T[k], b, alpha, scale)
        smo.backward(O[k], T[k], b, beta, scale)

        # printf("\n\nalpha:\n")
        #    ighmm_cmatrix_print(stdout,alpha,T_k,smo.N,"\t", ",", "")
        #    printf("\n\nbeta:\n")
        #    ighmm_cmatrix_print(stdout,beta,T_k,smo.N,"\t", ",", "")
        #    printf("\n\n")

        # weighted error function
        log_p += log_p_k * seq_w[k]
        # seq. is used for parameter estimation
        valid_parameter += 1

        # loop over all states
        for i in range(smo.N):

            state = smo.s[i]

            # Pi
            r.pi_num[i] += seq_w[k] * alpha[0][i] * beta[0][i]
            r.pi_denom += seq_w[k] * alpha[0][i] * beta[0][i]       # sum over all i

            # loop over t (time steps of seq.)
            for t in range(T_k):
                c_t = 1 / scale[t]
                if t > 0:
                    if smo.cos == 1:
                        osc = 0
                    else:
                        if not smo.class_change.get_class:
                            Log.error("get_class not initialized")

                        osc = smo.class_change.get_class(smo, O[k], k, t - 1)
                        #printf("osc=%d : cos = %d, k = %d, t = %d, state=%d\n",osc,smo.cos,smo.class_change.k,t,i)
                        if osc >= smo.cos:
                            Log.error("get_class returned index %d "
                                      "but model has only %d classesnot ", osc, smo.cos)

                    # A: starts at t=1 not !not
                    for j in range(smo.N):
                        contrib_t = (seq_w[k] * alpha[t - 1][i] * state.out_a[osc][j] * b[t][j][state.M] * beta[t][j] * c_t)

                        r.a_num[i][osc][j] += contrib_t
                        r.a_denom[i][osc] += contrib_t

                    # calculate sum (j=1..N):alp[t-1][j]*a_jc(t-1)i
                    sum_alpha_a_ji = 0.0
                    for j in range(smo.N):
                        sum_alpha_a_ji += alpha[t - 1][j] * state.in_a[osc][j]

                else:
                    # calculate sum(j=1..N):alpha[t-1][j]*a_jci, which is used below
                    #             for (t=1) = pi[i] (alpha[-1][i] not defined) not !not
                    sum_alpha_a_ji = state.pi
                    # if t>0
                # ========= if state fix, continue======================
                if state.fix:
                    continue
                    # C-denominator:
                r.c_denom[i] += seq_w[k] * alpha[t][i] * beta[t][i]

                # if sum_alpha_a_ji == 0.0, all following values are 0not
                if sum_alpha_a_ji == 0.0:
                    continue             # next t

                # loop over no of density functions for C-numer., mue and u
                for m in range(state.M):
                    #  c_im * b_im
                    f_im = b[t][i][m]
                    gamma = seq_w[k] * sum_alpha_a_ji * f_im * beta[t][i]
                    gamma_ct = gamma * c_t       # c[t] = 1/scale[t]

                    # numerator C:
                    r.c_num[i][m] += gamma_ct

                    # numerator Mue:
                    if smo.model_type & kMultivariate:
                        for d in range(state.e[m].dimension):
                            r.mue_num[i][m][d] += (gamma_ct * O[k][t][d])
                    else:
                        r.mue_num[i][m][0] += gamma_ct * O[k][t]

                    # denom. Mue/U:
                    r.mue_u_denom[i][m] += gamma_ct
                    # numerator U:
                    if smo.model_type & kMultivariate:
                        for di in range(state.e[m].dimension):
                            for dj in range(state.e[m].dimension):
                                r.u_num[i][m][di][dj] += (gamma_ct * (O[k][t][di] - state.e[m].mean[di]) * (O[k][t][dj] - state.e[m].mean[dj]))
                                r.sum_gt_otot[i][m] += (gamma_ct * O[k][t][di] * O[k][t][dj])
                    else:
                        r.u_num[i][m][0][0] += (gamma_ct * sqr(O[k][t] - state.e[m].mean))
                        r.sum_gt_otot[i][m] += (gamma_ct * sqr(O[k][t]))

                    # sum gamma_ct * O[k][t] * O[k][t] (truncated normal density):
    if smo.cos > 1:
        smo.class_change.k = -1

    if valid_parameter:
        sreestimate_setlambda(r, smo)
        smo.check()
    else:                        # NO sequence can be build from smodel smonot
        # diskret:  *log_p = +1
        Log.error(" NO sequence can be build from smodel smonot \n")

    return valid_logp, log_p

#============================================================================
# int ghmm_cmodel_baum_welch(smo, sqd) :
def ghmm_cmodel_baum_welch(cs):
    # truncated normal density needs  varialbles C_PHI and
    #     CC_PHI
    for i in range(cs.smo.N):
        for j in range(cs.smo.s[i].M):
            if isinstance(cs.smo.s[i].e[j], NormalRight):
                globals()["C_PHI"] = ighmm_rand_get_xPHIless1()
                globals()["CC_PHI"] = sqr(C_PHI)
                break

    # local store for all iterations
    r = sreestimate_alloc(cs.smo)
    sreestimate_init(r, cs.smo)

    log_p_old = -DBL_MAX
    valid_old = cs.sqd.seq_number
    n = 1

    max_iter_bw = min(GHMM_MAX_ITER_BW, cs.max_iter)
    eps_iter_bw = max(GHMM_EPS_ITER_BW, cs.eps)

    while n <= max_iter_bw:
        valid, log_p = sreestimate_one_step(cs.smo, r, cs.sqd.seq_number, cs.sqd.seq_len, cs.sqd.seq, cs.sqd.seq_w)
        # to follow convergence of bw: uncomment next line
        # Log.note("\tBW Iter %d\t Math.log(p) %.4f", n, log_p)
        diff = log_p - log_p_old

        if diff < -GHMM_EPS_PREC:
            if valid > valid_old:
                Log.note("log P < log P-old (more sequences (%d), n = %d)", valid - valid_old, n)

            # no convergence
            else:
                Log.error("NO convergence: log P(%e) < log P-old(%e)not  (n = %d)", log_p, log_p_old, n)



        # stop iteration
        if diff >= 0.0 and diff < abs(log_p):
            break

        else:
            # for next iteration
            valid_old = valid
            log_p_old = log_p
            # set values to zero
            sreestimate_init(r, cs.smo)
            n += 1


            # while n <= MAX_ITER_BW:
            # log_p outside this function
    cs.logp = log_p

