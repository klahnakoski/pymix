#******************************************************************************
#*
#*       This file is part of the General Hidden Markov Model Library,
#*       GHMM version __VERSION__, see http:# ghmm.org
#*
#*       Filename: ghmm/ghmm/sdfoba.c
#*       Authors:  Utz J. Pape
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
#*       This file is version $Revision: 1713 $
#*                       from $Date: 2006-10-16 10:06:28 -0400 (Mon, 16 Oct 2006) $
#*             last change by $Author: grunau $.
#*
#******************************************************************************
from math import log
from pyLibrary.maths import Math
from pymix.util.ghmm.types import kSilentStates
from pymix.util.ghmm.wrapper import GHMM_EPS_PREC, ARRAY_CALLOC, ighmm_cmatrix_alloc
from pymix.util.logs import Log


EPS_PREC = 1E-8


def sdfoba_initforward(mo, alpha_1, symb, scale):
    clazz = 0
    scale[0] = 0.0
    #iterate over non-silent states
    for i in range(mo.N):
        if not (mo.silent[i]):
            if symb != mo.M:
                alpha_1[i] = mo.s[i].pi * mo.s[i].b[symb]
                #      printf("\nalpha1[%i]=%f\n",i,alpha_1[i])


            else:
                alpha_1[i] = mo.s[i].pi

            scale[0] += alpha_1[i]



    #iterate over silent states
    for i in range(mo.topo_order_length):
        id = mo.topo_order[i]
        alpha_1[id] = mo.s[id].pi
        #      printf("\nsilent_start alpha1[%i]=%f\n",id,alpha_1[id])
        for j in range(mo.N):
            alpha_1[id] += mo.s[id].in_a[clazz][j] * alpha_1[j]

        #      printf("\n\tsilent_run alpha1[%i]=%f\n",id,alpha_1[id])
        scale[0] += alpha_1[id]

    #  printf("\n%f\n",scale[0])
    if scale[0] >= GHMM_EPS_PREC:
        c_0 = 1 / scale[0]
        for i in range(mo.N):
            alpha_1[i] *= c_0

    return (0)                   # attention scale[0] might be 0


def sdfoba_stepforward(s, alpha_t, b_symb, clazz):
    value = 0.0

    for i in range(len(s.in_a)):
        value += s.in_a[clazz][i] * alpha_t[i]

    value *= b_symb
    return (value)


def ghmm_dsmodel_forward(mo, O, len, alpha, scale):
# define CUR_PROC "ghmm_dsmodel_forward"
    clazz = 0

    #if mo.model_type & kSilentStates:
    #     ghmm_dsmodel_topo_order(mo)
    #
    sdfoba_initforward(mo, alpha[0], O[0], scale)
    if scale[0] < GHMM_EPS_PREC:
        # means: first symbol can't be generated by hmm
        Log.note("\nnach init gestoppt\n")
        log_p = +1

    else:
        log_p = -Math.log(1 / scale[0])
        for t in range(1, len):
            scale[t] = 0.0
            #      printf("\nStep t=%i mit len=%i, O[i]=%i\n",t,len,O[t])
            if mo.cos > 1:
                clazz = mo.get_class(mo.N, t - 1)
                #iterate over non-silent states
            #printf("\nnach clazz\n")
            for i in range(mo.N):
                if not (mo.model_type & kSilentStates) or not (mo.silent[i]):
                    if O[t] != mo.M:
                        dblems = mo.s[i].b[O[t]]

                    else:
                        dblems = 1.0

                    alpha[t][i] = sdfoba_stepforward(mo.s[i], alpha[t - 1], dblems, clazz)

                    #           printf("alpha[%i][%i] = %f\n", t, i, alpha[t][i])
                    scale[t] += alpha[t][i]
                    #printf("\nalpha[%d][%d] = %e, scale[%d] = %e", t,i, alpha[t][i], t, scale[t])

                    #printf("scale[%i] = %f\n", t, scale[t])


            #printf("\nvor silent states\n")
            #iterate over silent state
            if mo.model_type & kSilentStates:
                for i in range(mo.topo_order_length):
                    #printf("\nget id\n")
                    id = mo.topo_order[i]
                    #printf("\nin stepforward\n")
                    alpha[t][id] = sdfoba_stepforward(mo.s[id], alpha[t], 1, clazz)
                    #   printf("alpha[%i][%i] = %f\n", t, id, alpha[t][id])
                    #printf("\nnach stepforward\n")
                    scale[t] += alpha[t][id]
                    #printf("\nalpha[%d][%d] = %e, scale[%d] = %e", t,id, alpha[t][id], t, scale[t])
                    #printf("silent state: %d\n", id)


            #printf("\nnach silent states\n")
            if scale[t] < GHMM_EPS_PREC:
                #char *str =
                Log.note("numerically questionable: ")
                #GHMM_LOG(LCONVERTED, str)
                #printf("\neps = %e\n", EPS_PREC)
                #printf("\nscale kleiner als eps HUHU, Zeit t = %d, scale = %e\n", t, scale[t])
                #printf("\nlabel: %s, char: %d, ems: %f\n", mo.s[92].label,O[t], mo.s[4].b[O[t]])
                #ighmm_cvector_print(stdout, alpha[t],mo.N, "\t", " ", "\n")
                # O-string  can't be generated by hmm
                #      *log_p = +1.0
                #break

            c_t = 1 / scale[t]
            for i in range(mo.N):
                alpha[t][i] *= c_t
                # sum_ Math.log(c[t]) to get  Math.log( P(O|lambda) )
            log_p -= Math.log(c_t)

    return log_p


def sdfobau_initforward(mo, alpha_1, symb, scale):
    clazz = 0
    scale[0] = 0.0
    #iterate over non-silent states
    for i in range(mo.N):
        if not (mo.silent[i]):
            alpha_1[i] = mo.s[i].pi * mo.s[i].b[symb]
            #printf("\nalpha1[%i]=%f\n",i,alpha_1[i])
            scale[0] += alpha_1[i]


    #iterate over silent states
    for i in range(mo.topo_order_length):
        id = mo.topo_order[i]
        alpha_1[id] = mo.s[id].pi
        for j in range(mo.N):
            alpha_1[id] += mo.s[id].in_a[clazz][j] * alpha_1[j]

    if scale[0] >= EPS_PREC:
        c_0 = 1 / scale[0]
        for i in range(mo.N):
            alpha_1[i] *= c_0


def sdfobau_forward(mo, O, len, alpha, scale):
    clazz = 0

    if mo.model_type & kSilentStates:
        mo.topo_order()

    sdfobau_initforward(mo, alpha[0], O[0], scale)
    if scale[0] < EPS_PREC:
        # means: first symbol can't be generated by hmm
        log_p = +1

    else:
        log_p = -Math.log(1 / scale[0])
        for t in range(1, len):
            scale[t] = 0.0
            if mo.cos > 1:
                clazz = mo.get_class(mo.N, t - 1)
                #iterate over non-silent states
            for i in range(mo.N):
                if not (mo.model_type & kSilentStates) or not (mo.silent[i]):
                    alpha[t][i] = sdfoba_stepforward(mo.s[i], alpha[t - 1], mo.s[i].b[O[t]], clazz)
                    scale[t] += alpha[t][i]


            #iterate over silent state
            if mo.model_type & kSilentStates:
                for i in range(mo.topo_order_length):
                    id = mo.topo_order[i]
                    alpha[t][id] = sdfoba_stepforward(mo.s[id], alpha[t], 1, clazz)
                    #scale[t] += alpha[t][id]

            if scale[t] < EPS_PREC:
                # O-string  can't be generated by hmm
                log_p = +1.0
                break

            c_t = 1 / scale[t]
            for i in range(mo.N):
                alpha[t][i] *= c_t
                # sum_ Math.log(c[t]) to get  Math.log( P(O|lambda) )
            log_p -= Math.log(c_t)
    return log_p


def ghmm_dsmodel_forward_descale(alpha, scale, t, n, newalpha):
    for i in range(t):
        for j in range(n):
            newalpha[i][j] = alpha[i][j]
            for k in range(i):
                newalpha[i][j] *= scale[k]


def ghmm_dsmodel_backward(mo, O, len, beta, scale):
    beta_tmp = ARRAY_CALLOC(mo.N)
    for t in range(len):
        if not scale[t]:
            Log.error("not allowed")
        # initialize
    for i in range(mo.N):
        beta[len - 1][i] = 1
        beta_tmp[i] = 1 / scale[len - 1]


    # Backward Step for t = T-2, ..., 0
    # beta_tmp: Vector for storage of scaled beta in one time step
    for t in reversed(range(len - 1)):
        for i in range(mo.N):
            sum_ = 0.0
            for j in range(mo.N):
                j_id = j
                #sum_ += mo.s[i].out_a[j] * mo.s[j_id].b[O[t+1]] * beta_tmp[j_id]

            beta[t][i] = sum_

        for i in range(mo.N):
            beta_tmp[i] = beta[t][i] / scale[t]


def ghmm_dsmodel_logp(mo, O, len, log_p):
    alpha = ighmm_cmatrix_alloc(len, mo.N)
    scale = ARRAY_CALLOC(len)
    # run ghmm_dsmodel_forward
    ghmm_dsmodel_forward(mo, O, len, alpha, scale, log_p)
