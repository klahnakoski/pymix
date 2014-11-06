#******************************************************************************
#*
#*       This file is part of the General Hidden Markov Model Library,
#*       GHMM version __VERSION__, see http:# ghmm.org
#*
#*       Filename: ghmm/ghmm/sequence.c
#*       Authors:  Bernd Wichern, Andrea Weisse, Utz J. Pape, Benjamin Georgi
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
from math import ceil, exp, log
from pymix.util.ghmm.wrapper import ARRAY_CALLOC, GHMM_RNG_SET, RNG, GHMM_RNG_UNIFORM, ARRAY_REALLOC, GHMM_MAX_SEQ_NUMBER, DBL_MAX, ighmm_cmatrix_alloc, GHMM_PENALTY_LOGP
from pymix.util.logs import Log

kBlockAllocation = 1 << 0,
kHasLabels = 1 << 1,


def ghmm_dseq_read(filename, sq_number):
    pass


def ghmm_dseq_read_alloc(s):
    pass


def ghmm_cseq_read(filename, sqd_number):
    pass


def ghmm_cseq_read_alloc(s):
    pass

#============================================================================

# Truncate Sequences in a given sqd_field useful for Testing
#   returns truncated sqd_field
#   trunc_ratio 0: no truncation
#   trunc_ratio 1: truncation (mean truncation faktor = 0.5)
#   trunc_ratio -1: 100 % truncation
#

def ghmm_cseq_truncate(sqd_in, sqd_fields, trunc_ratio, seed):
#define CUR_PROC "ghmm_cseq_truncate"
    # Hack, use -1 for complete truncation
    if (0 > trunc_ratio or 1 < trunc_ratio) and trunc_ratio != -1:
        Log.error("Error: trunc_ratio not valid\n")

    sq = ARRAY_CALLOC(sqd_fields)

    GHMM_RNG_SET(RNG, seed)

    for i in range(0, sqd_fields):
        sq[i] = ghmm_cseq_calloc(sqd_in[i].seq_number)
        sq[i].total_w = sqd_in[i].total_w
        for j in range(0, sqd_in[i].seq_number):
            sq[i].seq[j] = ARRAY_CALLOC(sqd_in[i].seq_len[j])
            # length of truncated seq.
            if trunc_ratio == -1:
                trunc_len = 0
            else:
                trunc_len = ceil((sqd_in[i].seq_len[j] * (1 - trunc_ratio * GHMM_RNG_UNIFORM(RNG))))
            ghmm_cseq_copy(sq[i].seq[j], sqd_in[i].seq[j], trunc_len)
            ARRAY_REALLOC(sq[i].seq[j], trunc_len)
            sq[i].seq_len[j] = trunc_len
            sq[i].seq_id[j] = sqd_in[i].seq_id[j]
            sq[i].seq_w[j] = sqd_in[i].seq_w[j]
    return sq


def ghmm_cseq_calloc(seq_number):
    sqd = None

    if seq_number > GHMM_MAX_SEQ_NUMBER:
        Log.error("Number of sequences %ld exceeds possible range %d", seq_number, GHMM_MAX_SEQ_NUMBER)

    sqd = ARRAY_CALLOC(1)
    sqd.seq = ARRAY_CALLOC(seq_number)
    sqd.seq_len = ARRAY_CALLOC(seq_number)
    #ifdef GHMM_OBSOLETE
    sqd.seq_label = ARRAY_CALLOC(seq_number)
    #endif # GHMM_OBSOLETE
    sqd.seq_id = ARRAY_CALLOC(seq_number)
    sqd.seq_w = ARRAY_CALLOC(seq_number)
    sqd.seq_number = seq_number
    sqd.capacity = seq_number

    sqd.total_w = 0.0
    for i in range(0, seq_number):
        sqd.seq_id[i] = -1.0
        sqd.seq_w[i] = 1

    return sqd


def ghmm_dseq_calloc(seq_number):
    if seq_number > GHMM_MAX_SEQ_NUMBER:
        Log.error("Number of sequences %ld exceeds possible range %d", seq_number, GHMM_MAX_SEQ_NUMBER)

    sq = ARRAY_CALLOC(1)
    sq.seq = ARRAY_CALLOC(seq_number)
    #sq.states = ARRAY_CALLOC( seq_number)
    sq.seq_len = ARRAY_CALLOC(seq_number)
    #ifdef GHMM_OBSOLETE
    sq.seq_label = ARRAY_CALLOC(seq_number)
    #endif # GHMM_OBSOLETE
    sq.seq_id = ARRAY_CALLOC(seq_number)
    sq.seq_w = ARRAY_CALLOC(seq_number)
    sq.seq_number = seq_number
    sq.capacity = seq_number

    for i in range(seq_number):
    #ifdef GHMM_OBSOLETE
        sq.seq_label[i] = -1
        #endif # GHMM_OBSOLETE
        sq.seq_id[i] = -1.0
        sq.seq_w[i] = 1

    sq.state_labels = None
    sq.state_labels_len = None
    return sq


def ghmm_dseq_realloc(sq, seq_number):
    if seq_number > GHMM_MAX_SEQ_NUMBER:
        Log.error("Number of sequences %ld exceeds possible range", seq_number)

    ARRAY_REALLOC(sq.seq, seq_number)
    if sq.flags & kHasLabels and sq.states:
        ARRAY_REALLOC(sq.states, seq_number)
    ARRAY_REALLOC(sq.seq_len, seq_number)
    #ifdef GHMM_OBSOLETE
    ARRAY_REALLOC(sq.seq_label, seq_number)
    #endif # GHMM_OBSOLETE
    ARRAY_REALLOC(sq.seq_id, seq_number)
    ARRAY_REALLOC(sq.seq_w, seq_number)

    sq.capacity = seq_number

    return 0


def ghmm_dseq_calloc_state_labels(sq):
    sq.state_labels = ARRAY_CALLOC(sq.seq_number)
    sq.state_labels_len = ARRAY_CALLOC(sq.seq_number)


def ghmm_cseq_get_singlesequence(sq, index):
    res = ghmm_cseq_calloc(1)

    res.seq[0] = sq.seq[index]
    res.seq_len[0] = sq.seq_len[index]
    #ifdef GHMM_OBSOLETE
    res.seq_label[0] = sq.seq_label[index]
    #endif # GHMM_OBSOLETE
    res.seq_id[0] = sq.seq_id[index]
    res.seq_w[0] = sq.seq_w[index]
    res.total_w = res.seq_w[0]

    return res


def ghmm_dseq_get_singlesequence(sq, index):
    res = ghmm_dseq_calloc(1)
    res.seq[0] = sq.seq[index]
    res.seq_len[0] = sq.seq_len[index]
    #ifdef GHMM_OBSOLETE
    res.seq_label[0] = sq.seq_label[index]
    #endif # GHMM_OBSOLETE
    res.seq_id[0] = sq.seq_id[index]
    res.seq_w[0] = sq.seq_w[index]
    res.total_w = res.seq_w[0]

    if sq.state_labels:
        res.state_labels = ARRAY_CALLOC(1)
        res.state_labels_len = ARRAY_CALLOC(1)
        res.state_labels[0] = sq.state_labels[index]
        res.state_labels_len[0] = sq.state_labels_len[index]
    return res

#XXX TEST: frees everything but the seq field
def ghmm_dseq_subseq_free(sq):
    pass


def ghmm_cseq_subseq_free(sqd):
    pass


def ghmm_dseq_lexWords(n, M):
    cnt = 0
    j = n - 1
    if (n < 0) or (M <= 0):
        Log.error()

    seq_number = pow(M, n)
    sq = ghmm_dseq_calloc(seq_number)

    for i in range(0, seq_number):
        sq.seq[i] = ARRAY_CALLOC(n)
        sq.seq_len[i] = n
        sq.seq_id[i] = i

    seq = ARRAY_CALLOC(n)
    while not (j < 0):
        ghmm_dseq_copy(sq.seq[cnt], seq, n)
        j = n - 1
        while seq[j] == M - 1:
            seq[j] = 0
            j -= 1

        seq[j] += 1
        cnt += 1

    return sq


def ghmm_dseq_max_symbol(sq):
    max_symb = -1
    for i in range(0, sq.seq_number):
        for j in range(0, sq.seq_len[i]):
            if sq.seq[i][j] > max_symb:
                max_symb = sq.seq[i][j]

    return max_symb


def ghmm_dseq_copy(target, source, len):
    for i in range(0, len):
        target[i] = source[i]
        # ghmm_dseq_copy


def ghmm_cseq_copy(target, source, len):
    for i in range(0, len):
        target[i] = source[i]


def ghmm_dseq_add(target, source):
    res = -1
    old_seq = target.seq
    #int **old_seq_st    = target.states
    old_seq_len = target.seq_len
    #ifdef GHMM_OBSOLETE
    old_seq_label = target.seq_label
    #endif # GHMM_OBSOLETE
    old_seq_id = target.seq_id
    old_seq_w = target.seq_w
    old_seq_number = target.seq_number

    target.seq_number = old_seq_number + source.seq_number
    target.total_w += source.total_w

    target.seq = ARRAY_CALLOC(target.seq_number)
    #target.states = ARRAY_CALLOC( target.seq_number)
    target.seq_len = ARRAY_CALLOC(target.seq_number)
    #ifdef GHMM_OBSOLETE
    target.seq_label = ARRAY_CALLOC(target.seq_number)
    #endif # GHMM_OBSOLETE
    target.seq_id = ARRAY_CALLOC(target.seq_number)
    target.seq_w = ARRAY_CALLOC(target.seq_number)

    for i in range(0, old_seq_number):
        target.seq[i] = old_seq[i]
        #target.states[i] = old_seq_st[i]
        target.seq_len[i] = old_seq_len[i]
        #ifdef GHMM_OBSOLETE
        target.seq_label[i] = old_seq_label[i]
        #endif # GHMM_OBSOLETE
        target.seq_id[i] = old_seq_id[i]
        target.seq_w[i] = old_seq_w[i]

    for i in range(0, (target.seq_number - old_seq_number)):
        target.seq[i + old_seq_number] = ARRAY_CALLOC(source.seq_len[i])

        ghmm_dseq_copy(target.seq[i + old_seq_number], source.seq[i],
            source.seq_len[i])

        #target.states[i+old_seq_number] = ARRAY_CALLOC( source.seq_len[i])

        # ghmm_dseq_copy(target.states[i+old_seq_number], source.states[i],
        #       source.seq_len[i])

        target.seq_len[i + old_seq_number] = source.seq_len[i]
        #ifdef GHMM_OBSOLETE
        target.seq_label[i + old_seq_number] = source.seq_label[i]
        #endif # GHMM_OBSOLETE
        target.seq_id[i + old_seq_number] = source.seq_id[i]
        target.seq_w[i + old_seq_number] = source.seq_w[i]


def ghmm_cseq_add(target, source):
#define CUR_PROC "ghmm_cseq_add"

    old_seq = target.seq
    old_seq_len = target.seq_len
    #ifdef GHMM_OBSOLETE
    old_seq_label = target.seq_label
    #endif # GHMM_OBSOLETE
    old_seq_id = target.seq_id
    old_seq_w = target.seq_w
    old_seq_number = target.seq_number

    target.seq_number = old_seq_number + source.seq_number
    target.total_w += source.total_w

    target.seq = ARRAY_CALLOC(target.seq_number)
    target.seq_len = ARRAY_CALLOC(target.seq_number)
    #ifdef GHMM_OBSOLETE
    target.seq_label = ARRAY_CALLOC(target.seq_number)
    #endif # GHMM_OBSOLETE
    target.seq_id = ARRAY_CALLOC(target.seq_number)
    target.seq_w = ARRAY_CALLOC(target.seq_number)

    for i in range(0, old_seq_number):
        target.seq[i] = old_seq[i]
        target.seq_len[i] = old_seq_len[i]
        #ifdef GHMM_OBSOLETE
        target.seq_label[i] = old_seq_label[i]
        #endif # GHMM_OBSOLETE
        target.seq_id[i] = old_seq_id[i]
        target.seq_w[i] = old_seq_w[i]

    for i in range(0, (target.seq_number - old_seq_number)):
        target.seq[i + old_seq_number] = ARRAY_CALLOC(source.seq_len[i])

        ghmm_cseq_copy(target.seq[i + old_seq_number], source.seq[i],
            source.seq_len[i])
        target.seq_len[i + old_seq_number] = source.seq_len[i]
        #ifdef GHMM_OBSOLETE
        target.seq_label[i + old_seq_number] = source.seq_label[i]
        #endif # GHMM_OBSOLETE
        target.seq_id[i + old_seq_number] = source.seq_id[i]
        target.seq_w[i + old_seq_number] = source.seq_w[i]


def ghmm_dseq_check(sq, max_symb):
    for j in range(0, sq.seq_number):
        for i in range(0, sq.seq_len[j]):
            if (sq.seq[j][i] >= max_symb) or (sq.seq[j][i] < 0):
                Log.error("Wrong symbol \'%d\' in sequence %d at Pos. %d Should be within [0..%d]\n", sq.seq[j][i], j + 1, i + 1, max_symb - 1)


def ghmm_dseq_best_model(mo, model_number, sequence, seq_len, log_p):
# define CUR_PROC "seqence_best_model"
    log_p = -DBL_MAX
    model_index = -1
    for i in range(0, model_number):
        log_ptmp = mo[i].logp(sequence, seq_len)
        if log_ptmp != +1 and log_ptmp > log_p:
            log_p = log_ptmp
            model_index = i
    return model_index


def ghmm_dseq_print(sq, file):
    pass


def ghmm_cseq_print_xml(file, sq):
    pass


def ghmm_dseq_mathematica_print(sq, file, name):
    pass


def ghmm_cseq_gnu_print(sqd, file):
    pass


#============================================================================
def ghmm_cseq_print(sqd, file, discrete):
    pass


def ghmm_cseq_mathematica_print(sqd, file, name):
    pass


def ghmm_dseq_clean(sq):
    pass


def ghmm_cseq_clean(sqd):
    pass


def ghmm_dseq_free(sq):
    pass


def ghmm_cseq_free(sqd):
    pass


def ghmm_cseq_create_from_dseq(sq):
    sqd = ghmm_cseq_calloc(sq.seq_number)

    for j in range(0, sq.seq_number):
        sqd.seq[j] = ARRAY_CALLOC(sq.seq_len[j])
        for i in range(0, sq.seq_len[j]):
            sqd.seq[j][i] = sq.seq[j][i]
        sqd.seq_len[j] = sq.seq_len[j]
        #ifdef GHMM_OBSOLETE
        sqd.seq_label[j] = sq.seq_label[j]
        #endif # GHMM_OBSOLETE
        sqd.seq_id[j] = sq.seq_id[j]
        sqd.seq_w[j] = sq.seq_w[j]

    sqd.seq_number = sq.seq_number
    sqd.total_w = sq.total_w
    return sqd


def ghmm_dseq_create_from_cseq(sqd):
    sq = ghmm_dseq_calloc(sqd.seq_number)
    for j in range(0, sqd.seq_number):
        sq.seq[j] = ARRAY_CALLOC(sqd.seq_len[j])
        for i in range(0, sqd.seq_len[j]):
            sq.seq[j][i] = abs(sqd.seq[j][i])

        sq.seq_len[j] = sqd.seq_len[j]
        #ifdef GHMM_OBSOLETE
        sq.seq_label[j] = sqd.seq_label[j]
        #endif # GHMM_OBSOLETE
        sq.seq_id[j] = sqd.seq_id[j]
        sq.seq_w[j] = sqd.seq_w[j]

    sq.seq_number = sqd.seq_number
    sq.total_w = sqd.total_w
    return sq


def ghmm_dseq_max_len(sqd):
    return max(0, *sqd.seq_len)


def ghmm_cseq_max_len(sqd):
    return max(0, *sqd.seq_len)


def ghmm_cseq_mean(sqd):
    max_len = ghmm_cseq_max_len(sqd)
    out_sqd = ghmm_cseq_calloc(1)
    out_sqd.seq[0] = ARRAY_CALLOC(max_len)
    out_sqd.seq_len[0] = max_len

    for i in range(0, sqd.seq_number):
        for j in range(0, sqd.seq_len[i]):
            out_sqd.seq[0][j] += sqd.seq[i][j]

    for j in range(0, max_len):
        out_sqd.seq[0][j] /= sqd.seq_number

    return out_sqd


def ghmm_cseq_scatter_matrix(sqd, dim):
    dim = ghmm_cseq_max_len(sqd)
    W = ighmm_cmatrix_alloc(dim, dim)

    # Mean over all sequences. Individual counts for each dimension
    mean = ARRAY_CALLOC(dim)
    count = ARRAY_CALLOC(dim)
    for i in range(0, sqd.seq_number):
        for l in range(0, sqd.seq_len[i]):
            mean[l] += sqd.seq[i][l]
            count[l] += 1

    for l in range(0, dim):
        mean[l] /= count[l]
        # scatter matrix (triangle)
    for j in range(0, sqd.seq_number):
        for k in range(0, dim):
            for l in range(k, dim):
                if sqd.seq_len[j] > l:
                    W[k][l] += (sqd.seq[j][k] - mean[k]) * (sqd.seq[j][l] - mean[l])



    # norm with counts, set lower triangle
    for k in range(0, dim):
        for l in reversed(range(dim)):
            if l >= k:
                W[k][l] /= float(count[l])
            else:
                W[k][l] = W[l][k]

    return W


def ghmm_cseq_class(O, index, osum):
    return 0


# divide given field of seqs. randomly into two different fields. Also do
#   allocating. train_ratio determines approx. the fraction of seqs. that go
#   into the train_set and test_set resp.
#
def ghmm_cseq_partition(sqd, sqd_train, sqd_test, train_ratio):
    total_seqs = sqd.seq_number
    if total_seqs <= 0:
        Log.error("Error: number of seqs. less or equal zero\n")

    # waste of memory but avoids to many reallocations
    sqd_dummy = sqd_train
    for i in range(0, 2):
        sqd_dummy.seq = ARRAY_CALLOC(total_seqs)
        sqd_dummy.seq_len = ARRAY_CALLOC(total_seqs)
        #ifdef GHMM_OBSOLETE
        sqd_dummy.seq_label = ARRAY_CALLOC(total_seqs)
        #endif # GHMM_OBSOLETE
        sqd_dummy.seq_id = ARRAY_CALLOC(total_seqs)
        sqd_dummy.seq_w = ARRAY_CALLOC(total_seqs)
        sqd_dummy.seq_number = 0
        sqd_dummy = sqd_test

    for i in range(0, total_seqs):
        p = GHMM_RNG_UNIFORM(RNG)
        if p <= train_ratio:
            sqd_dummy = sqd_train
        else:
            sqd_dummy = sqd_test
        cur_number = sqd_dummy.seq_number
        sqd_dummy.seq[cur_number] = ARRAY_CALLOC(sqd.seq_len[i])
        # copy all entries
        ghmm_cseq_copy_all(sqd_dummy, cur_number, sqd, i)
        sqd_dummy.seq_number += 1


    # reallocs
    sqd_dummy = sqd_train
    for i in range(0, 2):
        ARRAY_REALLOC(sqd_dummy.seq, sqd_dummy.seq_number)
        ARRAY_REALLOC(sqd_dummy.seq_len, sqd_dummy.seq_number)
        #ifdef GHMM_OBSOLETE
        ARRAY_REALLOC(sqd_dummy.seq_label, sqd_dummy.seq_number)
        #endif # GHMM_OBSOLETE
        ARRAY_REALLOC(sqd_dummy.seq_id, sqd_dummy.seq_number)
        ARRAY_REALLOC(sqd_dummy.seq_w, sqd_dummy.seq_number)
        sqd_dummy = sqd_test


def ghmm_cseq_copy_all(target, t_num, source, s_num):
    ghmm_cseq_copy(target.seq[t_num], source.seq[s_num], source.seq_len[s_num])
    target.seq_len[t_num] = source.seq_len[s_num]
    #ifdef GHMM_OBSOLETE
    target.seq_label[t_num] = source.seq_label[s_num]
    #endif # GHMM_OBSOLETE
    target.seq_id[t_num] = source.seq_id[s_num]
    target.seq_w[t_num] = source.seq_w[s_num]


#============================================================================
# Likelihood function in a mixture model:
#   sum_k w^k log( sum_c (alpha_c p(O^k | lambda_c)))
#
def ghmm_cseq_mix_like(smo, smo_number, sqd, like):
    error_seqs = 0
    seq_like = 0.0
    like = 0.0

    for i in range(0, sqd.seq_number):
        seq_like = 0.0
        for k in range(0, smo_number):
            log_p = smo[k].logp(sqd.seq[i], sqd.seq_len[i])
            if log_p > -100:
                seq_like += exp(log_p) * smo[k].prior


        # no model fits
        if seq_like == 0.0:
            error_seqs += 1
            like += (GHMM_PENALTY_LOGP * sqd.seq_w[i])
        else:
            like += (log(seq_like) * sqd.seq_w[i])

    return error_seqs


def preproccess_alphabet(a):
    l = [None] * 128
    for i in range(a.size):
        if len(a.symbols[i]) == 1:
            index = a.symbols[i][0]
            if index < 128:
                l[index] = i
            else:
                Log.error("invalid alphabet for ghmm_dseq_open_fasta")
        else:
            Log.error("invalid alphabet for ghmm_dseq_open_fasta")
    return l


def get_internal(lookup, x):
    if x >= 0 and x < 128:
        return lookup[x]
    else:
        return -1


#============================================================================
def ghmm_dseq_open_fasta(filename, alphabet):
    pass

