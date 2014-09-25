from util.ghmm.wrapper import ARRAY_CALLOC, ighmm_cmatrix_stat_alloc


def nologSum(vec, len):
    return sum(vec[0:len])


def reestimate_free(r, N):
    pass



def ighmm_reestimate_alloc_matvek(T, N):
    alpha = ighmm_cmatrix_stat_alloc(T, N)
    beta = ighmm_cmatrix_stat_alloc(T, N)
    scale = ARRAY_CALLOC(T)
    return alpha, beta, scale


def ighmm_reestimate_free_matvek(alpha, beta, scale, T):
    pass
