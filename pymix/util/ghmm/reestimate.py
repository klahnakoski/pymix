from pymix.util.ghmm.wrapper import ARRAY_CALLOC, ighmm_cmatrix_stat_alloc
from pymix.util.logs import Log


def nologSum(vec, n):
    if n != len(vec):
        Log.error("Not expected")
    return sum(vec[0:n])

def ighmm_reestimate_alloc_matvek(T, N):
    alpha = ighmm_cmatrix_stat_alloc(T, N)
    beta = ighmm_cmatrix_stat_alloc(T, N)
    scale = ARRAY_CALLOC(T)
    return alpha, beta, scale

