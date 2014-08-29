## function sumlogs is borrowed from GQLMixture.py
import numpy
from pymix import _C_mixextend


def sumlogs_purepy(a):
    """ Given a Numeric.array a of log p_i, return log(sum p_i)

        Uses (assuming p_1 is maximal):
        log(\Sum p_i) = log(p_1) + log( 1 + \Sum_{i=2} exp(log(p_i) - log(p_1)))

        NOTE: The sumlogs functions returns the sum for values != -Inf

    """
    m = max(a) # Maximal value must not be unique
    result = 0.0
    #minus_infinity = -float(1E300)
    for x in a:
        if x >= m: # For every maximal value
            result += 1.0
        else:
            if x == float('-inf'): # zero probability, hence
                # -Inf log prob. Doesnt contribute
                continue
            x = x - m
            # Special case to avoid numerical problems
            if x < -1.0e-16: # <=> |x| >  1.0e-16
                result += numpy.exp(x)
            else: # |x| <  1.0e-16 => exp(x) = 1
                result += 1.0

    result = numpy.log(result)
    result += m
    return result


def sumlogs(a):
    """
    Call to C extension function sum_logs.
    """
    m = max(a) # Maximal value must not be unique
    result = _C_mixextend.sum_logs(a, m)
    return result


def matrixSumlogs(mat):
    """
    Call to C extension function matrix_sum_logs
    """
    return _C_mixextend.matrix_sum_logs(mat)


def dict_intersection(d1, d2):
    """
    Computes the intersections between the key sets of two Python dictionaries.
    Returns another dictionary with the intersection as keys.

    @param d1: dictionary object
    @param d2: dictionary object

    @return: dictionary with keys equal to the intersection of keys between d1 and d2.

    """
    int_dict = {}
    for e in d2:
        if d1.has_key(e):
            int_dict[e] = 1

    return int_dict

