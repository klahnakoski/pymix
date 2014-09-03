################################################################################
#
#       This file is part of the Modified Python Mixture Package, original
#       source code is from https://svn.code.sf.net/p/pymix/code.  Also see
#       http://www.pymix.org/pymix/index.php?n=PyMix.Download
#
#       Changes made by: Kyle Lahnakoski (kyle@lahnakoski.com)
#
################################################################################
#
#       This file is part of the Python Mixture Package
#
#       file:    mixture.py
#       author: Benjamin Georgi
#
#       Copyright (C) 2004-2009 Benjamin Georgi
#       Copyright (C) 2004-2009 Max-Planck-Institut fuer Molekulare Genetik,
#                               Berlin
#
#       Contact: georgi@molgen.mpg.de
#
#       This library is free software; you can redistribute it and/or
#       modify it under the terms of the GNU Library General Public
#       License as published by the Free Software Foundation; either
#       version 2 of the License, or (at your option) any later version.
#
#       This library is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#       Library General Public License for more details.
#
#       You should have received a copy of the GNU Library General Public
#       License along with this library; if not, write to the Free
#       Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
################################################################################

## function sumlogs is borrowed from GQLMixture.py

import numpy as np


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
                result += np.exp(x)
            else: # |x| <  1.0e-16 => exp(x) = 1
                result += 1.0

    result = np.log(result)
    result += m
    return result


def sum_logs(a):
    m = np.max(a)  # Maximal value must not be unique
    result = np.log(sum(np.exp(a - m))) + m
    return result


def matrix_sum_logs(a):
    m = np.max(a)  # Maximal value must not be unique
    result = np.log(sum(np.exp(a - m))) + m
    return result


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

