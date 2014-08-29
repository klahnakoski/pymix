## encoding: utf-8
################################################################################
#
#  This file is part of the Python Mixture Package
#
#  Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Library General Public
#  License as published by the Free Software Foundation; either
#  version 2 of the License, or (at your option) any later version.
#
#  This library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  Library General Public License for more details.
#
#  You should have received a copy of the GNU Library General Public
#  License along with this library; if not, write to the Free
#  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
################################################################################
from pymix import _C_mixextend

import numpy
import scipy


def get_normalized_posterior_matrix(data):
    logsum = numpy.log(numpy.sum(numpy.exp(data), axis=0))
    result = data - logsum
    return result, numpy.sum(logsum)


def substract_matrix(a, b):
    result = numpy.subtract(a, b)
    return result

def add_matrix(a, b):
    result = numpy.add(a, b)
    return result

def wrap_gsl_dirichlet_sample(alpha, n):
    result = scipy.random.dirichlet(alpha, n)
    return result

def set_gsl_rng_seed(seed):
    _C_mixextend.set_gsl_rng_seed(seed)




