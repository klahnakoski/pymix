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

import math
import random
import numpy as np
from numpy import linalg as la
from .prob import ProbDistribution
from pyLibrary.env.logs import Log
from pymix.util.ghmm.matrixop import ighmm_determinant, ighmm_inverse
from ..util.dataset import DataSet


class MultiNormalDistribution(ProbDistribution):
    """
    Multivariate Normal Distribution

    """

    def __init__(self, mu, sigma):
        """
        Constructor

        @param p: dimensionality of the distribution
        @param mu: mean parameter vector
        @param sigma: covariance matrix
        """

        assert len(mu) == len(sigma) == len(sigma[0]), str(len(mu)) + ' == ' + str(len(sigma)) + ' == ' + str(len(sigma[0]))
        self.mean = np.array(mu, dtype='Float64')
        self.variance = np.array(sigma, dtype='Float64')
        self.variance_det = ighmm_determinant(self.variance, len(sigma))
        self.variance_inv = ighmm_inverse(self.variance, len(sigma))
        self.fixed = 0  #allow parameter update


    @property
    def dimension(self):
        """
        @return: Number of dimensions
        """
        return len(self.mean)

    @property
    def freeParams(self):
        return self.dimension + self.dimension ** 2

    @property
    def suff_p(self):
        return self.dimension

    def __copy__(self):
        return MultiNormalDistribution(self.mean, self.variance)


    def __str__(self):
        return "Normal:  [" + str(self.mean) + ", " + str(self.variance.tolist()) + "]"

    def __eq__(self, other):
        if not isinstance(other, MultiNormalDistribution):
            return False
        if self.dimension != other.dimension:
            return False
        if not np.allclose(self.mean, other.mean) or not np.allclose(self.variance, other.variance):
            return False
        return True

    def pdf(self, data):
        if isinstance(data, DataSet):
            x = data.internalData
        elif hasattr(data, "__iter__"):
            x = data
        else:
            raise TypeError, "Unknown/Invalid input type."

        ff = math.pow(2 * math.pi, -self.dimension / 2.0) * math.pow(self.variance_det, -0.5);

        # centered input values
        centered = np.subtract(x, np.repeat([self.mean], len(x), axis=0))

        res = ff * np.exp(-0.5 * np.sum(np.multiply(centered, np.dot(centered, self.variance_inv)), 1))

        return np.log(res)


    def linear_pdf(self, x):
        ff = math.pow(2 * math.pi, -self.dimension / 2.0) * math.pow(self.variance_det, -0.5)
        centered = x-self.mean
        res = ff * np.exp(-0.5 * np.sum(np.multiply(centered, np.dot(centered, self.variance_inv)), 1))

        return res


    def MStep(self, posterior, data, mix_pi=None):

        if isinstance(data, DataSet):
            x = data.internalData
        elif hasattr(data, "__iter__"):
            x = data
        else:
            raise TypeError, "Unknown/Invalid input to MStep."

        post = posterior.sum() # sum of posteriors
        self.mean = np.dot(posterior, x) / post

        # centered input values (with new mus)
        centered = np.subtract(x, np.repeat([self.mean], len(x), axis=0))
        self.variance = np.dot(np.transpose(np.dot(np.identity(len(posterior)) * posterior, centered)), centered) / post


    def sample(self, A=None):
        """
        Samples from the mulitvariate Normal distribution.

        @param A: optional Cholesky decomposition of the covariance matrix self.variance, can speed up
        the sampling
        """
        if A == None:
            A = la.cholesky(self.variance)

        z = np.zeros(self.dimension, dtype='Float64')
        for i in range(self.dimension):
            z[i] = random.normalvariate(0.0, 1.0)  # sample p iid N(0,1) RVs

        X = np.dot(A, z) + self.mean
        return X.tolist()  # return value of sample must be Python list

    def sampleSet(self, nr):
        A = la.cholesky(self.variance)
        res = np.zeros((nr, self.dimension), dtype='Float64')
        for i in range(nr):
            res[i, :] = self.sample(A=A)
        return res

    def isValid(self, x):
        if len(x) != len(self.mean):
            Log.error("Expecting {{expected}} dimensions, got {{given}}", {"given":len(x), "expected":len(self.mean)})
        for v in x:
            try:
                float(v)
            except ValueError:
                Log.error("Invalid data: {{value}}", {"value": v})

    def flatStr(self, offset):
        offset += 1
        return "\t" * offset + ";MultiNormal;" + str(self.dimension) + ";" + str(self.mean.tolist()) + ";" + str(self.variance.tolist()) + "\n"


