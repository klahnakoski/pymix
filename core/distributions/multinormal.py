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
from core.distributions.prob import ProbDistribution
from core.pymix_util.errors import InvalidDistributionInput
from core.pymix_util.dataset import DataSet


class MultiNormalDistribution(ProbDistribution):
    """
    Multivariate Normal Distribution

    """

    def __init__(self, p, mu, sigma):
        """
        Constructor

        @param p: dimensionality of the distribution
        @param mu: mean parameter vector
        @param sigma: covariance matrix
        """

        assert len(mu) == len(sigma) == len(sigma[0]) == p, str(len(mu)) + ' == ' + str(len(sigma)) + ' == ' + str(len(sigma[0])) + ' == ' + str(p)
        self.p = p
        self.suff_p = p
        self.mu = np.array(mu, dtype='Float64')
        self.sigma = np.array(sigma, dtype='Float64')
        self.freeParams = p + p ** 2


    def __copy__(self):
        return MultiNormalDistribution(self.p, self.mu, self.sigma)


    def __str__(self):
        return "Normal:  [" + str(self.mu) + ", " + str(self.sigma.tolist()) + "]"

    def __eq__(self, other):
        if not isinstance(other, MultiNormalDistribution):
            return False
        if self.p != other.p:
            return False
        if not np.allclose(self.mu, other.mu) or not np.allclose(self.sigma, other.sigma):
            return False
        return True

    def pdf(self, data):
        if isinstance(data, DataSet):
            x = data.internalData
        elif hasattr(data, "__iter__"):
            x = data
        else:
            raise TypeError, "Unknown/Invalid input type."

        # initial part of the formula
        # this code depends only on the model parameters ... optmize?
        dd = la.det(self.sigma);
        inverse = la.inv(self.sigma);
        ff = math.pow(2 * math.pi, -self.p / 2.0) * math.pow(dd, -0.5);

        # centered input values
        centered = np.subtract(x, np.repeat([self.mu], len(x), axis=0))

        res = ff * np.exp(-0.5 * np.sum(np.multiply(centered, np.dot(centered, inverse)), 1))

        return np.log(res)

    def MStep(self, posterior, data, mix_pi=None):

        if isinstance(data, DataSet):
            x = data.internalData
        elif hasattr(data, "__iter__"):
            x = data
        else:
            raise TypeError, "Unknown/Invalid input to MStep."

        post = posterior.sum() # sum of posteriors
        self.mu = np.dot(posterior, x) / post

        # centered input values (with new mus)
        centered = np.subtract(x, np.repeat([self.mu], len(x), axis=0));
        self.sigma = np.dot(np.transpose(np.dot(np.identity(len(posterior)) * posterior, centered)), centered) / post


    def sample(self, A=None):
        """
        Samples from the mulitvariate Normal distribution.

        @param A: optional Cholesky decomposition of the covariance matrix self.sigma, can speed up
        the sampling
        """
        if A == None:
            A = la.cholesky(self.sigma)

        z = np.zeros(self.p, dtype='Float64')
        for i in range(self.p):
            z[i] = random.normalvariate(0.0, 1.0)  # sample p iid N(0,1) RVs

        X = np.dot(A, z) + self.mu
        return X.tolist()  # return value of sample must be Python list

    def sampleSet(self, nr):
        A = la.cholesky(self.sigma)
        res = np.zeros((nr, self.p), dtype='Float64')
        for i in range(nr):
            res[i, :] = self.sample(A=A)
        return res

    def isValid(self, x):
        if not len(x) == self.p:
            raise InvalidDistributionInput, "\n\tInvalid data: wrong dimension(s) " + str(len(x)) + " in MultiNormalDistribution(p=" + str(self.p) + ")."
        for v in x:
            try:
                float(v)
            except (ValueError):
                raise InvalidDistributionInput, "\n\tInvalid data: " + str(x) + " in MultiNormalDistribution."

    def flatStr(self, offset):
        offset += 1
        return "\t" * offset + ";MultiNormal;" + str(self.p) + ";" + str(self.mu.tolist()) + ";" + str(self.sigma.tolist()) + "\n"


