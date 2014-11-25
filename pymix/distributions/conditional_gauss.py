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
import numpy as np
from .prob import ProbDistribution
from pyLibrary.maths import Math
from pymix.distributions.normal import NormalDistribution
from pymix.util.errors import InvalidDistributionInput


class ConditionalGaussDistribution(ProbDistribution):
    """
    Constructor for conditional Gauss distributions. Conditional Gaussians
    use a sparse formulation of the covariance matrix, which allows computationally
    efficient modeling of covariance for high-dimensional data.

    Also, the conditional Gaussians implement a tree dependency structure.

    """

    def __init__(self, p, mu, w, sigma, parents):
        """
        Constructor

        @param p: dimensionality of the distribution
        @param mu: mean parameter vector
        @param w: covariance weights (representing off-diagonal entries in the full covariance matrix)
        @param sigma: standard deviations (diagonal entries of the covariance matrix)
        @param parents: parents in the tree structure implied by w
        """
        assert p == len(mu) == len(w) == len(sigma) == len(parents)

        self.dimension = p
        self.suff_p = p
        self.freeParams = p * 3
        self.mean = mu  # mean values
        self.w = w    # conditional weights
        self.variance = sigma  # standard deviations

        self.parents = parents  # tree structure encoded by parent index relationship

    def __str__(self):
        return 'ConditionalGaussian: \nmu=' + str(self.mean) + ', \nsigma=' + str(self.variance) + ', \nw=' + str(self.w) + ', \nparents=' + str(self.parents)


    def sample(self, native=False):
        s = [None] * self.dimension
        s[0] = NormalDistribution(self.mean[0], self.variance[0]).sample()

        for i in range(1, self.dimension):
            pid = self.parents[i]
            assert s[pid] != None   # XXX assumes that the features are in topological order
            shift_mu = self.mean[i] - (self.w[i] * self.mean[pid])
            s[i] = NormalDistribution(shift_mu + (self.w[i] * s[pid]), self.variance[i]).sample()

        return s

    def sampleSet(self, nr):
        s = np.zeros((nr, self.dimension))
        for i in range(nr):
            s[i, :] = self.sample()

        return s


    def pdf(self, data):

        # XXX assume root as first index
        assert self.parents[0] == -1
        assert self.w[0] == 0.0

        res = np.zeros(len(data))

        for i in range(len(data)):
            res[i] = Math.log((1.0 / (math.sqrt(2.0 * math.pi) * self.variance[0])) * math.exp(( data[i, 0] - self.mean[0]  ) ** 2 / (-2.0 * self.variance[0] ** 2)))
            for j in range(1, self.dimension):
                pind = self.parents[j]
                res[i] += Math.log(
                    (1.0 / (math.sqrt(2.0 * math.pi) * self.variance[j])) * math.exp(( data[i, j] - self.mean[j] - self.w[j] * ( data[i, pind] - self.mean[pind] )  ) ** 2 / (-2.0 * self.variance[j] ** 2)))

        return res


    def MStep(self, posterior, data, mix_pi=None):
        var = {}
        post_sum = np.sum(posterior)

        # checking for valid posterior: if post_sum is zero, this component is invalid
        # for this data set
        if post_sum != 0.0:
            # reestimate mu
            for j in range(self.dimension):
                self.mean[j] = np.dot(posterior, data[:, j]) / post_sum
                var[j] = np.dot(posterior, (data[:, j] - self.mean[j]) ** 2) / post_sum

            for j in range(self.dimension):
                # computing ML estimates for w and sigma
                pid = self.parents[j]
                cov_j = np.dot(posterior, (data[:, j] - self.mean[j]) * (data[:, pid] - self.mean[pid])) / post_sum

                if pid <> -1:  # has parents
                    self.w[j] = cov_j / var[pid]
                    print  var[j], self.w[j] ** 2, var[pid], var[j] - (self.w[j] ** 2 * var[pid])
                    self.variance[j] = math.sqrt(var[j] - (self.w[j] ** 2 * var[pid]))
                else:
                    self.variance[j] = math.sqrt(var[j])

        else:
            raise ValueError, 'Invalid posterior.'


    def isValid(self, x):
        if not len(x) == self.dimension:
            raise InvalidDistributionInput, "\n\tInvalid data: wrong dimension(s) " + str(len(x)) + " in MultiNormalDistribution(p=" + str(self.dimension) + ")."
        for v in x:
            try:
                float(v)
            except (ValueError):
                raise InvalidDistributionInput, "\n\tInvalid data: " + str(x) + " in MultiNormalDistribution."

