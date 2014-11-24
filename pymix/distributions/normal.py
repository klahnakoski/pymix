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

import copy
import random
import math
from scipy import stats
import numpy as np
from .prob import ProbDistribution
from pyLibrary.maths import Math
from pymix.util.ghmm import random_mt
from ..util.errors import InvalidPosteriorDistribution, InvalidDistributionInput
from ..util.dataset import DataSet


class NormalDistribution(ProbDistribution):
    """
    Univariate Normal Distribution

    """

    def __init__(self, mu, sigma, dummy=0):
        """
        Constructor

        @param mu: mean parameter
        @param sigma: standard deviation parameter
        @param dummy: for when initialization from number arrays
        """
        self.dimension = 1
        self.suff_p = 1
        self.mean = mu
        self.variance = sigma

        self.freeParams = 2

        self.min_sigma = 0.25  # minimal standard deviation
        self.fixed = 0  #allow parameter update

    def __eq__(self, other):
        res = False
        if isinstance(other, NormalDistribution):
            if np.allclose(other.mean, self.mean) and np.allclose(other.variance, self.variance):
                res = True
        return res

    def __copy__(self):
        return NormalDistribution(copy.deepcopy(self.mean), copy.deepcopy(self.variance))

    def __str__(self):
        return "Normal:  [" + str(self.mean) + ", " + str(self.variance) + "]"


    def pdf(self, data):

        # Valid input arrays will have the form [[sample1],[sample2],...] or
        # [sample1,sample2, ...], the latter being the input format to the extension function,
        # so we might have to reformat the data
        if isinstance(data, DataSet):
            assert data.internalData is not None, "Internal data not initialized."
            nr = len(data.internalData)
            assert data.internalData.shape == (nr, 1), 'shape = ' + str(data.internalData.shape)

            x = np.transpose(data.internalData)[0]

        elif hasattr(data, "__iter__"):
            nr = len(data)

            if data.shape == (nr, 1):  # data format needs to be changed
                x = np.transpose(data)[0]
            elif data.shape == (nr,):
                x = data
            else:
                raise TypeError, 'Invalid data shape: ' + str(data.shape)
        else:
            raise TypeError, "Unknown/Invalid input type:" + str(type(data))

        # computing log likelihood
        res = stats.norm.pdf(x, loc=self.mean, scale=self.variance)
        return np.log(res)

    def linear_pdf(self, x):
        # computing log likelihood
        res = stats.norm.pdf(x, loc=self.mean, scale=self.variance)
        return res

    def sample(self):
        r2 = -2.0 * Math.log(random_mt.float23())   # r2 ~ chi-square(2)
        theta = 2.0 * math.pi * random_mt.float23()  # theta ~ uniform(0, 2 \pi)
        return math.sqrt(self.variance) * math.sqrt(r2) * math.cos(theta) + self.mean


    def sampleSet(self, nr):
        res = np.zeros(nr, dtype='Float64')

        for i in range(nr):
            res[i] = self.sample()

        return res

    def sufficientStatistics(self, posterior, data):
        """
        Returns sufficient statistics for a given data set and posterior. In case of the Normal distribution
        this is the dot product of a vector of component membership posteriors with the data and the square
        of the data.

        @param posterior: numpy vector of component membership posteriors
        @param data: numpy vector holding the data

        @return: list with dot(posterior, data) and dot(posterior, data**2)
        """
        return np.array([np.dot(posterior, data)[0], np.dot(posterior, data ** 2)[0]], dtype='Float64')


    def MStep(self, posterior, data, mix_pi=None):
        # data has to be reshaped for parameter estimation
        if isinstance(data, DataSet):
            x = data.internalData[:, 0]
        elif hasattr(data, "__iter__"):
            x = data[:, 0]

        else:
            raise TypeError, "Unknown/Invalid input to MStep."
        nr = len(x)

        sh = x.shape
        assert sh == (nr,)  # XXX debug

        post_sum = np.sum(posterior)

        # checking for valid posterior: if post_sum is zero, this component is invalid
        # for this data set
        if post_sum != 0.0:
            # computing ML estimates for mu and sigma
            new_mu = np.dot(posterior, x) / post_sum
            new_sigma = math.sqrt(np.dot(posterior, (x - new_mu) ** 2) / post_sum)
        else:
            raise InvalidPosteriorDistribution, "Sum of posterior is zero: " + str(self) + " has zero likelihood for data set."

        if new_sigma < self.min_sigma:
        # enforcing non zero variance estimate
            new_sigma = self.min_sigma

        # assigning updated parameter values
        self.mean = new_mu
        self.variance = new_sigma

    def isValid(self, x):
        try:
            float(x)
        except (ValueError):
            #print "Invalid data: ",x,"in NormalDistribution."
            raise InvalidDistributionInput, "\n\tInvalid data: " + str(x) + " in NormalDistribution."

    def formatData(self, x):
        if isinstance(x, list) and len(x) == 1:
            x = x[0]
        self.isValid(x)  # make sure x is valid argument
        return [self.dimension, [x]]


    def flatStr(self, offset):
        offset += 1
        return "\t" * +offset + ";Norm;" + str(self.mean) + ";" + str(self.variance) + "\n"

    def posteriorTraceback(self, x):
        return self.pdf(x)

    def merge(self, dlist, weights):
        raise DeprecationWarning, 'Part of the outdated structure learning implementation.'
