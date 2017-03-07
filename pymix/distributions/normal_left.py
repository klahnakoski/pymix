import copy
import math
import numpy as np
from pyLibrary.debugs.logs import Log
from pyLibrary.maths import Math
from pymix.distributions.normal import NormalDistribution
from pymix.distributions.normal_right import NormalRight
from pymix.util.ghmm import random_mt
from pymix.util.ghmm.randvar import ighmm_rand_get_1overa, ighmm_rand_get_PHI
from ..util.errors import InvalidDistributionInput


class NormalLeft(NormalDistribution):
    """
    Univariate Normal Distribution

    """

    def __init__(self, mu, sigma, maximum, *dummy_args):
        """
        Constructor

        @param mu: mean parameter
        @param sigma: standard deviation parameter
        """
        self.dimension = 1
        self.suff_p = 1
        self.mean = mu
        self.variance = sigma
        self.maximum = maximum

        self.freeParams = 2

        self.min_sigma = 0.25  # minimal standard deviation
        self.fixed = 0  #allow parameter update

    def __eq__(self, other):
        pass

    def __copy__(self):
        return NormalLeft(self.mean, self.variance, self.maximum)

    def __str__(self):
        return "NormalLeft:  [" + str(self.mean) + ", " + str(self.variance) + "]"


    def pdf(self, data):
        return [math.log(self.linear_pdf(d)) for d in data]


    def linear_pdf(self, x):
        if x > self.maximum:
            return 0.0

        c = ighmm_rand_get_1overa(-self.maximum, -self.mean, self.variance)
        return c * NormalDistribution(-self.mean, self.variance).linear_pdf(-x)


    def cdf(self, x):
        """
        cumalative distribution function of a-truncated N(mean, u)
        """
        if x <= self.minimum:
            return 0.0
        if self.variance <= self.minimum:
            Log.error("u <= a not allowed\n")
            #
        #     Function: int erfc (x, result)
        #     These routines compute the complementary error function
        #     erfc(x) = 1 - erf(x) = 2/\sqrt(\pi) \int_x^\infty \exp(-t^2).
        #
        return 1.0 + (math.erf((x -self.mean) / math.sqrt(2*self.variance)) - 1.0) / math.erfc((self.minimum - self.mean) / math.sqrt(self.variance*2))


    def sample(self, seed=0, native=False):
        return -NormalRight(-self.mean, self.variance, -self.maximum).sample(seed=seed, native=native)


    def sampleSet(self, nr, native=False):
        gen = NormalRight(-self.mean, self.variance, -self.maximum)
        return [-gen.sample(native=native) for _ in range(nr)]


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
