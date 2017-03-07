import copy
import math
import numpy as np
from pyLibrary.debugs.logs import Log
from pyLibrary.maths import Math
from pymix.distributions.normal import NormalDistribution
from pymix.util.ghmm import random_mt
from pymix.util.ghmm.randvar import ighmm_rand_get_1overa, ighmm_rand_get_PHI
from ..util.errors import InvalidDistributionInput


class NormalRight(NormalDistribution):
    """
    Univariate Normal Distribution

    """

    def __init__(self, mu, sigma, minimum, *dummy_args):
        """
        Constructor

        @param mu: mean parameter
        @param sigma: standard deviation parameter
        """
        self.dimension = 1
        self.suff_p = 1
        self.mean = mu
        self.variance = sigma
        self.minimum = minimum

        self.freeParams = 2

        self.min_sigma = 0.25  # minimal standard deviation
        self.fixed = 0  #allow parameter update

    def __eq__(self, other):
        res = False
        if isinstance(other, NormalRight):
            if np.allclose(other.mean, self.mean) and np.allclose(other.variance, self.variance):
                res = True
        return res

    def __copy__(self):
        return NormalRight(copy.deepcopy(self.mean), copy.deepcopy(self.variance))

    def __str__(self):
        return "Normal:  [" + str(self.mean) + ", " + str(self.variance) + "]"


    def pdf(self, data):
        return [math.log(self.linear_pdf(d)) for d in data]


    def linear_pdf(self, x):
        if x < self.minimum:
            return 0.0

        # move mean to the right position
        c = ighmm_rand_get_1overa(self.minimum, self.mean, self.variance)
        return c * NormalDistribution.linear_pdf(self, x)


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
        C0 = 2.515517
        C1 = 0.802853
        C2 = 0.010328
        D1 = 1.432788
        D2 = 0.189269
        D3 = 0.001308


        if self.variance <= 0.0:
            Log.error("u <= 0.0 not allowed\n")

        sigma = math.sqrt(self.variance)

        if seed != 0:
            random_mt.set_seed(seed)


        # Inverse transformation with restricted sampling by Fishman
        U = random_mt.float23()
        Feps = ighmm_rand_get_PHI((self.minimum - self.mean) / sigma)

        Us = Feps + (1 - Feps) * U
        Us1 = 1 - Us
        t = min(Us, Us1)

        t = math.sqrt(-Math.log(t))

        T = sigma * (t - (C0 + t * (C1 + t * C2)) / (1 + t * (D1 + t * (D2 + t * D3))))

        if Us < Us1:
            x = self.mean - T
        else:
            x = self.mean + T

        return x


    def sampleSet(self, nr):
        pass

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
