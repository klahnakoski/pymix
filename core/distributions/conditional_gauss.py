from random import random
import math
import numpy as np
from core.distributions.prob import ProbDistribution
from core.pymix_util.errors import InvalidDistributionInput


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

        self.p = p
        self.suff_p = p
        self.freeParams = p * 3
        self.mu = mu  # mean values
        self.w = w    # conditional weights
        self.sigma = sigma  # standard deviations

        self.parents = parents  # tree structure encoded by parent index relationship

    def __str__(self):
        return 'ConditionalGaussian: \nmu=' + str(self.mu) + ', \nsigma=' + str(self.sigma) + ', \nw=' + str(self.w) + ', \nparents=' + str(self.parents)


    def sample(self):
        s = [None] * self.p
        s[0] = random.normalvariate(self.mu[0], self.sigma[0])

        for i in range(1, self.p):
            pid = self.parents[i]
            assert s[pid] != None   # XXX assumes that the features are in topological order
            shift_mu = self.mu[i] - (self.w[i] * self.mu[pid])
            s[i] = random.normalvariate(shift_mu + (self.w[i] * s[pid]), self.sigma[i])

        return s

    def sampleSet(self, nr):
        s = np.zeros((nr, self.p))
        for i in range(nr):
            s[i, :] = self.sample()

        return s


    def pdf(self, data):

        # XXX assume root as first index
        assert self.parents[0] == -1
        assert self.w[0] == 0.0

        res = np.zeros(len(data))

        for i in range(len(data)):
            res[i] = math.log((1.0 / (math.sqrt(2.0 * math.pi) * self.sigma[0])) * math.exp(( data[i, 0] - self.mu[0]  ) ** 2 / (-2.0 * self.sigma[0] ** 2)))
            for j in range(1, self.p):
                pind = self.parents[j]
                res[i] += math.log(
                    (1.0 / (math.sqrt(2.0 * math.pi) * self.sigma[j])) * math.exp(( data[i, j] - self.mu[j] - self.w[j] * ( data[i, pind] - self.mu[pind] )  ) ** 2 / (-2.0 * self.sigma[j] ** 2)))

        return res


    def MStep(self, posterior, data, mix_pi=None):
        var = {}
        post_sum = np.sum(posterior)

        # checking for valid posterior: if post_sum is zero, this component is invalid
        # for this data set
        if post_sum != 0.0:
            # reestimate mu
            for j in range(self.p):
                self.mu[j] = np.dot(posterior, data[:, j]) / post_sum
                var[j] = np.dot(posterior, (data[:, j] - self.mu[j]) ** 2) / post_sum

            for j in range(self.p):
                # computing ML estimates for w and sigma
                pid = self.parents[j]
                cov_j = np.dot(posterior, (data[:, j] - self.mu[j]) * (data[:, pid] - self.mu[pid])) / post_sum

                if pid <> -1:  # has parents
                    self.w[j] = cov_j / var[pid]
                    print  var[j], self.w[j] ** 2, var[pid], var[j] - (self.w[j] ** 2 * var[pid])
                    self.sigma[j] = math.sqrt(var[j] - (self.w[j] ** 2 * var[pid]))
                else:
                    self.sigma[j] = math.sqrt(var[j])

        else:
            raise ValueError, 'Invalid posterior.'


    def isValid(self, x):
        if not len(x) == self.p:
            raise InvalidDistributionInput, "\n\tInvalid data: wrong dimension(s) " + str(len(x)) + " in MultiNormalDistribution(p=" + str(self.p) + ")."
        for v in x:
            try:
                float(v)
            except (ValueError):
                raise InvalidDistributionInput, "\n\tInvalid data: " + str(x) + " in MultiNormalDistribution."

