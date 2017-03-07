

#-------------------------------------------------------------------------------
#- Distribution and derived  ---------------------------------------------------
from pymix.distributions.uniform import UniformDistribution


class Distribution(object):
    """ Abstract base class for distribution over EmissionDomains
    """

    # add density, mass, cumuliative dist, quantils, sample, fit pars,
    # moments


class DiscreteDistribution(Distribution):
    """ A DiscreteDistribution over an Alphabet: The discrete distribution
    is parameterized by the vectors of probabilities.

    """

    def __init__(self, alphabet):
        self.alphabet = alphabet
        self.prob_vector = None
        self.dimension = 1

    def set(self, prob_vector):
        self.prob_vector = prob_vector

    def get(self):
        return self.prob_vector


class ContinuousDistribution(Distribution):
    pass


class GaussianDistribution(ContinuousDistribution):
    # XXX attributes unused at this point
    def __init__(self, domain):
        self.emissionDomain = domain
        self.mean = None
        self.variance = None

    def set(self, values):
        """
        @param values tuple of mu, sigma, trunc
        """
        mu, sigma = values
        self.mean = mu
        self.variance = sigma

    def get(self):
        return (self.mean, self.variance)


class TruncGaussianDistribution(GaussianDistribution):
    # XXX attributes unused at this point
    def __init__(self, domain):
        GaussianDistribution.__init__(self, domain)
        self.trunc = None

    def set(self, values):
        """
        @param values tuple of mu, sigma, trunc
        """
        mu, sigma, trunc = values
        self.mean = mu
        self.variance = sigma
        self.trunc = trunc

    def get(self):
        return (self.mean, self.variance, self.trunc)


class GaussianMixtureDistribution(ContinuousDistribution):
    # XXX attributes unused at this point
    def __init__(self, domain):
        self.emissionDomain = domain
        self.M = None   # number of mixture components
        self.mean = []
        self.variance = []
        self.weight = []

    def set(self, index, values):
        """
        @param index index of mixture component
        @param values tuple of mu, sigma, w
        """
        mu, sigma, w = values
        pass

    def get(self):
        pass


class ContinuousMixtureDistribution(ContinuousDistribution):
    def __init__(self, domain):
        self.emissionDomain = domain
        self.M = 0   # number of mixture components
        self.components = []
        self.weight = []
        self.fix = []

    def add(self, w, fix, distribution):
        assert isinstance(distribution, ContinuousDistribution)
        self.M = self.M + 1
        self.weight.append(w)
        self.components.append(distribution)
        if isinstance(distribution, UniformDistribution):
            # uniform distributions are fixed by definition
            self.fix.append(1)
        else:
            self.fix.append(fix)

    def set(self, index, w, fix, distribution):
        if index >= self.M:
            raise IndexError

        assert isinstance(distribution, ContinuousDistribution)
        self.weight[index] = w
        self.components[index] = distribution
        if isinstance(distribution, UniformDistribution):
            # uniform distributions are fixed by definition
            self.fix[index](1)
        else:
            self.fix[index](fix)

    def get(self, i):
        assert self.M > i
        return self.weight[i], self.fix[i], self.components[i]

    def check(self):
        assert self.M == len(self.components)
        assert sum(self.weight) == 1
        assert sum(self.weight > 1) == 0
        assert sum(self.weight < 0) == 0


class MultivariateGaussianDistribution(ContinuousDistribution):
    def __init__(self, domain):
        self.emissionDomain = domain

