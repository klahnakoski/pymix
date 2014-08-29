import random

from numpy.linalg import linalg as la
from numpy.oldnumeric.functions import argmax
import numpy
import scipy.stats
from core.distributions.prob import ProbDistribution

from core.priors.prior import PriorDistribution
from core.pymix_util.dataset import DataSet
from core.pymix_util.errors import InvalidDistributionInput, MixtureError


class EmptyComponent(MixtureError):
    """
    Raised if a Component is empty.
    """
    pass


class LinearGaussianDistribution(ProbDistribution):
    """
    Linear Gaussian Distribution
    """

    def __init__(self, p, beta, sigma, noise=0, alpha=0.0):
        """
        Constructor

        @param p: dimensionality of the distribution
        @param beta: coeficiente
        @param sigma: desvio padrao do erro
        """
        assert len(beta) == p, len(sigma) == 1
        self.p = p
        self.suff_p = p
        self.beta = numpy.array(beta, dtype='Float64')        # create a array (numpy) for variable beta
        self.sigma = numpy.array(sigma, dtype='Float64')    # create a array (numpy) for variable sigma
        self.freeParams = p + 1
        self.predicted = []
        self.noise = noise
        self.alpha = alpha

    def __copy__(self):
        return LinearGaussianDistribution(self.p, self.beta, self.sigma, alpha=self.alpha)

    def __str__(self):
        return "LinearGaussian:  [" + str(self.beta) + ", " + str(self.sigma) + "]"

    def __eq__(self, other):
        if not isinstance(other, LinearGaussianDistribution):
            return False
        if self.p != other.p:
            return False
        if not numpy.allclose(self.beta, other.beta) or not numpy.allclose(self.sigma, other.sigma):
            return False
        return True

    def pdf(self, data):
        if isinstance(data, DataSet):
            dt = data.internalData
        elif isinstance(data, numpy.ndarray):
            dt = data
        else:
            raise TypeError, "Unknown/Invalid input type."

        # First column of data set of the matrix
        y = dt[:, 0]
        # Matrix column of 1's concatenated with rest of columns of data set of the matrix
        #x = numpy.concatenate((numpy.array([numpy.ones(len(dt))]).T, dt[:,1:]), axis=1)
        x = dt[:, 1:]

        ## Calculating the expoent (y - x*beta)^2 / (sigma)^2
        #exp = numpy.divide(numpy.power(numpy.subtract(y, numpy.dot(x, self.beta)),2), self.sigma[0] ** 2)
        ## Calculating the factor 1/sqrt(2*pi)*sigma)
        #fat = 1 / (((2 * numpy.pi)**2) * self.sigma[0])
        ## Probability result
        #res = numpy.log(fat) - exp

        xbt = self.beta[0] + numpy.dot(x, self.beta[1:])
        # computing log likelihood


        res = scipy.stats.norm.pdf(y - xbt, 0, self.sigma[0])
        if self.noise > 0:
            print self.noise
            res = (1 - self.noise) * res + self.noise * scipy.stats.norm.pdf(y, 0, 5)
        outliers = res < float(1e-307)
        #print 'min', res[outliers], numpy.nonzero(outliers) #, self.beta[0]+numpy.dot(x[numpy.argmin(res)],self.beta[1:])
        res[outliers] = float(1e-307)
        return numpy.log(res)

    def MStep(self, posterior, data, mix_pi=None):
        if isinstance(data, DataSet):
            dt = data.internalData
        elif isinstance(data, numpy.ndarray):
            dt = data
        else:
            raise TypeError, "Unknown/Invalid input to MStep."


        # First column of data set of the matrix
        y = dt[:, 0]
        # Matrix column of 1's concatenated with rest of columns of data set of the matrix
        #x = numpy.concatenate((numpy.array([numpy.ones(len(dt))]).T, dt[:,1:]), axis=1)
        x = dt[:, 1:]

        # Beta estimation
        xaux = numpy.array(numpy.multiply(x, numpy.matrix(posterior).T))
        yaux = numpy.array(numpy.multiply(y, numpy.matrix(posterior).T))

        mean = numpy.mean(yaux)

        beta_numerator = numpy.dot(xaux.T, y)
        beta_denominator = numpy.dot(xaux.T, x)

        try:
            betashort = numpy.dot(numpy.linalg.inv(beta_denominator), beta_numerator)
            self.beta = numpy.concatenate(([mean], betashort), axis=1)
        except la.LinAlgError:
            raise EmptyComponent, "Empty Component: Singular Matrix"

        # Sigma estimation
        self.predicted = mean + numpy.dot(x, betashort)
        y_x_betat = numpy.subtract(y, self.predicted)
        self.predicted = numpy.multiply(self.predicted, posterior)

        sigma_numerator = numpy.dot(numpy.multiply(y_x_betat, posterior), y_x_betat)
        sigma_denominator = posterior.sum()

        self.sigma[0] = max(0.0001, numpy.sqrt(sigma_numerator / sigma_denominator))
        self.currentPosterior = posterior

    def predict(self, data, posterior=[]):
        if isinstance(data, DataSet):
            dt = data.internalData
        elif isinstance(data, numpy.ndarray):
            dt = data
        else:
            raise TypeError, "Unknown/Invalid input to MStep."

        # Matrix column of 1's concatenated with rest of columns of data set of the matrix
        x = numpy.concatenate((numpy.array([numpy.ones(len(dt))]).T, dt[:, 1:]), axis=1)
        #x = dt[:,1:]
        if len(posterior) > 0:
            return numpy.multiply(posterior, numpy.dot(x, self.beta.T))
        else:
            return numpy.dot(x, self.beta.T)


    def sample(self):
        """
        Samples from the Linear Gaussian Distribution
        """
        s = [None] * self.p

        beta_zero = numpy.array([self.beta[0]]).T
        beta_lin = self.beta[1:]

        s[0] = 1
        # x's samples
        res = 1 * self.beta[0]
        for i in range(1, self.p):
            s[i] = random.uniform(0, 1)
            res = res + self.beta[i] * s[i]

        # y sample
        s[0] = random.normalvariate(res, self.sigma[0])

        return s

    def sampleSet(self, nr):
        s = numpy.zeros((nr, self.p))
        for i in range(nr):
            x = self.sample()
            s[i, :] = x
        return s

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
        return "\t" * offset + ";LinearGaussian;" + str(self.p) + ";" + str(self.mu.tolist()) + ";" + str(self.sigma.tolist()) + "\n"


class LinearGaussianPriorDistribution(PriorDistribution):
    """ Gausian prior with 0 mean as prior distribuion """

    def __init__(self, alpha, fixed=1):
        self.alpha = alpha
        self.freeParams = len(alpha)
        self.constant_hyperparams = 1  # hyperparameters are constant
        self.fixed = 1

    def pdf(self, models):
        prior = 0
        for i, m in enumerate(models):
            #prior = prior + (len(m.beta)/2)*numpy.log(self.alpha[i]/(2*numpy.pi)) - numpy.dot(m.beta,m.beta.T)*self.alpha[i]/2
            for j in range(len(m.beta)):
                prior = prior + numpy.log(scipy.stats.norm.pdf(m.beta, 0, 1 / numpy.sqrt(self.alpha[i]))[0])
        if numpy.isnan(prior):
            print "prior is nan"
            for m in models:
                print
            return -100.0
        else:
            return prior


    def isValid(self, x):
        if isinstance(x, LinearGaussianDistribution):
            return True
        else:
            return False

    def __copy__(self):
        return LinearGaussianPriorDistribution(self.alpha, fixed=self.fixed)

    def __str__(self):
        return "LinearGaussianPrior: " + str(self.alpha)

    def updateHyperparameters(self, dists, posterior, data):
        """
        Update the hyperparameters in an empirical Bayes fashion.

        @param dists: list of ProbabilityDistribution objects
        @param posterior: numpy matrix of component membership posteriors
        @param data: DataSet object
        """
        pass

    def posterior(self, m, x):
        raise NotImplementedError, "Needs implementation"

    def marginal(self, x):
        raise NotImplementedError, "Needs implementation"

    def mapMStep(self, dist, posterior, data, mix_pi=None, dist_ind=None):

        """ This funciton is based on an empirical bayesian estimation of a linear gaussian with a gaussian prior over B.
            See bishop page 169 for details.
        """

        assert isinstance(dist, LinearGaussianDistribution)

        if isinstance(data, DataSet):
            dt = data.internalData
        elif isinstance(data, numpy.ndarray):
            dt = data
        else:
            raise TypeError, "Unknown/Invalid input to MStep."

        # First column of data set of the matrix
        y = dt[:, 0]
        # Matrix column of 1's concatenated with rest of columns of data set of the matrix
        #x = numpy.concatenate((numpy.array([numpy.ones(len(dt))]).T, dt[:,1:]), axis=1)
        x = dt[:, 1:]

        yaux = numpy.array(numpy.multiply(y, numpy.matrix(posterior).T))
        mean = numpy.mean(yaux)

        #eigen values of X^t.X/sigma^2
        xaux = numpy.array(numpy.multiply(x, numpy.matrix(posterior).T))
        XXs = numpy.dot(xaux.T, x) / numpy.power(dist.sigma[0], 2)

        if (self.fixed == 0):
            lambdas = la.eigvals(XXs)
            # estimate gamma
            self.gamma = numpy.sum(numpy.divide(lambdas, lambdas + self.alpha[dist_ind]))

        # Beta estimation
        beta_numerator = numpy.dot(xaux.T, y) / numpy.power(dist.sigma[0], 2)
        beta_denominator = self.alpha[dist_ind] * numpy.identity(len(x[0])) + XXs
        try:
            betashort = numpy.dot(numpy.linalg.inv(beta_denominator), beta_numerator)
            dist.beta = numpy.concatenate(([mean], betashort), axis=1)
        except la.LinAlgError:
            raise EmptyComponent, "Empty Component: Singular Matrix"

        # Sigma estimation
        dist.predicted = mean + numpy.dot(x, betashort)
        y_x_betat = numpy.subtract(y, dist.predicted)
        dist.predicted = numpy.multiply(dist.predicted, posterior)

        sigma_numerator = numpy.dot(numpy.multiply(y_x_betat, posterior), y_x_betat)
        if (self.fixed == 0):
            sigma_denominator = posterior.sum() - self.gamma
        else:
            sigma_denominator = posterior.sum() - 1

        try:
            dist.sigma[0] = numpy.sqrt(sigma_numerator / sigma_denominator)
        except FloatingPointError:
            dist.sigma[0] = 0.0001

        # alpha
        if (self.fixed == 0):
            self.alpha[dist_ind] = self.gamma / numpy.dot(dist.beta, dist.beta.T)

        dist.currentPosterior = posterior

        #print 'alpha', self.alpha
        #print 'gamma', self.gamma
        #print 'sigma', self.sigma


def evaluateRegression(mix, data, type=2, train=[], sparse=[]):
    """ evaluation types
         0 - use y and x to decide
         1 - use only mixing coeficients
         2 - use x
         3 - user expected y
    """

    if isinstance(data, DataSet):
        dt = data.internalData
    elif isinstance(data, numpy.ndarray):
        dt = data
    else:
        raise TypeError, "Unknown/Invalid input to MStep."

    # First column of data set of the matrix
    y = dt[:, 0]

    predy = numpy.zeros((1, len(y)))[0]

    for c in mix.components:
        c.noise = 0.01

    [log_l, log_p] = mix.EStep(data)
    p = numpy.exp(log_l)
    pmax = numpy.zeros((len(p), len(p[0])))
    for i in range(len(p[0])):
        pmax[numpy.argmax(p[:, i]), i] = 1
        pmult = numpy.multiply(pmax, p)

    if type in [2, 3]:
        [log_l, log_p] = mix.EStep(train)
        ptrain = numpy.exp(log_l)
        pmaxtrain = []
        for i in range(len(ptrain[0])):
            pmaxtrain.append(numpy.argmax(ptrain[:, i]))
        d1 = numpy.array(data.dataMatrix)
        d2 = numpy.array(train.dataMatrix)
        if len(sparse) == 0:
            dist = scipy.spatial.distance.cdist(d1[:, 1:], d2[:, 1:], 'euclidean')
        else:
            # if sparse version was used than only relevant variables should be compared
            dist = numpy.zeros(len(d1), len(d2))
            for l in range(len(train)):
                dist[:, l] = scipy.spatial.distance.cdist(d1[:, sparse[pmaxtrain[l]]], d2[:, sparse[pmaxtrain[l]]], 'euclidean') / numpy.sqrt(len(sparse[pmaxtrain[l]]))
        labels = knn(d1[:, 1:], d2[:, 1:], pmaxtrain, 1)
        for i in range(len(p[0])):
            pmax[:, i] = 0
            pmax[labels[i], i] = 1

        if type == 3:
            meansy = []
            stdsy = []
            # estimate means and std
            for l, m in enumerate(mix.components):
                yprime = numpy.multiply(d2[:, 0], ptrain[:, l])
                meansy.append(sum(yprime) / sum(ptrain[:, l]))
                stdsy.append(numpy.multiply(yprime, d2[:, 0]) / sum(ptrain[:, l]) - meansy[l])

    means = []
    stds = []
    genes = []

    predyaux = []
    posteriorpred = []
    for i, m in enumerate(mix.components):
        if type == 0:
            predy = predy + m.distList[0].predict(data, posterior=pmax[i])
        elif type == 2:
            predy = predy + m.distList[0].predict(data, posterior=pmax[i])
        elif type == 1:
            predy = predy + mix.pi[i] * m.distList[0].predict(data)
        elif type == 3:
            aux = m.distList[0].predict(data)
            predyaux.append(aux)
            posteriorpred.append(scipy.stats.norm.pdf(aux, meansy[i], stdsy[i]))

    if type == 3:
        predyaux = numpy.array(predyaux).T
        posteriorpred = numpy.array(posteriorpred).T
        for i, d in enumerate(data):
            predy = predyaux[i, argmax(posteriorpred[i, :])]

    # estimate pearson
    [r, p] = scipy.stats.pearsonr(y, predy)
    errorv = numpy.power(y - predy, 2)
    error = sum(errorv) / len(y)

    for i, m in enumerate(mix.components):
        list = [(j, data.sampleIDs[k]) for k, j in enumerate(errorv) if pmult[i, k] > 0]
        list.sort()
        genes.append([j for (k, j) in list])
    return [r, p, predy, y, means, stds, error, genes]


def knn(train, test, labels, k):
    labels = numpy.array(labels)
    if max(labels) > 0:
        dist = scipy.spatial.distance.cdist(train, test, 'euclidean')
        testlabel = []
        for i in range(len(train)):
            indices = numpy.argsort(dist[i, :])
            values = labels[indices[:k]]
            hist = numpy.zeros((max(values) + 1))
            for j in values:
                hist[j] = hist[j] + 1
            testlabel.append(numpy.argmax(hist))
    else:
        testlabel = len(train) * [0]
    return testlabel
