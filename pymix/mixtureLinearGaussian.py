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

import random

from numpy.linalg import linalg as la
import numpy as np
import scipy.stats
from pymix.distributions.prob import ProbDistribution
from pymix.distributions.uniform import UniformDistribution

from pymix.priors.prior import PriorDistribution
from pymix.util.dataset import DataSet
from pymix.util.errors import InvalidDistributionInput, MixtureError


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
        self.dimension = p
        self.suff_p = p
        self.beta = np.array(beta, dtype='Float64')        # create a array (numpy) for variable beta
        self.variance = np.array(sigma, dtype='Float64')    # create a array (numpy) for variable sigma
        self.freeParams = p + 1
        self.predicted = []
        self.noise = noise
        self.alpha = alpha

    def __copy__(self):
        return LinearGaussianDistribution(self.dimension, self.beta, self.variance, alpha=self.alpha)

    def __str__(self):
        return "LinearGaussian:  [" + str(self.beta) + ", " + str(self.variance) + "]"

    def __eq__(self, other):
        if not isinstance(other, LinearGaussianDistribution):
            return False
        if self.dimension != other.dimension:
            return False
        if not np.allclose(self.beta, other.beta) or not np.allclose(self.variance, other.variance):
            return False
        return True

    def pdf(self, data):
        if isinstance(data, DataSet):
            dt = data.internalData
        elif hasattr(data, "__iter__"):
            dt = data
        else:
            raise TypeError, "Unknown/Invalid input type."

        # First column of data set of the matrix
        y = dt[:, 0]
        # Matrix column of 1's concatenated with rest of columns of data set of the matrix
        #x = np.concatenate((np.array([np.ones(len(dt))]).T, dt[:,1:]), axis=1)
        x = dt[:, 1:]

        ## Calculating the expoent (y - x*beta)^2 / (sigma)^2
        #exp = np.divide(np.power(np.subtract(y, np.dot(x, self.beta)),2), self.variance.0] ** 2)
        ## Calculating the factor 1/sqrt(2*pi)*sigma)
        #fat = 1 / (((2 * np.pi)**2) * self.variance[0])
        ## Probability result
        #res = np.log(fat) - exp

        xbt = self.beta[0] + np.dot(x, self.beta[1:])
        # computing log likelihood


        res = scipy.stats.norm.pdf(y - xbt, 0, self.variance[0])
        if self.noise > 0:
            print self.noise
            res = (1 - self.noise) * res + self.noise * scipy.stats.norm.pdf(y, 0, 5)
        outliers = res < float(1e-307)
        #print 'min', res[outliers], np.nonzero(outliers) #, self.beta[0]+np.dot(x[np.argmin(res)],self.beta[1:])
        res[outliers] = float(1e-307)
        return np.log(res)

    def MStep(self, posterior, data, mix_pi=None):
        if isinstance(data, DataSet):
            dt = data.internalData
        elif hasattr(data, "__iter__"):
            dt = data
        else:
            raise TypeError, "Unknown/Invalid input to MStep."


        # First column of data set of the matrix
        y = dt[:, 0]
        # Matrix column of 1's concatenated with rest of columns of data set of the matrix
        #x = np.concatenate((np.array([np.ones(len(dt))]).T, dt[:,1:]), axis=1)
        x = dt[:, 1:]

        # Beta estimation
        xaux = np.array(np.multiply(x, np.matrix(posterior).T))
        yaux = np.array(np.multiply(y, np.matrix(posterior).T))

        mean = np.mean(yaux)

        beta_numerator = np.dot(xaux.T, y)
        beta_denominator = np.dot(xaux.T, x)

        try:
            betashort = np.dot(np.linalg.inv(beta_denominator), beta_numerator)
            self.beta = np.concatenate(([mean], betashort), axis=1)
        except la.LinAlgError:
            raise EmptyComponent, "Empty Component: Singular Matrix"

        # Sigma estimation
        self.predicted = mean + np.dot(x, betashort)
        y_x_betat = np.subtract(y, self.predicted)
        self.predicted = np.multiply(self.predicted, posterior)

        sigma_numerator = np.dot(np.multiply(y_x_betat, posterior), y_x_betat)
        sigma_denominator = posterior.sum()

        self.variance[0] = max(0.0001, np.sqrt(sigma_numerator / sigma_denominator))
        self.currentPosterior = posterior

    def predict(self, data, posterior=[]):
        if isinstance(data, DataSet):
            dt = data.internalData
        elif hasattr(data, "__iter__"):
            dt = data
        else:
            raise TypeError, "Unknown/Invalid input to MStep."

        # Matrix column of 1's concatenated with rest of columns of data set of the matrix
        x = np.concatenate((np.array([np.ones(len(dt))]).T, dt[:, 1:]), axis=1)
        #x = dt[:,1:]
        if len(posterior) > 0:
            return np.multiply(posterior, np.dot(x, self.beta.T))
        else:
            return np.dot(x, self.beta.T)


    def sample(self):
        """
        Samples from the Linear Gaussian Distribution
        """
        s = [None] * self.dimension

        beta_zero = np.array([self.beta[0]]).T
        beta_lin = self.beta[1:]

        s[0] = 1
        # x's samples
        res = 1 * self.beta[0]
        for i in range(1, self.dimension):
            s[i] = UniformDistribution(0,1).sample()
            res = res + self.beta[i] * s[i]

        # y sample
        s[0] = random.normalvariate(res, self.variance[0])

        return s

    def sampleSet(self, nr):
        s = np.zeros((nr, self.dimension))
        for i in range(nr):
            x = self.sample()
            s[i, :] = x
        return s

    def isValid(self, x):
        if not len(x) == self.dimension:
            raise InvalidDistributionInput, "\n\tInvalid data: wrong dimension(s) " + str(len(x)) + " in MultiNormalDistribution(p=" + str(self.dimension) + ")."
        for v in x:
            try:
                float(v)
            except (ValueError):
                raise InvalidDistributionInput, "\n\tInvalid data: " + str(x) + " in MultiNormalDistribution."

    def flatStr(self, offset):
        offset += 1
        return "\t" * offset + ";LinearGaussian;" + str(self.dimension) + ";" + str(self.mean.tolist()) + ";" + str(self.variance.tolist()) + "\n"


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
            #prior = prior + (len(m.beta)/2)*np.log(self.alpha[i]/(2*np.pi)) - np.dot(m.beta,m.beta.T)*self.alpha[i]/2
            for j in range(len(m.beta)):
                prior = prior + np.log(scipy.stats.norm.pdf(m.beta, 0, 1 / np.sqrt(self.alpha[i]))[0])
        if np.isnan(prior):
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
        elif hasattr(data, "__iter__"):
            dt = data
        else:
            raise TypeError, "Unknown/Invalid input to MStep."

        # First column of data set of the matrix
        y = dt[:, 0]
        # Matrix column of 1's concatenated with rest of columns of data set of the matrix
        #x = np.concatenate((np.array([np.ones(len(dt))]).T, dt[:,1:]), axis=1)
        x = dt[:, 1:]

        yaux = np.array(np.multiply(y, np.matrix(posterior).T))
        mean = np.mean(yaux)

        #eigen values of X^t.X/sigma^2
        xaux = np.array(np.multiply(x, np.matrix(posterior).T))
        XXs = np.dot(xaux.T, x) / np.power(dist.variance[0], 2)

        if (self.fixed == 0):
            lambdas = la.eigvals(XXs)
            # estimate gamma
            self.gamma = np.sum(np.divide(lambdas, lambdas + self.alpha[dist_ind]))

        # Beta estimation
        beta_numerator = np.dot(xaux.T, y) / np.power(dist.variance[0], 2)
        beta_denominator = self.alpha[dist_ind] * np.identity(len(x[0])) + XXs
        try:
            betashort = np.dot(np.linalg.inv(beta_denominator), beta_numerator)
            dist.beta = np.concatenate(([mean], betashort), axis=1)
        except la.LinAlgError:
            raise EmptyComponent, "Empty Component: Singular Matrix"

        # Sigma estimation
        dist.predicted = mean + np.dot(x, betashort)
        y_x_betat = np.subtract(y, dist.predicted)
        dist.predicted = np.multiply(dist.predicted, posterior)

        sigma_numerator = np.dot(np.multiply(y_x_betat, posterior), y_x_betat)
        if (self.fixed == 0):
            sigma_denominator = posterior.sum() - self.gamma
        else:
            sigma_denominator = posterior.sum() - 1

        try:
            dist.variance[0] = np.sqrt(sigma_numerator / sigma_denominator)
        except FloatingPointError:
            dist.variance[0] = 0.0001

        # alpha
        if (self.fixed == 0):
            self.alpha[dist_ind] = self.gamma / np.dot(dist.beta, dist.beta.T)

        dist.currentPosterior = posterior

        #print 'alpha', self.alpha
        #print 'gamma', self.gamma
        #print 'sigma', self.variance


def evaluateRegression(mix, data, type=2, train=[], sparse=[]):
    """ evaluation types
         0 - use y and x to decide
         1 - use only mixing coeficients
         2 - use x
         3 - user expected y
    """

    if isinstance(data, DataSet):
        dt = data.internalData
    elif hasattr(data, "__iter__"):
        dt = data
    else:
        raise TypeError, "Unknown/Invalid input to MStep."

    # First column of data set of the matrix
    y = dt[:, 0]

    predy = np.zeros((1, len(y)))[0]

    for c in mix.components:
        c.noise = 0.01

    [log_l, log_p] = mix.EStep(data)
    p = np.exp(log_l)
    pmax = np.zeros((len(p), len(p[0])))
    for i in range(len(p[0])):
        pmax[np.argmax(p[:, i]), i] = 1
        pmult = np.multiply(pmax, p)

    if type in [2, 3]:
        [log_l, log_p] = mix.EStep(train)
        ptrain = np.exp(log_l)
        pmaxtrain = []
        for i in range(len(ptrain[0])):
            pmaxtrain.append(np.argmax(ptrain[:, i]))
        d1 = np.array(data.dataMatrix)
        d2 = np.array(train.dataMatrix)
        if len(sparse) == 0:
            dist = scipy.spatial.distance.cdist(d1[:, 1:], d2[:, 1:], 'euclidean')
        else:
            # if sparse version was used than only relevant variables should be compared
            dist = np.zeros(len(d1), len(d2))
            for l in range(len(train)):
                dist[:, l] = scipy.spatial.distance.cdist(d1[:, sparse[pmaxtrain[l]]], d2[:, sparse[pmaxtrain[l]]], 'euclidean') / np.sqrt(len(sparse[pmaxtrain[l]]))
        labels = knn(d1[:, 1:], d2[:, 1:], pmaxtrain, 1)
        for i in range(len(p[0])):
            pmax[:, i] = 0
            pmax[labels[i], i] = 1

        if type == 3:
            meansy = []
            stdsy = []
            # estimate means and std
            for l, m in enumerate(mix.components):
                yprime = np.multiply(d2[:, 0], ptrain[:, l])
                meansy.append(sum(yprime) / sum(ptrain[:, l]))
                stdsy.append(np.multiply(yprime, d2[:, 0]) / sum(ptrain[:, l]) - meansy[l])

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
        predyaux = np.array(predyaux).T
        posteriorpred = np.array(posteriorpred).T
        for i, d in enumerate(data):
            predy = predyaux[i, np.argmax(posteriorpred[i, :])]

    # estimate pearson
    [r, p] = scipy.stats.pearsonr(y, predy)
    errorv = np.power(y - predy, 2)
    error = sum(errorv) / len(y)

    for i, m in enumerate(mix.components):
        list = [(j, data.sampleIDs[k]) for k, j in enumerate(errorv) if pmult[i, k] > 0]
        list.sort()
        genes.append([j for (k, j) in list])
    return [r, p, predy, y, means, stds, error, genes]


def knn(train, test, labels, k):
    labels = np.array(labels)
    if max(labels) > 0:
        dist = scipy.spatial.distance.cdist(train, test, 'euclidean')
        testlabel = []
        for i in range(len(train)):
            indices = np.argsort(dist[i, :])
            values = labels[indices[:k]]
            hist = np.zeros((max(values) + 1))
            for j in values:
                hist[j] = hist[j] + 1
            testlabel.append(np.argmax(hist))
    else:
        testlabel = len(train) * [0]
    return testlabel
