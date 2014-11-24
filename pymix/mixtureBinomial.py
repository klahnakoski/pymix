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


"""
Mixtures of Bionomials and Related Distributions
"""
from mixture import *

import numpy as np
import copy
from pymix.distributions.prob import ProbDistribution
from pymix.distributions.product import ProductDistribution
from pymix.models.mixture import MixtureModel
from pymix.util.dataset import DataSet
from pymix.util.errors import InvalidPosteriorDistribution, InvalidDistributionInput, ConvergenceFailureEM


class BernoulliDistribution(ProbDistribution):
    """
    Bernoulli Distribution

    """

    def __init__(self, theta):
        """
        Constructor

        @param theta
        """
        self.dimension = 1
        self.suff_p = 1
        self.freeParams = 1
        self.theta = theta


    def __eq__(self, other):
        res = False
        if isinstance(other, BernoulliDistribution):
            if (np.allclose(other.theta, self.theta)):
                res = True
        return res

    def __copy__(self):
        return BernoulliDistribution(copy.deepcopy(self.theta))

    def __str__(self):
        return "Binomial:  [" + str(self.theta) + " " + str(self.dimension) + "]"


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
        # this pdf is a logit representation of a bionomial (with large n???)
        res = np.power(self.theta, x) * np.power(1 - self.theta, 1 - x)
        return np.log(res)

    def sample(self):
        return np.random.binomial(1, self.theta)

    def sampleSet(self, nr):
        res = np.zeros(nr, dtype='Float64')

        for i in range(nr):
            res[i] = self.sample()

        return res

    def sufficientStatistics(self, posterior, data):
        """
        Returns sufficient statistics for a given data set and posterior. In case of the binomial distribution
        this is the dot product of a vector of component membership posteriors with the data and mean value of the variable

        @param posterior: numpy vector of component membership posteriors
        @param data: numpy vector holding the data

        @return: list with dot(posterior, data) and sum(data)
        """
        nu0 = np.dot(posterior, data)[0],
        nu = np.sum(data)

        return [nu, nu0]


    def MStep(self, posterior, data, mix_pi=None):
        # data has to be reshaped for parameter estimation
        if isinstance(data, DataSet):
            x = data.internalData[:, 0]
        elif isinstance(data, np.ndarray):
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

            new_theta = np.dot(posterior, x) / post_sum
        else:
            raise InvalidPosteriorDistribution, "Sum of posterior is zero: " + str(self) + " has zero likelihood for data set."


    def isValid(self, x):
        try:
            float(x)
        except (ValueError):
            #print "Invalid data: ",x,"in BinomialDistribution."
            raise InvalidDistributionInput, "\n\tInvalid data: " + str(x) + " in BinomialDistribution."

    def formatData(self, x):
        if isinstance(x, list) and len(x) == 1:
            x = x[0]
        self.isValid(x)  # make sure x is valid argument
        return [self.dimension, [x]]


    def flatStr(self, offset):
        offset += 1
        return "\t" * +offset + ";Bernouli;" + str(self.theta) + "\n"

    def posteriorTraceback(self, x):
        return self.pdf(x)

    def merge(self, dlist, weights):
        raise DeprecationWarning, 'Part of the outdated structure learning implementation.'


class BinomialRegularizedDistribution(ProbDistribution):
    """
    Regularized Binomial Distribution

    This is the implementation of the Tsuda et al paper on a bionomial
    distribution with regularization.

    Koji Tsuda, Taku Kudo: Clustering graphs by weighted substructure
    mining. ICML 2006: 953-960

    """

    def __init__(self, theta, lambd):
        """
        Constructor

        @param theta
        """
        self.dimension = 1
        self.suff_p = 1
        self.freeParams = 1
        self.theta = theta
        self.lambd = lambd
        self.flag_selected = 1
        self.nu = 0
        #self.theta0 = theta0
        #self.lambdal = lambdal


    def __eq__(self, other):
        res = False
        if isinstance(other, BinomialRegularizedDistribution):
            if (np.allclose(other.theta, self.theta)) and (np.allclose(other.lambd, self.lambd)):
                res = True
        return res

    def __copy__(self):
        return BinomialRegularizedDistribution(copy.deepcopy(self.theta), copy.deepcopy(self.lambd))

    def __str__(self):
        return "Binomial:  [" + str(self.theta) + " " + str(self.lambd) + "  " + str(self.flag_selected) + "]"


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
        # this pdf is a logit representation of a bionomial (with large n???)

        dem = 1 + np.exp(self.theta);
        #print self.theta,dem
        try:
            res = np.exp(x * self.theta);
        except FloatingPointError:
            print self.theta
            raise FloatingPointError
            #print res
        res = res / dem
        #print "pdf", self.theta, np.log(res)
        return np.log(res)

    def sample(self):
        # XXX - not ready ... use scipy???
        return 0;


    def sampleSet(self, nr):
        res = np.zeros(nr, dtype='Float64')

        for i in range(nr):
            res[i] = self.sample()

        return res

    def sufficientStatistics(self, posterior, data):
        """
        Returns sufficient statistics for a given data set and posterior. In case of the binomial distribution
        this is the dot product of a vector of component membership posteriors with the data and mean value of the variable

        @param posterior: numpy vector of component membership posteriors
        @param data: numpy vector holding the data

        @return: list with dot(posterior, data) and sum(data)
        """
        #nu0 =  np.dot(posterior, data)[0],
        #nu = np.sum(data)
        #return [nu,nu0]
        return 0


    def MStep(self, posterior, data, mix_pi=None):
        # data has to be reshaped for parameter estimation
        if isinstance(data, DataSet):
            x = data.internalData[:, 0]
        elif isinstance(data, np.ndarray):
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
            # computing ML estimates for nu
            new_nu = min(max(np.dot(posterior, x) / post_sum, 0.000000001), 0.9999999999)# eq 15
            new_nu0 = min(max(np.sum(x) / nr, 0.000000001), 0.9999999999) # eq 13 plus pseudo count
            new_theta0 = np.log(new_nu0) - np.log(1 - new_nu0); # eq (12)
            #print self.lambd
            new_lambdl = self.lambd * nr / post_sum;

            #print new_theta0,new_nu0,new_nu

            #equation (17)
            #print new_nu,new_nu0,new_lambdl
            if (new_nu >= new_nu0 + new_lambdl):
                new_theta = np.log(new_nu - new_lambdl) - np.log(1 - (new_nu - new_lambdl))
                self.flag_selected = 1;
            elif (new_nu <= new_nu0 - new_lambdl):
                new_theta = np.log(new_nu + new_lambdl) - np.log(1 - (new_nu - new_lambdl))
                self.flag_selected = 1;
            else:
                new_theta = new_theta0;
                self.flag_selected = 0;
                #print "non selected", self.flag_selected
        else:
            raise InvalidPosteriorDistribution, "Sum of posterior is zero: " + str(self) + " has zero likelihood for data set."

        # assigning updated parameter values
        #self.lambdl = new_lambdl
        #self.theta_0 = new_theta_0
        #print new_theta, new_lambdl, new_nu, new_nu0
        self.theta = new_theta
        self.nu = new_nu


    def isValid(self, x):
        try:
            float(x)
        except (ValueError):
            #print "Invalid data: ",x,"in BinomialDistribution."
            raise InvalidDistributionInput, "\n\tInvalid data: " + str(x) + " in BinomialDistribution."

    def formatData(self, x):
        if isinstance(x, list) and len(x) == 1:
            x = x[0]
        self.isValid(x)  # make sure x is valid argument
        return [self.dimension, [x]]


    def flatStr(self, offset):
        offset += 1
        return "\t" * +offset + ";Binomial;" + str(self.theta) + ";" + str(self.lambd) + "\n"

    def posteriorTraceback(self, x):
        return self.pdf(x)

    def merge(self, dlist, weights):
        raise DeprecationWarning, 'Part of the outdated structure learning implementation.'

    def selectionScore(self):
        # Evaluation of features
        # Eq. 26 of tsuda (nu is proportional to it)
        if self.flag_selected:
            return self.nu
        else:
            return 0


class MixtureModelFeatureSelection(MixtureModel):
    """
    This class implements functions for mixture models with
    distributions with feature selection.

    This class is restricture for distributions using product dist.

    """

    def selectedFeatures(self):
        # for all dimensions
        flags = np.zeros(len(self.components[0]))
        for i in range(len(self.components[0])):
            for j in range(self.G):
                flags[i] = flags[i] or self.components[j][i].flag_selected
        return flags

    def rankFeatures(self, tissues=1):
        # if multiple tisses, check sub-graph and tissue
        dim = len(self.components[0])
        features = dim / tissues
        scores = []
        for i in range(dim):
            #scoreaux = 0
            scoreaux = [self.components[j][i].selectionScore() for j in range(self.G)]
            argmax = np.argmax(scoreaux)
            max = np.max(scoreaux)
            scores.append((max, argmax + 1, i / features + 1, i % features + 1))
        scores.sort()
        scores.reverse()
        return scores

    def __copy__(self):
        copy_components = []
        copy_pi = copy.deepcopy(self.pi)
        copy_compFix = copy.deepcopy(self.compFix)
        for i in range(self.G):
            copy_components.append(copy.deepcopy(self.components[i]))

        copy_model = MixtureModelFeatureSelection(self.G, copy_pi, copy_components, compFix=copy_compFix)
        copy_model.nr_tilt_steps = self.nr_tilt_steps
        copy_model.suff_p = self.suff_p
        copy.identFlag = self.identFlag

        if self.struct:
            copy_model.initStructure()

            copy_leaders = copy.deepcopy(self.leaders)
            copy_groups = copy.deepcopy(self.groups)

            copy_model.leaders = copy_leaders
            copy_model.groups = copy_groups

        return copy_model


def printFeatureResults(fileName, features):
    """
    Outputs the results of the feature selection criteria from
    function rankFeatures to a flat file
    """

    file = open(fileName, 'w')
    file.write('Feature\tTissue\tGroup\tIndex\n')
    for (r, g, t, f) in features:
        file.write(str(f) + '\t' + str(t) + '\t' + str(g) + '\t' + str(r) + '\n')
    file.close()


def estimateWithReplication(mixture, data, repetitions, iterations, stopCriteria):
    """
    Function replicating em estimation and returning maximum likelihood replicate
    """
    mixtureBest = []
    max = -9999999999999999.99;
    for j in range(repetitions):
        try:
            maux = copy.copy(mixture)
            maux.modelInitialization(data)
            #print maux
            maux.EM(data, iterations, stopCriteria)
            [l, log_l] = maux.EStep(data)
            if log_l > max:
                max = log_l
                mixtureBest = maux
        except  ConvergenceFailureEM:
            pass
    if mixtureBest == []:
        raise ConvergenceFailureEM, "Convergence failed."
    return mixtureBest


if __name__ == '__main__':

    from pymix import mixture
    import scipy.stats



    # repeat 15 random samplings and estimate accuracies of clustering

    datasets = 10 # no o datasets
    dim = 100 # no of random dimesions
    real = 10 # no of real dimensions
    # varying feature selection variable (lambda)
    laux = range(0, 10)
    laux = np.array(laux) * 0.005

    resall = []
    specall = []
    sensall = []
    featuresSelectedAll = []

    for i in range(datasets):

        featuresTrue = []

        # for simple testing, I sample data from a bernouli distribition with arbitrarly large dimension

        # group 1
        # first random dimensions
        components = []
        for i in range(dim):
            components.append(BernoulliDistribution(0.2))
            featuresTrue.append(0)

        # few real dimensions
        for i in range(dim, dim + real):
            components.append(BernoulliDistribution(0.1))
            featuresTrue.append(1)

        c1 = ProductDistribution(components)


        # group 2
        # first random dimensions
        components = []
        for i in range(dim):
            components.append(BernoulliDistribution(0.2))
            # few real dimensions
        for i in range(dim, dim + real):
            components.append(BernoulliDistribution(0.9))

        c2 = ProductDistribution(components)

        # creating the mixture mode model
        pi = [0.5, 0.5]
        m = MixtureModel(2, pi, [c1, c2])

        # sample data
        data = m.sampleDataSet(200)
        # get real classes
        creal = m.classify(data, silent=1)

        res = []
        sens = []
        spec = []
        featuresSelected = []


        # test now all selections of lambda

        for l in laux:

            # starting the mixture model

            component = []
            for i in range(dim + real):
                component.append(BinomialRegularizedDistribution(0, l))
            c1 = ProductDistribution(component)
            component = []
            for i in range(dim + real):
                component.append(BinomialRegularizedDistribution(0, l))
            c2 = ProductDistribution(component)
            # creating the model
            pi = [0.4, 0.6]

            try:

                m2 = MixtureModelFeatureSelection(2, pi, [c1, c2])

                m2 = estimateWithReplication(m2, data, 5, 30, 0.1)
                c = m2.classify(data)
                res.append(min(float(sum(c == creal)) / len(creal), 1 - float(sum(c == creal)) / len(creal)))

                # keeping selected features
                featuresSelected.append(m2.rankFeatures())

                # checking feature selection
                # as we have only two, we simply need to check of the components
                total = 0
                rightfeature = 0

                featuresFlags = m2.selectedFeatures()

                total = sum(featuresFlags)
                rightfeature = sum((featuresTrue + featuresFlags) == 2)

                print featuresFlags, rightfeature, sum(featuresTrue), float(rightfeature) / sum(featuresTrue)

                #for (i,dist) in enumerate(c1):
                #  total = total + dist.flag_selected;
                #  if featuresTrue[i] == 1 and (dist.flag_selected == 1):
                #      rightfeature = 1 + rightfeature;
                #         #   elif featuresTrue[i] == 1 and (dist.flag_selected == 0):
                #         #       missedFeature = 1 + missedFeature;


                sens.append(float(rightfeature) / sum(featuresTrue));
                spec.append(float(total) / len(featuresTrue));


            except  ConvergenceFailureEM:
                print "Convergence error "
                res.append('nan')
                spec.append('nan')
                sens.append('nan')
                featuresSelected.append([])

        resall.append(res)
        sensall.append(sens)
        specall.append(spec)
        featuresSelectedAll.append(featuresSelected)

    print  scipy.stats.stats.nanmean(resall, 0)
    print  scipy.stats.stats.nanstd(resall, 0)
    print  scipy.stats.stats.nanmean(specall, 0)
    print  scipy.stats.stats.nanmean(sensall, 0)

    for fs in featuresSelectedAll:
        for f in fs:
            print f


