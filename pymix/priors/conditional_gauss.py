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
from ..distributions.conditional_gauss import ConditionalGaussDistribution
from ..util import mixextend
from ..util.errors import InvalidDistributionInput
from .prior import PriorDistribution


class ConditionalGaussPrior(PriorDistribution):
    """
    Prior over ConditionalGaussDistribution. Assumes Normal prior over the covariance parameters w.

    """

    def __init__(self, nr_comps, p):
        """
            Constructor

            @param nr_comps: number of components in the mixture the prior is applied to
            @param p:  number of features in the ConditionalGaussDistribution the prior is applied to
        """

        self.constant_hyperparams = 0  # hyperparameters are updated as part of the mapEM
        self.nr_comps = nr_comps    # number of components in the mixture the prior is applied to
        self.dimension = p   # number of features in the ConditionalGaussDistribution the prior is applied to

        # no initial value needed, is updated as part of EM in updateHyperparameters
        self.beta = np.zeros((self.nr_comps, self.dimension))
        self.nu = np.zeros((self.nr_comps, self.dimension))

        # XXX initialization of sufficient statistics, necessary for hyperparameter updates
        self.post_sums = np.zeros(self.nr_comps)
        self.var = np.zeros((self.nr_comps, self.dimension))
        self.cov = np.zeros((self.nr_comps, self.dimension))
        self.mean = np.zeros((self.nr_comps, self.dimension))


    def __str__(self):
        return 'ConditionalGaussPrior(beta=' + str(self.beta) + ')'


    def pdf(self, d):
        if type(d) == list:
            N = np.sum(self.post_sums)

            res = np.zeros(len(d))
            for i in range(len(d)):
                for j in range(1, d[i].dimension):
                    pid = d[i].parents[j]
                    res[i] += (1.0 / self.cov[i, j] ** 2) / (self.nu[i, j] * (self.post_sums[i] / N))
                    res[i] += np.log(mixextend.wrap_gsl_ran_gaussian_pdf(
                        0.0,
                        math.sqrt((self.beta[i, j] * self.cov[i, j] ** 2) / (self.var[i, pid] * (self.post_sums[i] / N) )),
                        [d[i].w[j]]
                    ))
        else:
            raise TypeError, 'Invalid input ' + str(type(d))

        return res


    def posterior(self, m, x):
        raise NotImplementedError, "Needs implementation"

    def marginal(self, x):
        raise NotImplementedError, "Needs implementation"


    def mapMStep(self, dist, posterior, data, mix_pi=None, dist_ind=None):
        assert not dist_ind == None # XXX debug

        post_sum = np.sum(posterior)
        self.post_sums[dist_ind] = post_sum

        # checking for valid posterior: if post_sum is zero, this component is invalid
        # for this data set
        if post_sum != 0.0:

            # reestimate mu
            for j in range(dist.dimension):
                # computing ML estimates for w and sigma
                self.mean[dist_ind, j] = np.dot(posterior, data[:, j]) / post_sum
                #self.var[dist_ind,j] = np.dot(posterior, (data[:,j] - dist.mean[j])**2 ) / post_sum
                self.var[dist_ind, j] = np.dot(posterior, (data[:, j] - self.mean[dist_ind, j]) ** 2) / post_sum

                if j > 0:  # w[0] = 0.0 is fixed
                    pid = dist.parents[j]
                    self.cov[dist_ind, j] = np.dot(posterior, (data[:, j] - self.mean[dist_ind, j]) * (data[:, pid] - self.mean[dist_ind, pid])) / post_sum

                    # update hyperparameters beta
                    self.beta[dist_ind, j] = post_sum / ( (( self.var[dist_ind, j] * self.var[dist_ind, pid]) / self.cov[dist_ind, j] ** 2) - 1 )

                    # update hyperparameters nu
                    self.nu[dist_ind, j] = - post_sum / (2 * dist.variance[j] ** 2)

                    # update regression weights
                    dist.w[j] = self.cov[dist_ind, j] / (dist.variance[pid] ** 2 * (1 + self.beta[dist_ind, j] ** -1 ) )

                    # update standard deviation
                    dist.variance[j] = math.sqrt(self.var[dist_ind, j] - (dist.w[j] ** 2 * dist.variance[pid] ** 2 * (1 + (1.0 / self.beta[dist_ind, j])) ) - self.nu[dist_ind, j] ** -1)
                    # update means
                    dist.mean[j] = self.mean[dist_ind, j] #- (dist.w[j] * self.mean[dist_ind,pid])

                else:
                    dist.variance[j] = math.sqrt(self.var[dist_ind, j])  # root variance
                    dist.mean[j] = self.mean[dist_ind, j]


    def updateHyperparameters(self, dists, posterior, data):
        """
        Updates the hyperparamters in an empirical Bayes fashion as part of the EM parameter estimation.

        """
        assert len(dists) == posterior.shape[0]  # XXX debug

        # update component-specific hyperparameters
        for i in range(self.nr_comps):
            self.post_sums[i] = np.sum(posterior[i, :])
            for j in range(self.dimension):
                #  var_j = np.dot(posterior, (data[:,j] - dist.mean[j])**2 ) / post_sum
                self.var[i, j] = np.dot(posterior[i, :], (data[:, j] - dists[i].mean[j]) ** 2) / self.post_sums[i]

                if j > 0: # feature 0 is root by convention
                    pid_i_j = dists[i].parents[j]
                    self.cov[i, j] = np.dot(posterior[i, :], (data[:, j] - dists[i].mean[j]) * (data[:, pid_i_j] - dists[i].mean[pid_i_j])) / self.post_sums[i]
                    self.beta[i, j] = self.post_sums[i] / ( (( self.var[i, j] * self.var[i, pid_i_j]) / self.cov[i, j] ** 2) - 1 )
                    self.nu[i, j] = - self.post_sums[i] / (2 * dists[i].variance[j] ** 2)


    def isValid(self, x):
        if not isinstance(x, ConditionalGaussDistribution):
            raise InvalidDistributionInput, "ConditionalGaussPrior: " + str(x)


