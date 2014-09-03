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
import numpy as np
from ..distributions.discrete import DiscreteDistribution
from ..distributions.multinomial import MultinomialDistribution
from ..util.errors import InvalidDistributionInput
from .prior import PriorDistribution
from ..util.candidate_group import CandidateGroup
from ..util.maths import sum_logs, matrix_sum_logs


class DirichletMixturePrior(PriorDistribution):
    """
    Mixture of Dirichlet distributions prior for multinomial data.
    """

    def __init__(self, G, M, pi, dComp):
        """
        @param G: number of components
        @param M: dimensions of component Dirichlets
        @param pi: mixture weights
        @param dComp: list of DirichletPrior distributions
        """
        assert len(dComp) == len(pi) == G
        for d in dComp:
            assert d.M == M

        self.G = G
        self.M = M
        self.pi = np.array(pi, dtype='Float64')
        self.log_pi = np.log(self.pi)  # assumes pi is not changed from the outside (XXX accesor functions ?)
        self.dComp = dComp
        self.constant_hyperparams = 1  # hyperparameters are constant

    def __str__(self):
        s = ['DirichletMixturePrior( G=' + str(self.G) + ' )']
        s.append('\tpi=' + str(self.pi) + '\n')
        for i in range(self.G):
            s.append('\t\t' + str(self.dComp[i]) + '\n')
        return ''.join(s)

    def __eq__(self, other):
        if not isinstance(other, DirichletMixturePrior):
            return False
        if self.G != other.G or self.M != other.M:
            return False
        if not np.alltrue(other.pi == self.pi):
            return False
        for i, d1 in enumerate(self.dComp):
            if not d1 == other.dComp[i]:
                return False
        return True

    def __copy__(self):
        cp_pi = copy.deepcopy(self.pi)
        cp_dC = [copy.deepcopy(self.dComp[i]) for i in range(self.G)]
        return DirichletMixturePrior(self.G, self.M, cp_pi, cp_dC)

    def pdf(self, m):
        if isinstance(m, MultinomialDistribution):  # XXX debug
            logp_list = np.zeros(self.G, dtype='Float64')
            for i in range(self.G):
                logp_list[i] = self.log_pi[i] + self.dComp[i].pdf(m)
            res = sum_logs(logp_list)
            return res

        elif type(m) == list:
            logp_mat = np.zeros((self.G, len(m)))
            for i in range(self.G):
                logp_mat[i, :] = self.dComp[i].pdf(m)

            for i in range(len(m)):  # XXX slow
                logp_mat[:, i] += self.log_pi

            res = matrix_sum_logs(logp_mat)
            return res
        else:
            raise TypeError


    def marginal(self, dist, posterior, data):
        suff_stat = np.zeros(self.M, dtype='Float64')
        if isinstance(dist, DiscreteDistribution):
            for i in range(self.M):
                i_ind = np.where(data == i)[0]
                suff_stat[i] = np.sum(posterior[i_ind])
        elif isinstance(dist, MultinomialDistribution):
            for i in range(self.M):
                suff_stat[i] = np.dot(data[:, i], posterior)
        else:
            raise TypeError, 'Invalid input ' + str(dist.__class__)

        res = 0.0
        for i in range(self.G):
            res += self.dComp[i].marginal(suff_stat) + np.log(self.pi[i])
        return res


    def posterior(self, dist):
        """
        Component membership posterior distribution of MultinomialDistribution 'dist'.

        @param dist: MultinomialDistribution object

        @return: numpy of length self.G containing the posterior of component membership
        """
        prior_post = np.array([dirich.pdf(dist) + self.log_pi[i] for i, dirich in enumerate(self.dComp)], dtype='Float64')
        log_sum = sum_logs(prior_post)
        prior_post -= log_sum

        prior_post = np.exp(prior_post)
        return prior_post

    def mapMStep(self, dist, posterior, data, mix_pi=None, dist_ind=None):
        suff_stat = np.zeros(self.M, dtype='Float64')
        if isinstance(dist, DiscreteDistribution):
            for i in range(self.M):
                i_ind = np.where(data == i)[0]
                suff_stat[i] = np.sum(posterior[i_ind])

        elif isinstance(dist, MultinomialDistribution):
            for i in range(self.M):
                suff_stat[i] = np.dot(data[:, i], posterior)
        else:
            raise TypeError, 'Invalid input ' + str(dist.__class__)

        # posterior of the given multinomial distribution 'dist'
        # with respect to the components of the Dirichlet mixture prior
        prior_post = self.posterior(dist)

        fix_flag = 0
        fix_phi = 1.0
        dsum = 0.0
        for i in range(self.M):
            if dist.parFix[i] == 1:  # checking for fixed entries in phi
                fix_flag = 1
                fix_phi -= dist.phi[i]
            else: # updating phi[i]
                e = np.zeros(self.G, dtype='Float64')
                for k in range(self.G):
                    e[k] = (suff_stat[i] + self.dComp[k].alpha[i] ) / ( sum(suff_stat) + self.dComp[k].alpha_sum  )

                est = np.dot(prior_post, e)
                dist.phi[i] = est
                dsum += est

        # re-normalizing parameter estimates if necessary
        if fix_flag:
            ind = np.where(dist.parFix == 0)[0]
            dist.phi[ind] = (dist.phi[ind] * fix_phi) / dsum

    def mapMStepMerge(self, group_list):

        new_dist = copy.copy(group_list[0].dist)

        prior_post = self.posterior(group_list[0].dist)
        pool_req_stat = copy.copy(group_list[0].req_stat)
        pool_post_sum = group_list[0].post_sum
        pool_pi_sum = group_list[0].pi_sum

        for i in range(1, len(group_list)):
            pool_req_stat += group_list[i].req_stat
            pool_post_sum += group_list[i].post_sum
            pool_pi_sum += group_list[i].pi_sum

            prior_post += self.posterior(group_list[i].dist) * group_list[i].pi_sum

        prior_post = prior_post / pool_pi_sum

        assert 1.0 - sum(prior_post) < 1e-10, str(prior_post) + ' , ' + str(sum(prior_post)) + ', ' + str(1.0 - sum(prior_post))  # XXX debug

        fix_flag = 0
        fix_phi = 1.0
        dsum = 0.0
        for i in range(self.M):
            if new_dist.parFix[i] == 1:  # assumes parFix is consistent
                fix_flag = 1
                fix_phi -= new_dist.phi[i]
            else: # updating phi[i]
                e = np.zeros(self.G, dtype='Float64')
                for k in range(self.G):
                    e[k] = (pool_req_stat[i] + self.dComp[k].alpha[i]) / ( pool_post_sum + self.dComp[k].alpha_sum  )

                est = np.dot(prior_post, e)
                new_dist.phi[i] = est
                dsum += est

                #print i,est

        # re-normalizing parameter estimates if necessary
        if fix_flag:
            ind = np.where(new_dist.parFix == 0)[0]
            new_dist.phi[ind] = (new_dist.phi[ind] * fix_phi) / dsum

        return CandidateGroup(new_dist, pool_post_sum, pool_pi_sum, pool_req_stat)


    def isValid(self, x):
        if not isinstance(x, MultinomialDistribution):
            raise InvalidDistributionInput, "DirichletMixturePrior: " + str(x)

        if x.M != self.M:
            raise InvalidDistributionInput, "DirichletMixturePrior: unequal dimensions " + str(x.M) + " != " + str(self.M)


    def flatStr(self, offset):
        offset += 1
        s = "\t" * offset + ";DirichMixPrior;" + str(self.G) + ";" + str(self.M) + ";" + str(self.pi.tolist()) + "\n"
        for d in self.dComp:
            s += d.flatStr(offset)
        return s


