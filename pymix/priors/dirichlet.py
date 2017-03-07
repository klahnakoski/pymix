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
import scipy
from ..distributions.discrete import DiscreteDistribution
from ..distributions.multinomial import MultinomialDistribution
from ..util import mixextend
from ..util.candidate_group import CandidateGroup
from ..util.errors import InvalidPosteriorDistribution, InvalidDistributionInput
from .prior import PriorDistribution


class DirichletPrior(PriorDistribution):  # DirichletDistribution,
    """
    Dirichlet distribution as Bayesian prior for MultinomialDistribution and derived .
    """

    def __init__(self, M, alpha):
        """
        @param M: number of dimensions
        @param alpha: distribution parameters
        """
        assert M == len(alpha)
        #for a in alpha:
        #    assert a > 0.0, "Invalid parameter."

        self.M = M
        self.alpha = np.array(alpha, dtype='Float64')
        self.alpha_sum = np.sum(alpha) # assumes alphas to remain constant !
        self.dimension = M
        self.suff_p = M
        self.freeParams = M

        self.constant_hyperparams = 1  # hyperparameters are constant

    def __copy__(self):
        cp_alpha = copy.deepcopy(self.alpha)
        return DirichletPrior(self.M, cp_alpha)

    def __str__(self):
        return "DirichletPrior: " + str(self.alpha)

    def __eq__(self, other):
        if isinstance(other, DirichletPrior):
            if self.M == other.M and np.alltrue(self.alpha == other.alpha):
                return True
            else:
                return False
        else:
            return False

    def sample(self):
        """
        Samples from Dirichlet distribution
        """
        phi = scipy.random.dirichlet(self.alpha, self.M)

        d = DiscreteDistribution(self.M, phi)
        return d


    def pdf(self, m):

        # XXX should be unified ...
        if isinstance(m, MultinomialDistribution):
            # use GSL implementation
            #res = pygsl.rng.dirichlet_lnpdf(self.alpha,[phi])[0] XXX
            try:
                res = mixextend.wrap_gsl_dirichlet_lnpdf(self.alpha, [m.phi])
            except ValueError:
                print m
                print self
                raise
            return res[0]

        elif isinstance(m, list):
            in_l = [d.phi for d in m]
            # use GSL implementation
            res = mixextend.wrap_gsl_dirichlet_lnpdf(self.alpha, in_l)
            return res
        else:
            raise TypeError

    def posterior(self, m, x):
        """
        Returns the posterior for multinomial distribution 'm' for multinomial count data 'x'
        The posterior is again Dirichlet.
        """
        assert isinstance(m, MultinomialDistribution)
        res = np.ones(len(x), dtype='Float64')
        for i, d in enumerate(x):
            post_alpha = self.alpha + d
            res[i] = np.array(mixextend.wrap_gsl_dirichlet_lnpdf(post_alpha, [m.phi]))

        return res

    def marginal(self, x):
        """
        Returns the log marginal likelihood of multinomial counts 'x' (sufficient statistics)
        with Dirichlet prior 'self' integrated over all parameterizations of the multinomial.
        """
        # XXX should be eventually replaced by more efficient implementation
        # in Dirchlet mixture prior paper (K. Sjoelander,Karplus,..., D.Haussler)

        x_sum = sum(x)

        term1 = mixextend.wrap_gsl_sf_lngamma(self.alpha_sum) - mixextend.wrap_gsl_sf_lngamma(self.alpha_sum + x_sum)
        term2 = 0.0
        for i in range(self.dimension):
            term2 += mixextend.wrap_gsl_sf_lngamma(self.alpha[i] + x[i]) - mixextend.wrap_gsl_sf_lngamma(self.alpha[i])

        res = term1 + term2
        return res

    def mapMStep(self, dist, posterior, data, mix_pi=None, dist_ind=None):
        # Since DiscreteDistribution is a special case of MultinomialDistribution
        # the DirichletPrior applies to both. Therefore we have to distinguish the
        # two cases here. The cleaner alternative would be to derive specialized prior
        # distributions but that would be unnecessarily complicated at this point.
        if isinstance(dist, DiscreteDistribution):
            ind = np.where(dist.parFix == 0)[0]
            fix_phi = 1.0
            dsum = 0.0
            for i in range(dist.M):
                if dist.parFix[i] == 1:
                    fix_phi -= dist.phi[i]
                    continue
                else:
                    i_ind = np.where(data == i)[0]
                    est = np.sum(posterior[i_ind]) + self.alpha[i] - 1
                    dist.phi[i] = est
                    dsum += est

            # normalizing parameter estimates
            dist.phi[ind] = (dist.phi[ind] * fix_phi) / dsum
        elif isinstance(dist, MultinomialDistribution):

            fix_phi = 1.0
            dsum = 0.0
            # reestimating parameters
            for i in range(dist.M):
                if dist.parFix[i] == 1:
                    #print "111"
                    fix_phi -= dist.phi[i]
                    continue
                else:
                    est = np.dot(data[:, i], posterior) + self.alpha[i] - 1
                    dist.phi[i] = est
                    dsum += est

            if dsum == 0.0:
                raise InvalidPosteriorDistribution, "Invalid posterior in MStep."

            ind = np.where(dist.parFix == 0)[0]
            # normalzing parameter estimates
            dist.phi[ind] = (dist.phi[ind] * fix_phi) / dsum

        else:
            raise TypeError, 'Invalid input ' + str(dist.__class__)


    def mapMStepMerge(self, group_list):
        #XXX only for DiscreteDistribution for now, MultinomialDistribution to be done
        assert isinstance(group_list[0].dist, DiscreteDistribution), 'only for DiscreteDistribution for now'

        pool_req_stat = copy.copy(group_list[0].req_stat)
        pool_post_sum = group_list[0].post_sum
        pool_pi_sum = group_list[0].pi_sum

        for i in range(1, len(group_list)):
            pool_req_stat += group_list[i].req_stat
            pool_post_sum += group_list[i].post_sum
            pool_pi_sum += group_list[i].pi_sum

        new_dist = copy.copy(group_list[0].dist)  # XXX copy necessary ?

        ind = np.where(group_list[0].dist.parFix == 0)[0]
        fix_phi = 1.0
        dsum = 0.0
        for i in range(group_list[0].dist.M):
            if group_list[0].dist.parFix[i] == 1:
                assert group_list[1].dist.parFix[i] == 1  # incomplete consistency check of parFix (XXX)

                fix_phi -= new_dist.phi[i]
                continue
            else:
                est = pool_req_stat[i] + self.alpha[i] - 1
                new_dist.phi[i] = est

                dsum += est

        # normalizing parameter estimates
        new_dist.phi[ind] = (new_dist.phi[ind] * fix_phi) / dsum

        return CandidateGroup(new_dist, pool_post_sum, pool_pi_sum, pool_req_stat)


    def mapMStepSplit(self, toSplitFrom, toBeSplit):
        #XXX only for DiscreteDistribution for now, MultinomialDistribution to be done
        assert isinstance(toSplitFrom.dist, DiscreteDistribution), 'only for DiscreteDistribution for now'

        split_req_stat = copy.copy(toSplitFrom.req_stat)
        split_req_stat -= toBeSplit.req_stat

        split_post_sum = toSplitFrom.post_sum - toBeSplit.post_sum
        split_pi_sum = toSplitFrom.pi_sum - toBeSplit.pi_sum

        new_dist = copy.copy(toSplitFrom.dist)  # XXX copy necessary ?

        ind = np.where(toSplitFrom.dist.parFix == 0)[0]
        fix_phi = 1.0
        dsum = 0.0
        for i in range(toSplitFrom.dist.M):
            if toSplitFrom.dist.parFix[i] == 1:

                fix_phi -= new_dist.phi[i]
                continue
            else:
                est = split_req_stat[i] + self.alpha[i] - 1
                new_dist.phi[i] = est
                dsum += est

        # normalizing parameter estimates
        new_dist.phi[ind] = (new_dist.phi[ind] * fix_phi) / dsum

        return CandidateGroup(new_dist, split_post_sum, split_pi_sum, split_req_stat)


    def isValid(self, x):
        if not isinstance(x, MultinomialDistribution):
            raise InvalidDistributionInput, "in DirichletPrior: " + str(x)
        else:
            if self.M != x.M:
                raise InvalidDistributionInput, "in DirichletPrior: " + str(x)

    def flatStr(self, offset):
        offset += 1
        return "\t" * offset + ";DirichletPr;" + str(self.M) + ";" + str(self.alpha.tolist()) + "\n"



