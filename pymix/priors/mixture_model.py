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
from pymix.util.logs import Log
from ..util.errors import InvalidDistributionInput

from ..models.mixture import MixtureModel
from .dirichlet import DirichletPrior
from .prior import PriorDistribution
from .product import ProductDistributionPrior
from ..util.candidate_group import CandidateGroup


class MixtureModelPrior(PriorDistribution):
    """
    Mixture model prior.
    """

    def __init__(self, structPrior, nrCompPrior, piPrior, compPrior):
        """
        Constructor

        @param structPrior: hyperparameter over structure complexity (0< structPrior < 1), stored on log scale internally
        @param nrCompPrior:  hyperparameter over number of components (0< nrCompPrior < 1), stored on log scale internally
        @param piPrior: DirichletPrior object
        @param compPrior: list of PriorDistribution objects
        """
        assert isinstance(piPrior, DirichletPrior)
        self.piPrior = piPrior

        for p in compPrior:
            assert isinstance(p, PriorDistribution)

        self.compPrior = ProductDistributionPrior(compPrior)
        self.structPrior = np.log(structPrior)
        self.nrCompPrior = np.log(nrCompPrior)

        self.constant_hyperparams = 1
        self.hp_update_indices = []
        for i in range(len(self.compPrior.priorList)):
            if self.compPrior.priorList[i].constant_hyperparams == 0:
                self.hp_update_indices.append(i)

        if len(self.hp_update_indices) > 0:
            self.constant_hyperparams = 0  # there is at least one prior which requires hyper parameter updates


    def __str__(self):
        outstr = "MixtureModelPrior: \n"
        outstr += "num_prior=" + str(len(self.compPrior.priorList)) + "\n"
        outstr += "structPrior =" + str(self.structPrior) + "\n"
        outstr += "nrCompPrior =" + str(self.nrCompPrior) + "\n"
        outstr += "  piPrior = " + str(self.piPrior) + "\n"
        for dist in self.compPrior:
            outstr += "    " + str(dist) + "\n"

        return outstr


    def __eq__(self, other):
        if not isinstance(other, MixtureModelPrior):
            return False

        if self.structPrior != other.structPrior or self.nrCompPrior != other.nrCompPrior:
            return False
        if not self.piPrior == other.piPrior:
            return False

        if not self.compPrior == other.compPrior:
            return False

        return True


    def __copy__(self):
        cp_pi = copy.copy(self.piPrior)
        cp_comp = []
        for i in range(len(self.compPrior.priorList)):
            cp_comp.append(copy.copy(self.compPrior[i]))

        # initialise copy with dummy values for .structPrior, .nrCompPrior
        cp_pr = MixtureModelPrior(0.0, 0.0, cp_pi, cp_comp)
        # set values of hyperparameters
        cp_pr.structPrior = self.structPrior
        cp_pr.nrCompPrior = self.nrCompPrior

        return cp_pr


    def pdf(self, mix):
        #assert isinstance(mix,MixtureModel), str(mix.__class__)
        #assert len(self.compPrior) == mix.components[0].dist_nr
        temp = DiscreteDistribution(mix.G, mix.pi)
        res = self.piPrior.pdf(temp)

        # XXX fixed components do not contribute to the prior (HACK)
        # this is needed if we use mixtures of mixtures to model missing data XXX
        if sum(mix.compFix) > 0:
            for j in range(len(mix.components[0])):
                for l in range(mix.G):
                    if mix.compFix[l] != 2:
                        p = self.compPrior[j].pdf(mix.components[l][j])
                        res += p
        else:
            # operate column wise on mix.components
            for j in range(len(self.compPrior.priorList)):

                if not isinstance(mix.components[0].distList[j], MixtureModel):
                    d_j = [mix.components[i].distList[j] for i in range(mix.G)]
                    res += np.sum(self.compPrior[j].pdf(d_j))
                else:
                    for i in range(mix.G):
                        res += self.compPrior[j].pdf(mix.components[i][j])


        # prior over number of components
        res += self.nrCompPrior * mix.G

        # prior over number of distinct groups
        if mix.struct:
            for j in range(len(mix.components[0].distList)):
                res += self.structPrior * len(mix.leaders[j])
        else:
            for j in range(len(mix.components[0].distList)):
                res += self.structPrior * mix.G

        if np.isnan(res):
        # uncomment code below for detailed information where the nan value came from (DEBUG)
        #            print '--------------------------------'
        #            print 'MixtureModelPrior.pdf ',res
        #            temp = DiscreteDistribution(mix.G,mix.pi)
        #            print '   MixPrior.pdf.pi:',temp,self.piPrior.pdf(temp)
        #            if sum(mix.compFix) > 0:
        #                for j in range(len(mix.components[0].distList)):
        #                    for l in range(mix.G):
        #                        if mix.compFix[l] != 2:
        #                            p = self.compPrior[j].pdf( mix.components[l][j] )
        #                            print l,j,p
        #            else:
        #                for l in range(mix.G):
        #                    #print     mix.components[l]
        #                    print '    comp ',l,':',self.compPrior.pdf(mix.components[l])
        #                    for o,d in enumerate(self.compPrior):
        #                        print '       dist',o,mix.components[l][o],':',d.pdf(mix.components[l][o])
        #
        #            print 'nrCompPrior=',  self.nrCompPrior * mix.G
        #
        #            # prior over number of distinct groups
        #            if mix.struct:
        #                for j in range(len(mix.components[0].distList)):
        #                    print '    struct:',self.structPrior * len(mix.leaders[j])
        #            else:
        #                for j in range(len(mix.components[0].distList)):
        #                  print '    struct:', self.structPrior * mix.G
            raise ValueError, 'nan result in MixtureModelPrior.pdf'
        return res


    # mapMStep is used for parameter estimation of lower hierarchy mixtures
    def mapMStep(self, dist, posterior, data, mix_pi=None, dist_ind=None):
        dist.mapEM(data, self, 1, 0.1, silent=True, mix_pi=mix_pi, mix_posterior=posterior)

    def updateHyperparameters(self, dists, posterior, data):

        assert self.constant_hyperparams == 0
        assert isinstance(dists, MixtureModel) # XXX debug

        for j in self.hp_update_indices:
            d_j = [dists.components[i].distList[j] for i in range(dists.G)]
            self.compPrior[j].updateHyperparameters(d_j, posterior, data.getInternalFeature(j))


    def flatStr(self, offset):
        offset += 1
        s = "\t" * offset + ";MixPrior;" + str(len(self.compPrior.priorList)) + ";" + str(np.exp(self.structPrior)) + ";" + str(np.exp(self.nrCompPrior)) + "\n"

        s += self.piPrior.flatStr(offset)
        for d in self.compPrior:
            s += d.flatStr(offset)

        return s

    def posterior(self, dist):
        raise NotImplementedError

    def isValid(self, m):
        if not isinstance(m, MixtureModel):
            raise InvalidDistributionInput, "MixtureModelPrior: " + str(m)
        else:
            if self.piPrior.M != m.G:
                raise InvalidDistributionInput, "MixtureModelPrior: invalid size of piPrior."

            try:
                # check validity of each component
                for i in range(m.G):
                    self.compPrior.isValid(m.components[i])
            except InvalidDistributionInput, ex:
                Log.error("in MixtureModelPrior for component " + str(i), ex)

    def structPriorHeuristic(self, delta, N):
        """
        Heuristic for setting the structure prior hyper-parameter 'self.structPrior', depending
        on the size of a data set 'N' and parameter 'delta'.
        """
        self.structPrior = - np.log(1 + delta) * N


    def mapMStepMerge(self, group_list):
        new_dist = copy.copy(group_list[0].dist)
        new_req_stat = copy.deepcopy(group_list[0].req_stat)

        assert len(new_dist.distList) == 1 # XXX
        assert new_dist.G == 2

        for i in range(new_dist.G):
            if new_dist.compFix[i] == 2:
                continue
            sub_group_list = []
            for r in range(len(group_list)):
                sub_group_list.append(CandidateGroup(group_list[r].dist.components[i][0], group_list[r].post_sum, group_list[r].pi_sum, group_list[r].req_stat[i]))

            d_i = self.compPrior[0].mapMStepMerge(sub_group_list)
            new_dist.components[i][0] = d_i.dist
            new_req_stat[i] = d_i.req_stat

        return CandidateGroup(new_dist, d_i.post_sum, d_i.pi_sum, new_req_stat)


