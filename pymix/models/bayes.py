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
import random
import sys
import numpy as np

from ..distributions.discrete import DiscreteDistribution
from ..distributions.product import ProductDistribution
from .mixture import MixtureModel
from ..util import setPartitions, mixextend, stats
from ..util.candidate_group import CandidateGroup
from ..util.dataset import DataSet
from ..util.errors import InvalidPosteriorDistribution, ConvergenceFailureEM
from ..util.maths import matrix_sum_logs
from ..util.setPartitions import init_last, prev_partition
from ..util.stats import sym_kl_dist


class BayesMixtureModel(MixtureModel):
    """
    Bayesian mixture models
    """

    def __init__(self, G, pi, components, prior, compFix=None, struct=0, identifiable=1):
        """
        Constructor

        @param G: number of components
        @param pi: mixture weights
        @param components: list of ProductDistribution objects, each entry is one component
        @param prior: MixtureModelPrior object
        @param compFix: list of optional flags for fixing components in the reestimation
                         the following values are supported:
                         1 distribution parameters are fixed,
                         2 distribution parameters and mixture coefficients are fixed
        @param struct: Flag for CSI structure,
            0 = no CSI structure
            1 = CSI structure
        """
        MixtureModel.__init__(self, G, pi, components, compFix=compFix, struct=struct, identifiable=identifiable)

        # check and set model prior
        self.prior = None
        prior.isValid(self)
        self.prior = prior

    def __str__(self):
        s = MixtureModel.__str__(self)
        s += "\n" + str(self.prior)
        return s

    def __eq__(self, other):
        if not isinstance(other, BayesMixtureModel):
            return False
        res = MixtureModel.__eq__(self, other)
        if res == False:
            return res
        else:
            res = self.prior.__eq__(other.prior)
            return res

    def __copy__(self):
        copy_components = []
        copy_pi = copy.deepcopy(self.pi)
        copy_compFix = copy.deepcopy(self.compFix)
        for i in range(self.G):
            copy_components.append(copy.deepcopy(self.components[i]))
        copy_prior = copy.copy(self.prior)
        copy_model = BayesMixtureModel(self.G, copy_pi, copy_components, copy_prior, compFix=copy_compFix)
        copy_model.nr_tilt_steps = self.nr_tilt_steps
        copy_model.suff_p = self.suff_p
        copy_model.identFlag = self.identFlag

        if self.struct:
            copy_model.initStructure()
            copy_leaders = copy.deepcopy(self.leaders)
            copy_groups = copy.deepcopy(self.groups)
            copy_model.leaders = copy_leaders
            copy_model.groups = copy_groups
        return copy_model

    def modelInitialization(self, data, rtype=1):
        """
        Perform model initialization given a random assigment of the
        data to the models.

        @param data: DataSet object
        @param rtype: type of random assignments.
        0 = fuzzy assignment
        1 = hard assignment

        @return: posterior assigments
        """
        if not isinstance(data, DataSet):
            raise TypeError, "DataSet object required, got" + str(data.__class__)
        else:
            if data.internalData is None:
                data.internalInit(self)
                # reset structure if applicable
        if self.struct:
            self.initStructure()
            # generate 'random posteriors'
        l = np.zeros((self.G, len(data)), dtype='Float64')
        for i in range(len(data)):
            if rtype == 0:
                for j in range(self.G):
                    l[j, i] = random.uniform(0.1, 1)
                s = sum(l[:, i])
                for j in range(self.G):
                    l[j, i] /= s
            else:
                l[random.randint(0, self.G - 1), i] = 1

        # do one M Step
        fix_pi = 1.0
        unfix_pi = 0.0
        fix_flag = 0   # flag for fixed mixture components
        for i in range(self.G):
            # setting values for pi
            if self.compFix[i] == 2:
                fix_pi -= self.pi[i]
                fix_flag = 1
            else:
                #self.pi[i] =  l[i,:].sum() / len(data)
                self.pi[i] = (l[i, :].sum() + self.prior.piPrior.alpha[i] - 1.0) / (len(data) + ( self.prior.piPrior.alpha_sum - self.G))
                unfix_pi += self.pi[i]
            if self.compFix[i] == 1 or self.compFix[i] == 2:
                # fixed component
                continue
            else:
                # components are product distributions that may contain mixtures
                if isinstance(self.components[i], ProductDistribution):
                    last_index = 0
                    for j in range(len(self.components[i])):
                        if isinstance(self.components[i][j], MixtureModel):
                            dat_j = data.singleFeatureSubset(j)
                            self.components[i][j].modelInitialization(dat_j, rtype=rtype)
                        else:
                            loc_l = l[i, :]
                            # masking missing values from parameter estimation
                            if data.missingSymbols.has_key(j):
                                ind_miss = data.getMissingIndices(j)
                                for k in ind_miss:
                                    loc_l[k] = 0.0
                            self.prior.compPrior[j].mapMStep(self.components[i][j], loc_l, data.getInternalFeature(j), dist_ind=i)
                else:  # components are not ProductDistributions -> invalid
                    raise TypeError

        # renormalizing mixing proportions in case of fixed components
        if fix_flag:
            if unfix_pi == 0.0:
            #print "----\n",self,"----\n"
                print "unfix_pi = ", unfix_pi
                print "fix_pi = ", fix_pi
                print "pi = ", self.pi
                raise RuntimeError, "unfix_pi = 0.0"
            for i in range(self.G):
                if self.compFix[i] == 0:
                    self.pi[i] = (self.pi[i] * fix_pi) / unfix_pi

        # updating hyperparameters in prior where apprpriate
        if self.prior.constant_hyperparams != 1:
            self.prior.updateHyperparameters(self, l, data)

        return l

    def mapEM(self, data, max_iter, delta, silent=False, mix_pi=None, mix_posterior=None, tilt=0):
        return MixtureModel.mapEM(self, data, self.prior, max_iter, delta, silent=silent, mix_pi=mix_pi, mix_posterior=mix_posterior, tilt=tilt)

    def removeComponent(self, ind):
        """
        Deletes a component from the model.

        @param ind: index of component to be removed
        """
        from ..priors.dirichlet import DirichletPrior

        MixtureModel.removeComponent(self, ind)
        # update component prior
        alpha = self.prior.piPrior.alpha.tolist()
        alpha.pop(ind)
        n_pipr = DirichletPrior(self.G, alpha)
        self.prior.piPrior = n_pipr


    # XXX CODE REVIEW


    def updateStructureBayesian(self, data, objFunction='MAP', silent=1):
        """
        Updating structure by chosing optimal local merge with respect to the posterior.


        Features: - store merges in a learning history to prevent recomputation of merge parameters
                  - compute parameters of candidate structures from parameters of groups to be merged


        @param data: DataSet object
        @param silent: verbosity flag

        @return: number of structure changes
        """
        assert self.struct == 1, "No structure in model."
        assert objFunction in ['MAP'] # for now only MAP estimation

        new_leaders = []
        new_groups = []
        change = 0
        # building data likelihood factor matrix for the current group structure
        l = np.zeros((len(self.components[0]), self.G, data.N), dtype='Float64')
        for j in range(len(self.components[0])):
            # extracting current feature from the DataSet
            if isinstance(self.components[0][j], MixtureModel): # XXX
                data_j = data.singleFeatureSubset(j)
            else:
                data_j = data.getInternalFeature(j)

            for lead_j in self.leaders[j]:
                l_row = self.components[lead_j][j].pdf(data_j)
                l[j, lead_j, :] = l_row
                for v in self.groups[j][lead_j]:
                    l[j, v, :] = l_row

        # g is the matrix of log posterior probabilities of the components given the data
        g = np.sum(l, axis=0)
        for k in range(self.G):
            g[k, :] += np.log(self.pi[k])

        sum_logs = matrix_sum_logs(g)

        try:
            g_norm = g - sum_logs
        except FloatingPointError:
            print sum_logs
            raise

        tau = np.exp(g_norm)

        if not silent:
            print "\ntau="
            for tt in tau:
                print tt.tolist()
            print

        # computing posterior as model selection criterion
        temp = DiscreteDistribution(self.G, self.pi)
        pi_prior = self.prior.piPrior.pdf(temp)
        log_prior = pi_prior
        log_prior_list = [0.0] * len(self.components[0])
        for j in range(len(self.components[0])):
            for r in range(self.G):
                log_prior_list[j] += self.prior.compPrior[j].pdf(self.components[r][j])
        log_prior += sum(log_prior_list)

        # prior over number of components
        log_prior += self.prior.nrCompPrior * self.G
        # prior over number of distinct groups
        for j in range(len(self.components[0])):
            log_prior += self.prior.structPrior * len(self.leaders[j])

        # get posterior
        lk = np.sum(sum_logs)
        post = lk + log_prior
        if not silent:
            print "0: ", lk, "+", log_prior, "=", post
            print log_prior_list

        changes = 0
        g_wo_j = np.zeros((self.G, data.N), dtype='Float64')

        # initialising temporary group structure with copies of the current structure
        temp_leaders = copy.deepcopy(self.leaders)
        temp_groups = copy.deepcopy(self.groups)
        for j in range(len(self.components[0])):
            L = {}  # initialize merge history

            if not silent:
                print "\n************* j = ", j, "*****************\n"

            # unnormalized posterior matrix without the contribution of the jth feature
            try:
                g_wo_j = g - l[j]
            except FloatingPointError:

                # if there was an exception we have to compute each
                # entry in g_wo_j seperately to set -inf - -inf = -inf
                g_wo_j = mixextend.substract_matrix(g, l[j])

            # checking whether feature j is already fully merged
            nr_lead = len(self.leaders[j])
            if nr_lead == 1:
                continue  # nothing to be done...

            term = 0
            if not silent:
                print self.leaders
                print self.groups

            # extracting current feature from the DataSet
            if isinstance(self.components[0][j], MixtureModel): # XXX
                data_j = data.singleFeatureSubset(j)
            else:
                data_j = data.getInternalFeature(j)

            # initialize merge history
            tau_pool = np.zeros(data.N, dtype='Float64')

            for lead in self.leaders[j]:
                #el_dist = copy.copy(self.components[lead][j])

                # NOT a copy, changes in el_dist changes self !
                el_dist = self.components[lead][j]
                tau_pool = copy.copy(tau[lead, :])
                pi_pool = self.pi[lead]
                for z in self.groups[j][lead]:
                    tau_pool += tau[z, :]
                    pi_pool += self.pi[z]

                if objFunction == 'MAP':
                    self.prior.compPrior[j].mapMStep(el_dist, tau_pool, data_j, pi_pool)
                else:
                    # should never get here...
                    raise TypeError

                stat = el_dist.sufficientStatistics(tau_pool, data_j)
                M = CandidateGroup(el_dist, np.sum(tau_pool), pi_pool, stat)
                l_row = el_dist.pdf(data_j)
                cdist_prior = self.prior.compPrior[j].pdf(el_dist)
                M.l = l_row
                M.dist_prior = cdist_prior

                L[(lead,) + tuple(self.groups[j][lead])] = M

                # update likelihood matrix for initial model
                for ll in [lead] + self.groups[j][lead]:
                    l[j, ll, :] = l_row

            while not term:
                best_dist = None   # accepted candidate distributions
                best_post = float('-inf')   # corresponding posteriors
                best_indices = None
                best_l_j = l[j]
                for mc1 in range(len(temp_leaders[j])):
                    merge_cand1 = temp_leaders[j][mc1]
                    for mc2 in range(mc1 + 1, len(temp_leaders[j])):
                        merge_cand2 = temp_leaders[j][mc2]
                        if not silent:
                            print "-------------------"
                            print merge_cand1, " -> ", merge_cand2
                            print self.components[merge_cand1][j], '( sum(tau) = ', sum(tau[merge_cand1, :]), ')'
                            print self.components[merge_cand2][j], '( sum(tau) = ', sum(tau[merge_cand2, :]), ')'

                        nr_leaders_j = len(temp_leaders[j]) - 1
                        cand_group_j = temp_groups[j][merge_cand1] + [merge_cand2] + temp_groups[j][merge_cand2]

                        hist_ind_part1 = (merge_cand1,) + tuple(temp_groups[j][merge_cand1])
                        hist_ind_part2 = (merge_cand2,) + tuple(temp_groups[j][merge_cand2])
                        hist_ind_complete = hist_ind_part1 + hist_ind_part2

                        recomp = 0

                        if L.has_key(hist_ind_complete):
                            recomp = 1

                        if not silent:
                            print "\ncandidate model structure: "
                            print 'merge:', hist_ind_part1, hist_ind_part2, '->', hist_ind_complete
                            #print "lead = ",leaders_j
                            #print "groups = ",groups_j
                            #print "others = ",others,"\n"

                        if not recomp:
                            assert L.has_key(hist_ind_part1), str(hist_ind_part1) + ' missing.'
                            assert L.has_key(hist_ind_part2), str(hist_ind_part2) + ' missing.'

                            M = self.prior.compPrior[j].mapMStepMerge([L[hist_ind_part1], L[hist_ind_part2]])
                            candidate_dist = M.dist

                            if not silent:
                                print "candidate:", candidate_dist

                            l_row = candidate_dist.pdf(data_j)
                            cdist_prior = self.prior.compPrior[j].pdf(candidate_dist)

                            M.l = l_row
                            M.dist_prior = cdist_prior
                            L[hist_ind_complete] = M
                        else:
                            # retrieve merge data from history
                            candidate_dist = L[hist_ind_complete].dist

                            if not silent:
                                print "candidate:", candidate_dist

                            l_row = L[hist_ind_complete].l
                            cdist_prior = L[hist_ind_complete].dist_prior

                            # computing change in likelihood matrix for this step
                        l_j_1 = copy.copy(l[j])

                        # updating l_j_1 with the new candidate distribution
                        l_j_1[merge_cand1, :] = l_row
                        for v in cand_group_j:
                            l_j_1[v, :] = l_row

                        # get updated unnormalized posterior matrix
                        g = mixextend.add_matrix(g_wo_j, l_j_1)
                        sum_logs = matrix_sum_logs(g)
                        lk_1 = np.sum(sum_logs)

                        # computing posterior as model selection criterion
                        log_prior_1 = pi_prior

                        # compute parameter prior for the candidate merge parameters
                        log_prior_list_j = 0.0
                        for r in range(self.G):
                            if r in [merge_cand1] + cand_group_j:
                                log_prior_list_j += cdist_prior
                            else:
                                log_prior_list_j += self.prior.compPrior[j].pdf(self.components[r][j])

                        log_prior_1 += sum(log_prior_list)
                        log_prior_1 -= log_prior_list[j]
                        log_prior_1 += log_prior_list_j

                        # prior over number of components
                        log_prior_1 += self.prior.nrCompPrior * self.G
                        # prior over number of distinct groups
                        for z in range(len(self.components[0])):
                            if z == j:
                                log_prior_1 += self.prior.structPrior * nr_leaders_j
                            else:
                                log_prior_1 += self.prior.structPrior * len(temp_leaders[z])   # XXX len could be cached ?

                        post_1 = lk_1 + log_prior_1

                        if not silent:
                            print '\nPosterior:', post_1, '=', lk_1, '+', log_prior_1

                        if post_1 >= post:
                            if not silent:
                                print "*** Merge accepted", post_1, ">=", post

                            if post_1 > best_post:  # current merge is better than previous best
                                best_dist = candidate_dist
                                best_post = post_1
                                best_indices = [merge_cand1, merge_cand2]
                                best_l_j = l_j_1
                                best_log_prior_list_j = log_prior_list_j
                        else:
                            if not silent:
                                print "*** Merge rejected:", post_1, "!>", post

                # if there is no possible merge that increases the score we are done
                if best_post == float('-inf'):
                    if not silent:
                        print "*** Finished !"
                        # setting updated structure in model
                    self.leaders[j] = temp_leaders[j]
                    self.groups[j] = temp_groups[j]

                    # reset posterior matrix to the last accepted merge
                    g = g_wo_j + best_l_j
                    term = 1

                # otherwise we update the model with the best merge found
                else:
                    if not silent:
                        print "\n--- Winner ---"
                        print "post:", best_post
                        print "indices:", best_indices
                        print "dist: ", best_dist
                        #print "lead:",best_leaders
                        #print "group:", best_groups

                    post = best_post
                    l[j] = best_l_j
                    g = g_wo_j + best_l_j  # posterior matrix for the next iteration
                    log_prior_list[j] = best_log_prior_list_j

                    # updating model
                    # removing merged leader from new_leaders
                    ind = temp_leaders[j].index(best_indices[1])
                    temp_leaders[j].pop(ind)

                    # joining merged groups and removing old group entry
                    temp_groups[j][best_indices[0]] += [best_indices[1]] + temp_groups[j][best_indices[1]]
                    temp_groups[j].pop(best_indices[1])

                    # assigning distributions according to new structure
                    self.components[best_indices[0]][j] = best_dist
                    for d in temp_groups[j][best_indices[0]]:
                        self.components[d][j] = self.components[best_indices[0]][j]
                    change += 1

        return change

    #---------------------------------------------------------------------------------------------------------------

    def updateStructureBayesianFullEnumerationFixedOrder(self, data, objFunction='MAP', silent=1):
        """
        Updating structure by chosing optimal local merge with respect to the posterior.

        Enumerates all possible structures for each feature seperately, i.e. returns the optimal structure
        for the given ordering of features. For a complete enumeration of the structure space, including permutation of
        feature order use the functions in fullEnumerationExhaustive.py.

        @param data: DataSet object
        @param silent: verbosity flag

        @return: number of structure changes
        """
        assert self.struct == 1, "No structure in model."
        assert objFunction in ['MAP'] # for now only MAP estimation

        new_leaders = []
        new_groups = []
        change = 0
        # building data likelihood factor matrix for the current group structure
        l = np.zeros((len(self.components[0]), self.G, data.N), dtype='Float64')
        for j in range(len(self.components[0])):
            for lead_j in self.leaders[j]:
                l_row = self.components[lead_j][j].pdf(data.getInternalFeature(j))
                l[j, lead_j, :] = l_row
                for v in self.groups[j][lead_j]:
                    l[j, v, :] = l_row

        # g is the matrix of log posterior probabilities of the components given the data
        g = np.sum(l, axis=0)
        for k in range(self.G):
            g[k, :] += np.log(self.pi[k])

        sum_logs = matrix_sum_logs(g)
        g_norm = g - sum_logs
        tau = np.exp(g_norm)

        if not silent:
            print "\ntau="
            for tt in tau:
                print tt.tolist()
            print

        # computing posterior as model selection criterion
        temp = DiscreteDistribution(self.G, self.pi)
        pi_prior = self.prior.piPrior.pdf(temp)
        log_prior = pi_prior
        log_prior_list = [0.0] * len(self.components[0])
        for j in range(len(self.components[0])):
            for r in range(self.G):
                log_prior_list[j] += self.prior.compPrior[j].pdf(self.components[r][j])
        log_prior += sum(log_prior_list)

        # prior over number of components
        log_prior += self.prior.nrCompPrior * self.G
        # prior over number of distinct groups
        for j in range(len(self.components[0])):
            log_prior += self.prior.structPrior * len(self.leaders[j])

        # get posterior
        lk = np.sum(sum_logs)
        best_post = lk + log_prior
        if not silent:
            print "0: ", lk, "+", log_prior, "=", best_post
            print log_prior_list

        changes = 0
        g_wo_j = np.zeros((self.G, data.N), dtype='Float64')

        # initialising temporary group structure with copies of the current structure
        #temp_leaders = copy.deepcopy(self.leaders)
        #temp_groups =  copy.deepcopy(self.groups)
        for j in range(len(self.components[0])):
            L = {}  # initialize merge history

            if not silent:
                print "\n************* j = ", j, "*****************\n"

            # unnormalized posterior matrix without the contribution of the jth feature
            try:
                g_wo_j = g - l[j]
            except FloatingPointError:
                # if there was an exception we have to compute each
                # entry in g_wo_j seperately to set -inf - -inf = -inf
                g_wo_j = mixextend.substract_matrix(g, l[j])


            # checking whether feature j is already fully merged
            nr_lead = len(self.leaders[j])
            if nr_lead == 1:
                continue  # nothing to be done...

            term = 0
            if not silent:
                print self.leaders
                print self.groups, '\n'

            # extracting current feature from the DataSet
            if isinstance(self.components[0][j], MixtureModel): # XXX
                data_j = data.singleFeatureSubset(j)
            else:
                data_j = data.getInternalFeature(j)

            for lead in self.leaders[j]:
                el_dist = copy.copy(self.components[lead][j])

                tau_pool = copy.copy(tau[lead, :])
                pi_pool = self.pi[lead]
                for z in self.groups[j][lead]:
                    tau_pool += tau[z, :]
                    pi_pool += self.pi[z]

                if objFunction == 'MAP':
                    self.prior.compPrior[j].mapMStep(el_dist, tau_pool, data_j, pi_pool)
                else:
                    # should never get here...
                    raise TypeError

                stat = el_dist.sufficientStatistics(tau_pool, data_j)
                M = CandidateGroup(el_dist, np.sum(tau_pool), pi_pool, stat)

                l_row = el_dist.pdf(data_j)
                cdist_prior = self.prior.compPrior[j].pdf(el_dist)

                M.l = l_row
                M.dist_prior = cdist_prior
                L[(lead,) + tuple(self.groups[j][lead])] = M

            # first partition is full structure matrix
            kappa, max_kappa = init_last(self.G)  # XXX first partition posterior is computed twice !

            best_dist = None   # accepted candidate distributions
            best_indices = None
            best_l_j = l[j]
            best_log_prior_list_j = log_prior_list[j]
            best_partition = [(ll,) + tuple(self.groups[j][ll]) for ll in self.leaders[j]]
            best_post = float('-inf')

            while 1:
                curr_part = setPartitions.decode_partition(np.arange(self.G), kappa, max_kappa)
                if not silent:
                    print "\n-------------------"
                    #print 'History:', L.keys()
                    print 'Current structure:', kappa, '->', j, curr_part

                    fullstruct = []
                    for jj in range(len(self.components[0])):
                        if jj != j:
                            fullstruct.append([tuple([ll] + self.groups[jj][ll]) for ll in self.leaders[jj]])
                        else:
                            fullstruct.append(curr_part)
                    print 'Full:', fullstruct

                # computing change in likelihood matrix for this step
                #l_j_1 = copy.copy(l[j])
                l_j_1 = np.zeros((self.G, data.N ))  # XXX needs only be done once

                for group in curr_part:
                    if L.has_key(group):
                        # retrieve merge data from history
                        candidate_dist = L[group].dist

                        if not silent:
                            print "  candidate:", group, candidate_dist

                        l_row = L[group].l
                        cdist_prior = L[group].dist_prior
                    else:
                        M = self.prior.compPrior[j].mapMStepMerge([L[(c,)] for c in group])
                        candidate_dist = M.dist

                        if not silent:
                            print "  candidate:", group, candidate_dist

                        l_row = candidate_dist.pdf(data_j)
                        cdist_prior = self.prior.compPrior[j].pdf(candidate_dist)

                        M.l = l_row
                        M.dist_prior = cdist_prior

                        L[group] = M

                    for g in group:
                        l_j_1[g, :] = l_row

                # get updated unnormalized posterior matrix
                g = g_wo_j + l_j_1
                sum_logs = matrix_sum_logs(g)
                lk_1 = np.sum(sum_logs)

                # computing posterior as model selection criterion
                log_prior_1 = pi_prior

                # compute parameter prior for the candidate merge parameters
                log_prior_list_j = 0.0
                for r in curr_part:
                    log_prior_list_j += L[r].dist_prior * len(r)

                log_prior_1 += sum(log_prior_list)
                log_prior_1 -= log_prior_list[j]
                log_prior_1 += log_prior_list_j

                # prior over number of components
                log_prior_1 += self.prior.nrCompPrior * self.G
                # prior over number of distinct groups
                for z in range(len(self.components[0])):
                    if z == j:
                        log_prior_1 += self.prior.structPrior * len(curr_part)
                    else:
                        log_prior_1 += self.prior.structPrior * len(self.leaders[z])   # XXX len could be cached ?
                post_1 = lk_1 + log_prior_1

                if not silent:
                    print '\nPosterior:', post_1, '=', lk_1, '+', log_prior_1

                if post_1 >= best_post: # current candidate structure is better than previous best
                    if not silent:
                        print "*** New best candidate", post_1, ">=", best_post
                    best_post = post_1
                    best_partition = curr_part # XXX
                    best_l_j = l_j_1
                    best_log_prior_list_j = log_prior_list_j

                else:
                    if not silent:
                        print "*** worse than previous best", best_partition, '(', post_1, "!>", best_post, ')'

                ret = prev_partition(kappa, max_kappa)
                if ret == None:  # all candidate partitions have been scored
                    if not silent:
                        print "*** Finished with post=", best_post
                        # setting updated structure in model
                    lead = []
                    groups = {}
                    for gr in best_partition:
                        gr_list = list(gr)
                        gr_lead = gr_list.pop(0)
                        lead.append(gr_lead)
                        groups[gr_lead] = gr_list

                        # assigning distributions according to new structure
                        self.components[gr_lead][j] = L[gr].dist
                        for d in gr_list:
                            self.components[d][j] = self.components[gr_lead][j]

                    self.leaders[j] = lead
                    self.groups[j] = groups

                    # reset posterior matrix to the last accepted merge
                    g = g_wo_j + best_l_j
                    log_prior_list[j] = best_log_prior_list_j
                    break

                kappa, max_kappa = ret


                #---------------------------------------------------------------------------------------------------------------

    def updateStructureBayesianBottomUp(self, data, objFunction='MAP', silent=1):
        """
        Updating structure by chosing optimal local merge with respect to the posterior. The procedure
        starts with the minimally complex structure, i.e. every feature has a single distribution.


        @param data: DataSet object
        @param silent: verbosity flag

        @return: number of structure changes
        """
        assert self.struct == 1, "No structure in model."
        assert objFunction in ['MAP'] # for now only MAP estimation

        new_leaders = []
        new_groups = []
        change = 0
        # building data likelihood factor matrix for the current group structure
        l = np.zeros((len(self.components[0]), self.G, data.N), dtype='Float64')
        for j in range(len(self.components[0])):
            for lead_j in self.leaders[j]:
                l_row = self.components[lead_j][j].pdf(data.getInternalFeature(j))
                l[j, lead_j, :] = l_row
                for v in self.groups[j][lead_j]:
                    l[j, v, :] = l_row

        # g is the matrix of log posterior probabilities of the components given the data
        g = np.sum(l, axis=0)
        for k in range(self.G):
            g[k, :] += np.log(self.pi[k])

        sum_logs = matrix_sum_logs(g)
        g_norm = g - sum_logs
        tau = np.exp(g_norm)

        if not silent:
            print "\ntau="
            for tt in tau:
                print tt.tolist()
            print

        # computing posterior as model selection criterion
        temp = DiscreteDistribution(self.G, self.pi)
        pi_prior = self.prior.piPrior.pdf(temp)

        # compute feature wise parameter prior contributions
        log_prior_list = [0.0] * len(self.components[0])
        for j in range(len(self.components[0])):
            for r in range(self.G):
                log_prior_list[j] += self.prior.compPrior[j].pdf(self.components[r][j])

        changes = 0
        g_wo_j = np.zeros((self.G, data.N), dtype='Float64')

        # initialising starting group structure
        temp_leaders = copy.copy(self.leaders)
        temp_groups = copy.copy(self.groups)
        for j in range(len(self.components[0])):
            L = {}  # initialize merge history

            if not silent:
                print "\n************* j = ", j, "*****************\n"

            # initialising starting group structure for feature j
            temp_leaders[j] = [0]
            temp_groups[j] = {0: range(1, self.G)}

            # unnormalized posterior matrix without the contribution of the jth feature
            try:
                g_wo_j = g - l[j]
            except FloatingPointError:
                # if there was an exception we have to compute each
                # entry in g_wo_j seperately to set -inf - -inf = -inf
                g_wo_j = mixextend.substract_matrix(g, l[j])

            term = 0

            if not silent:
                print temp_leaders
                print temp_groups

            # extracting current feature from the DataSet
            if isinstance(self.components[0][j], MixtureModel): # XXX
                data_j = data.singleFeatureSubset(j)
            else:
                data_j = data.getInternalFeature(j)

            # initial model structure
            tau_pool = np.ones(data.N, dtype='Float64')
            pi_pool = 1.0
            el_dist = copy.copy(self.components[0][j])

            if objFunction == 'MAP':
                self.prior.compPrior[j].mapMStep(el_dist, tau_pool, data_j, pi_pool)
            else:
                # should never get here...
                raise TypeError

            # set initial distribution in model
            for i in range(self.G):
                self.components[i][j] = el_dist

            stat = el_dist.sufficientStatistics(tau_pool, data_j)
            M = CandidateGroup(el_dist, np.sum(tau_pool), pi_pool, stat)

            l_row = el_dist.pdf(data_j)
            cdist_prior = self.prior.compPrior[j].pdf(el_dist)

            M.l = l_row
            M.dist_prior = cdist_prior
            L[tuple(range(self.G))] = M

            sum_logs = matrix_sum_logs(g_wo_j + l_row)

            temp = copy.copy(l_row)
            temp = temp.reshape(1, data.N)
            l[j] = temp.repeat(self.G, axis=0)

            # get likelihood
            lk = np.sum(sum_logs)

            log_prior = pi_prior
            log_prior_list[j] = self.G * self.prior.compPrior[j].pdf(el_dist)
            log_prior += np.sum(log_prior_list)

            # prior over number of components
            log_prior += self.prior.nrCompPrior * self.G
            # prior over number of distinct groups
            for jj in range(len(self.components[0])):
                log_prior += self.prior.structPrior * len(temp_leaders[jj])

            post = lk + log_prior
            if not silent:
                print "0: ", lk, "+", log_prior, "=", post

            split_dist = copy.copy(self.components[0][j])
            while not term:
                best_split_dist = None   # accepted split distributions
                best_remainder_dist = None   # accepted remainder distributions
                best_log_prior_list_j = log_prior_list[j]
                best_post = float('-inf')   # corresponding posteriors
                best_leaders = None
                best_groups = None
                #best_indices = None
                best_l_j = l[j]
                for mc1 in range(len(temp_leaders[j])):
                    merge_cand1 = temp_leaders[j][mc1]

                    if len(temp_groups[j][merge_cand1]) == 0:
                        continue    # nothing to split

                    for mc1_grp in temp_groups[j][merge_cand1]:
                        if not silent:
                            print "-------------------"
                            print '*** leader ' + str(merge_cand1) + ': split', mc1_grp, 'from', temp_groups[j][merge_cand1]
                            print self.components[merge_cand1][j], '( sum(tau) = ', sum(tau[merge_cand1, :]), ')'

                        # initialising candidate group structure with copies of the temporary structure
                        leaders_j = copy.copy(temp_leaders[j])
                        groups_j = copy.deepcopy(temp_groups[j])

                        hist_ind_presplit = (merge_cand1,) + tuple(groups_j[merge_cand1])

                        # adding new leader created by split to leaders_j
                        leaders_j.append(mc1_grp)

                        # removing new leader from merge_cand1 group
                        ind = groups_j[merge_cand1].index(mc1_grp)
                        groups_j[merge_cand1].pop(ind)

                        # adding new group
                        groups_j[mc1_grp] = []

                        if not silent:
                            print "\ncandidate model structure:"
                            print "lead = ", leaders_j
                            print 'groups =', groups_j
                            print j, [(ll,) + tuple(groups_j[ll]) for ll in leaders_j]

                        nr_leaders_j = len(temp_leaders[j]) - 1

                        hist_ind_remainder = (merge_cand1,) + tuple(groups_j[merge_cand1])
                        hist_ind_split = (mc1_grp,)

                        # computing change in likelihood matrix for this step
                        l_j_1 = copy.copy(l[j])

                        # computing parameters for new single component group
                        if objFunction == 'MAP':
                            self.prior.compPrior[j].mapMStep(split_dist, tau[mc1_grp, :], data_j, self.pi[mc1_grp])
                        else:
                            # should never get here...
                            raise TypeError
                        if not silent:
                            print "split dist:", split_dist

                        # updating l_j_1 with the new split distribution
                        l_row = split_dist.pdf(data_j)
                        l_j_1[mc1_grp, :] = l_row

                        # add candidategroup
                        stat = split_dist.sufficientStatistics(tau[mc1_grp, :], data_j)
                        M = CandidateGroup(split_dist, np.sum(tau[mc1_grp, :]), self.pi[mc1_grp], stat)
                        L[hist_ind_split] = M

                        split_dist_prior = self.prior.compPrior[j].pdf(split_dist)

                        M.l = l_row
                        M.dist_prior = split_dist_prior

                        # computing parameters for group which has been split
                        recomp = 0

                        if L.has_key(hist_ind_remainder):
                            recomp = 1

                        if not recomp:
                            assert L.has_key(hist_ind_presplit), str(hist_ind_presplit) + ' missing.'
                            assert L.has_key(hist_ind_split), str(hist_ind_split) + ' missing.'
                            M = self.prior.compPrior[j].mapMStepSplit(L[hist_ind_presplit], L[hist_ind_split])
                            remainder_dist = M.dist

                            if not silent:
                                print "remainder dist:", remainder_dist

                            l_row = remainder_dist.pdf(data_j)
                            remainder_dist_prior = self.prior.compPrior[j].pdf(remainder_dist)

                            M.l = l_row
                            M.dist_prior = remainder_dist_prior

                            L[hist_ind_remainder] = M
                        else:
                            # retrieve merge data from history
                            remainder_dist = L[hist_ind_remainder].dist

                            if not silent:
                                print "remainder dist:", remainder_dist

                            l_row = L[hist_ind_remainder].l
                            remainder_dist_prior = L[hist_ind_remainder].dist_prior

                        # updating l_j_1 with the new remainder distribution
                        l_j_1[merge_cand1, :] = l_row
                        for v in groups_j[merge_cand1]:
                            l_j_1[v, :] = l_row

                        sum_logs = matrix_sum_logs(g)
                        lk_1 = np.sum(sum_logs)

                        # computing posterior as model selection criterion
                        log_prior_1 = pi_prior

                        # compute parameter prior for the candidate merge parameters
                        log_prior_list_j = 0.0
                        for r in range(self.G):
                            if r in [merge_cand1] + groups_j[merge_cand1]:
                                log_prior_list_j += remainder_dist_prior
                            elif r == mc1_grp:
                                log_prior_list_j += split_dist_prior
                            else:
                                log_prior_list_j += self.prior.compPrior[j].pdf(self.components[r][j])

                        log_prior_1 += sum(log_prior_list)
                        log_prior_1 -= log_prior_list[j]
                        log_prior_1 += log_prior_list_j

                        # prior over number of components
                        log_prior_1 += self.prior.nrCompPrior * self.G
                        # prior over number of distinct groups
                        for z in range(len(self.components[0])):
                            if z == j:
                                log_prior_1 += self.prior.structPrior * (len(temp_leaders[z]) + 1)
                            else:
                                log_prior_1 += self.prior.structPrior * len(temp_leaders[z])   # XXX len could be cached ?

                        post_1 = lk_1 + log_prior_1
                        if not silent:
                            print '\nPosterior:', post_1, '=', lk_1, '+', log_prior_1

                        if post_1 >= post:
                            if not silent:
                                print "*** Split accepted", post_1, ">=", post

                            if post_1 > best_post:  # current merge is better than previous best
                                best_split_dist = copy.copy(split_dist)
                                best_remainder_dist = copy.copy(remainder_dist)
                                best_post = post_1
                                best_leaders = leaders_j
                                best_groups = groups_j

                                best_indices = [merge_cand1, mc1_grp]
                                #best_l_j = copy.copy(l_j_1)
                                best_l_j = l_j_1
                                best_log_prior_list_j = log_prior_list_j
                        else:
                            if not silent:
                                print "*** Split rejected:", post_1, "!>", post

                # if there is no possible split that increases the score we are done
                if best_post == float('-inf'):
                    if not silent:
                        print "*** Finished with post", post
                        # setting updated structure in model
                    self.leaders[j] = temp_leaders[j]
                    self.groups[j] = temp_groups[j]

                    # reset posterior matrix to the last accepted merge
                    l[j] = best_l_j
                    g = g_wo_j + best_l_j
                    term = 1

                # otherwise we update the model with the best merge found
                else:
                    if not silent:
                        print "\n--- Winner ---"

                        print "post:", best_post
                        print "indices:", best_indices
                        print "remainder dist: ", best_remainder_dist
                        print "split dist: ", best_split_dist

                        print "lead:", best_leaders
                        print "group:", best_groups

                    post = best_post
                    l[j] = best_l_j
                    g = g_wo_j + best_l_j  # posterior matrix for the next iteration
                    log_prior_list[j] = best_log_prior_list_j

                    # updating model
                    temp_leaders[j] = best_leaders
                    temp_groups[j] = best_groups

                    # assigning distributions according to new structure
                    self.components[best_indices[0]][j] = copy.copy(best_remainder_dist)

                    for d in temp_groups[j][best_indices[0]]:
                        self.components[d][j] = self.components[best_indices[0]][j]

                    self.components[best_indices[1]][j] = copy.copy(best_split_dist)
                    change += 1

        return change

    def KLFeatureRanks(self, data, comps, silent=False):
        """
        Ranks the features by the symmetric relative entropy between the parameters induced in a structure
        where all components in 'comps' are merged and the parameters induced by all components (i.e. the uninformative
        structure).

        This gives a ranking of the relevance,i.e. discriminatory information of features for distinguishing a
        subset of components.

        @param data: DataSet object
        @param comps: list of component indices

        @return: list of tuples of (feature index, score) pairs in descending score order
        """
        from .mixture import MixtureModel

        assert type(comps) == list  # XXX for debugging mostly

        # some initial checks
        for c in comps:
            assert c in range(self.G) # check for valid entries
        assert len(comps) < self.G  # all components doesn`t make sense

        comps.sort()

        others = range(self.G)
        for c in comps:
            others.remove(c)

        if not silent:
            print 'comps', comps
            print 'others:', others

        # building data likelihood factor matrix for the current group structure
        l = np.zeros((len(self.components[0]), self.G, data.N), dtype='Float64')
        for j in range(len(self.components[0])):
            if isinstance(self.components[0][j], MixtureModel):
                data_j = data.singleFeatureSubset(j)
            else:
                data_j = data.getInternalFeature(j)
            for lead_j in self.leaders[j]:
                l_row = self.components[lead_j][j].pdf(data_j)
                l[j, lead_j, :] = l_row
                for v in self.groups[j][lead_j]:
                    l[j, v, :] = l_row

        # g is the matrix of log posterior probabilities of the components given the data
        g = np.sum(l, axis=0)

        for k in range(self.G):
            g[k, :] += np.log(self.pi[k])

        sum_logs = np.zeros(data.N, dtype='Float64')
        g_norm = np.zeros((self.G, data.N), dtype='Float64')
        for n in range(data.N):
            sum_logs[n] = stats.sum_logs(g[:, n])
            # normalizing log posterior
            g_norm[:, n] = g[:, n] - sum_logs[n]

        tau = np.exp(g_norm)
        model = copy.copy(self)
        score = []
        for j in range(len(model.leaders)):
            if len(model.leaders[j]) == 1:
                if not silent:
                    print 'Feature ' + str(data.headers[j]) + ' uninformative.'
                    # if the feature is uninformative already the score is set to zero
                score.append((0.0, j))
                continue
            else:
                # this whole section is more general than needed, might get optimized as some point

                if not silent:
                    print '\nFeature ' + str(data.headers[j]) + ' ( index ' + str(j) + ' ) usefull. '

                # data for the jth feature
                if isinstance(model.components[0][j], MixtureModel):
                    data_j = data.singleFeatureSubset(j)
                else:
                    data_j = data.getInternalFeature(j)

                # updating groups and leaders:
                # backup leaders and groups of feature j
                backup_leaders = copy.copy(model.leaders[j])
                backup_groups = copy.copy(model.groups[j])

                # clear structure for j
                model.groups[j] = {}

                # assign structure for the ranking
                model.leaders[j] = [comps[0], others[0]]
                model.groups[j][comps[0]] = comps[1:]
                model.groups[j][others[0]] = others[1:]

                # saving distribution parameters for feature j
                temp = []
                for i in range(model.G):
                    temp.append(model.components[i][j])

                # estimating distribution for component to be ranked
                comps_dist_j = copy.copy(model.components[0][j])
                tau_pool = copy.copy(tau[comps[0], :])
                pi_pool = self.pi[comps[0]]
                for z in model.groups[j][comps[0]]:
                    tau_pool += tau[z, :]
                    pi_pool += self.pi[z]
                model.prior.compPrior[j].mapMStep(comps_dist_j, tau_pool, data_j, pi_pool)

                # estimating distribution for all components except the one to be ranked
                others_dist_j = copy.copy(model.components[0][j])
                tau_pool = copy.copy(tau[others[0], :])
                pi_pool = self.pi[others[0]]
                for z in model.groups[j][others[0]]:
                    tau_pool += tau[z, :]
                    pi_pool += self.pi[z]
                model.prior.compPrior[j].mapMStep(others_dist_j, tau_pool, data_j, pi_pool)

                # compute feature score
                score.append(( sym_kl_dist(comps_dist_j, others_dist_j), j))

        score.sort()
        score.reverse()
        return score

    def randMaxTraining(self, data, nr_runs, nr_steps, delta, tilt=0, objFunction='MAP', rtype=1, silent=False):
        """
        Performs `nr_runs` MAP EM runs with random initial parameters
        and returns the model which yields the maximum likelihood.

        @param data: DataSet object
        @param nr_runs: number of repeated random initializations
        @param nr_steps: maximum number of steps in each run
        @param delta: minimim difference in log-likelihood before convergence
        @param tilt: 0/1 flag, toggles the use of a deterministic annealing in the training
        @param silent:0/1 flag, toggles verbose output

        @return: log-likelihood of winning model
        """
        if hasattr(data, "__iter__"):
            raise TypeError, "DataSet object required."
        elif isinstance(data, DataSet):
            if data.internalData is None:
                sys.stdout.write("Parsing data set...")
                sys.stdout.flush()
                data.internalInit(self)
                sys.stdout.write("done\n")
                sys.stdout.flush()
        else:
            raise ValueError, "Unknown input type format: " + str(data.__class__)

        assert objFunction in ['MAP']  # only MAP estimation for now

        best_logp = float('-inf')
        best_model = None
        logp_list = []
        for i in range(nr_runs):
            # copying inside of the loop is necessary for the dirichlet
            # models to get independent parameter initializations
            candidate_model = copy.copy(self)  # copying the model parameters
            # we do repeated random intializations until we get a model with valid posteriors in all components
            init = 0
            while not init:
                try:
                    candidate_model.modelInitialization(data, rtype=rtype)    # randomizing parameters of the model copy
                except InvalidPosteriorDistribution:
                    pass
                else:
                    init = 1
            try:
                if objFunction == 'MAP': # maximum a posteriori estimation
                    (l, log_p) = candidate_model.mapEM(data, nr_steps, delta, silent=silent, tilt=tilt)  # running EM
                else:
                    # should never get here...
                    raise TypeError
                logp_list.append(log_p)

            except ConvergenceFailureEM:
                sys.stdout.write("Run " + str(i) + " produced invalid model, omitted.\n")
            except ValueError:
                sys.stdout.write("Run " + str(i) + " produced invalid model, omitted.\n")
            except InvalidPosteriorDistribution:
                sys.stdout.write("Run " + str(i) + " produced invalid model, omitted.\n")
            else:
                # current model is better than previous optimum
                if log_p > best_logp:
                    best_model = copy.copy(candidate_model)
                    best_logp = log_p

        if not silent:
            print "\nBest model likelihood over ", nr_runs, "random initializations ( " + str(nr_runs - len(logp_list)) + " runs failed):"
            if len(logp_list) > 0:
                print "Model likelihoods:", logp_list
                print "Average logp: ", sum(logp_list) / len(logp_list), " SD:", np.array(logp_list).std()
                print "Best logp:", best_logp

        self.components = best_model.components  # assign best parameter set to model 'self'
        self.pi = best_model.pi
        return best_logp  # return final data log likelihood

    def shortInitMAP(self, data, nr_runs, nr_init, nr_steps, delta, tilt=0, nr_seed_steps=5, silent=False):
        """
        EM strategy:
            - 'nr_init' random initial models
            - short EM runs with each start model
            - take model with best likelihood for long EM run.

        The short runs are set to 5 iterations by default

        @param data: DataSet object
        @param nr_runs: number of repeated random initializations
        @param nr_init: number of random models for each run
        @param nr_steps: maximum number of steps for the long training run
        @param delta: minimim difference in log-likelihood before convergence
        @param tilt: 0/1 flag, toggles the use of a deterministic annealing in the training
        @param nr_seed_steps: number of EM steps for each seed model, default is 5
        @param silent:0/1 flag, toggles verbose output

        @return: log-likelihood of winning model
        """
        if hasattr(data, "__iter__"):
            raise TypeError, "DataSet object required."
        elif isinstance(data, DataSet):
            if data.internalData is None:
                sys.stdout.write("Parsing data set...")
                sys.stdout.flush()
                data.internalInit(self)
                sys.stdout.write("done\n")
                sys.stdout.flush()
        else:
            raise ValueError, "Unknown input type format: " + str(data.__class__)

        seed_model = copy.copy(self)  # copying the model parameters
        best_model = None
        best_model_logp = float('-inf')
        model_logp = []
        for i in range(nr_runs):
            print "i = ", i
            best_seed_model = None
            best_seed_logp = float('-inf')
            seed_logp = []

            for j in range(nr_init):
                seed_model.modelInitialization(data)    # randomizing parameters of the model copy
                invalid_model = 0  # flag for models that produced an exception in EM
                try:
                    (l, logp_i) = seed_model.mapEM(data, nr_seed_steps, 0.1, silent=silent)
                except InvalidPosteriorDistribution:
                    print "Invalid seed model discarded: ", j
                    invalid_model = 1

                # only valid models are considered in the maximization
                if not invalid_model:
                    if logp_i > best_seed_logp:
                        #print 'better',j
                        best_seed_logp = logp_i
                        best_seed_model = copy.copy(seed_model)
                    seed_logp.append(logp_i)

            sys.stdout.write("--- picking best seed model ---\n")
            try:
                (l, logp) = best_seed_model.mapEM(data, nr_steps, delta, silent=silent, tilt=tilt)
            except InvalidPosteriorDistribution:
                sys.stdout.write("***** Run " + str(i) + " produced invalid model.\n")
            except ZeroDivisionError:
                sys.stdout.write("***** Run " + str(i) + " is numerically instable.\n")
            else:
                if logp > best_model_logp:
                    best_model_logp = logp
                    best_model = copy.copy(best_seed_model)
                model_logp.append(logp)

        if not silent:
            print "\nBest model likelihood over ", nr_runs, " repeats ( " + str(nr_runs - len(model_logp)) + " runs failed):"
            print "Model likelihoods:", model_logp
            print "Average logp: ", sum(model_logp) / len(model_logp), " SD:", np.array(model_logp).std()
            print "Best logp:", best_model_logp

        final_logp = np.array(model_logp, dtype='Float64')
        # assign best parameter set to model 'self'
        self.components = best_model.components
        self.pi = best_model.pi
        return best_model_logp

    def bayesStructureEM(self, data, nr_repeats, nr_runs, nr_steps, delta, tilt=0, objFunction='MAP', silent=False, min_struct=1, rtype=1):
        """
        EM training for models with CSI structure.
        First a candidate model is generated by using the randMaxMAP procedure,
        then the structure is trained.

        @param data: DataSet object
        @param nr_repeats: number of candidate models to be generated
        @param nr_runs: number of repeated random initializations
        @param nr_steps: maximum number of steps for the long training run
        @param delta: minimim difference in log-likelihood before convergence
        @param tilt: 0/1 flag, toggles the use of a deterministic annealing in the training
        @param silent:0/1 flag, toggles verbose output
        @param min_struct: 0/1 flag, toggles merging of components with identical paramters

        @return: log-likelihood of winning model
        """
        if hasattr(data, "__iter__"):
            raise TypeError, "DataSet object required."
        elif isinstance(data, DataSet):
            if data.internalData is None:
                sys.stdout.write("Parsing data set...")
                sys.stdout.flush()
                data.internalInit(self)
                sys.stdout.write("done\n")
                sys.stdout.flush()
        else:
            raise ValueError, "Unknown input type format: " + str(data.__class__)

        assert objFunction in ['MAP']  # only MAP estimation for now
        assert self.struct
        logp_list = []
        best_logp = None
        best_candidate = None
        for r in range(nr_repeats):
            error = 0
            candidate_model = copy.copy(self)  # copying the model parameters
            # we do repeated random intializations until we get a model with valid posteriors in all components
            init = 0
            while not init:
                try:
                    #print "bayesStructureEM: try init   "
                    candidate_model.modelInitialization(data, rtype=rtype)    # randomizing parameters of the model copy
                except InvalidPosteriorDistribution:
                    pass
                else:
                    init = 1

            log_p = candidate_model.randMaxTraining(data, nr_runs, nr_steps, delta, tilt=tilt, objFunction=objFunction, rtype=rtype, silent=silent)

            try:
                ch = candidate_model.updateStructureBayesian(data, objFunction=objFunction, silent=1)
            except ValueError:
                error = 1
                print 'ERROR: failed structure lerarning.'
                continue

            if not silent:
                print "Changes = ", ch
            while ch != 0:
                try:
                    if objFunction == 'MAP':
                        candidate_model.mapEM(data, nr_steps, delta, silent=silent, tilt=0)
                    else:
                        # should never get here...
                        raise TypeError
                    ch = candidate_model.updateStructureBayesian(data, objFunction=objFunction, silent=1)
                    if not silent:
                        print "Changes = ", ch
                except ConvergenceFailureEM:
                    print 'FAILURE:'
                    error = 1
                    break

            try:
                # DEBUG: check structure validity
                self.validStructure()
            except AssertionError:
                print 'ERROR: Produced invalid structure'
                error = 1
            if not error:
                try:
                    if objFunction == 'MAP':
                        (l, log_p) = candidate_model.mapEM(data, nr_steps, delta, silent=silent, tilt=0)
                    else:
                        # should never get here...
                        raise TypeError
                except ConvergenceFailureEM:
                    continue
                logp_list.append(log_p)
                if r == 0 or log_p > best_logp:
                    best_logp = log_p
                    best_candidate = candidate_model
            else:
                continue

        self.components = best_candidate.components  # assign best parameter set to model 'self'
        self.pi = best_candidate.pi
        self.groups = best_candidate.groups
        self.leaders = best_candidate.leaders
        if min_struct:
            # remove redundant components
            self.minimalStructure()

        # update free parameters
        self.updateFreeParams()
        if not silent:
            print 'Structural EM (', nr_repeats, ' runs over', nr_runs, 'random inits each):'
            print 'logp:', logp_list
            print "Average logp: ", sum(logp_list) / len(logp_list), " SD:", np.array(logp_list).std()
            print "Best logp:", best_logp
        return best_logp


    def minimalStructure(self):
        from ..priors.dirichlet import DirichletPrior
        from .mixture import MixtureModel


        d = MixtureModel.minimalStructure(self)
        # if there have been changes to the number of
        # components the prior needs to be updated
        if d is not None:
            l = d.keys()
            l.sort()
            l.reverse()
            alpha = self.prior.piPrior.alpha.tolist()
            for i in l:
                a = alpha.pop(i)
                alpha[d[i]] += (a - 1.0)

            #print "alpha =",alpha
            assert len(alpha) == self.G
            new_piPr = DirichletPrior(self.G, alpha)
            self.prior.piPrior = new_piPr

