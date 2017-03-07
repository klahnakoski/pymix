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
from .bayes import BayesMixtureModel
from .mixture import MixtureModel
from ..util import mixextend
from ..util.candidate_group import CandidateGroup
from ..util.constrained_dataset import ConstrainedDataSet
from ..util.dataset import DataSet
from ..util.errors import ConvergenceFailureEM, InvalidPosteriorDistribution
from ..util.maths import matrix_sum_logs


class labeledBayesMixtureModel(BayesMixtureModel):
    """

    Bayesian mixture models with labeled data.

    """

    def __init__(self, G, pi, components, prior, compFix=None, struct=0):
        BayesMixtureModel.__init__(self, G, pi, components, prior, compFix=compFix, struct=struct, identifiable=0)


    def __copy__(self):
        copy_components = []

        #copy_pi = copy.copy(self.pi)
        copy_pi = copy.deepcopy(self.pi)
        copy_compFix = copy.deepcopy(self.compFix)

        for i in range(self.G):
            copy_components.append(copy.deepcopy(self.components[i]))

        copy_prior = copy.copy(self.prior)

        copy_model = labeledBayesMixtureModel(self.G, copy_pi, copy_components, copy_prior, compFix=copy_compFix)
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
        0 = fuzzy assingment
        1 = hard assingment
        @return posterior assigments
        """
        assert self.G >= len(data.labels), 'Insufficent number of components for given labeling.'

        if not isinstance(data, ConstrainedDataSet):
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

        # peform label assigments (non random!)
        for i, cl in enumerate(data.labels): # for each class
            for o in cl: # for each observation in a class
                p_vec = np.zeros(self.G, dtype='Float64')
                p_vec[i] = 1.0
                l[:, o] = p_vec



        #print 'Random init l:'
        #for u in range(self.G):
        #    print u,l[u,:].tolist()

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
                self.pi[i] = l[i, :].sum() / len(data)
                unfix_pi += self.pi[i]

            if self.compFix[i] == 1 or self.compFix[i] == 2:
                # fixed component
                continue
            else:
                # components are product distributions that may contain mixtures
                if isinstance(self.components[i], ProductDistribution):
                    last_index = 0
                    for j in range(len(self.components[i].distList)):
                        if isinstance(self.components[i].distList[j], MixtureModel):
                            dat_j = data.singleFeatureSubset(j)
                            self.components[i].distList[j].modelInitialization(dat_j, rtype=rtype)
                        else:
                            loc_l = l[i, :]
                            # masking missing values from parameter estimation
                            if data.missingSymbols.has_key(j):
                                ind_miss = data.getMissingIndices(j)
                                for k in ind_miss:
                                    loc_l[k] = 0.0
                                    #self.components[i].distList[j].mapMStep(loc_l,data.getInternalFeature(j),self.prior.compPrior[j] )
                            self.prior.compPrior.priorList[j].mapMStep(self.components[i].distList[j], loc_l, data.getInternalFeature(j))

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
        return l


    def mapEM(self, data, max_iter, delta, silent=False, mix_pi=None, mix_posterior=None, tilt=0):
        """
        Reestimation of maximum a posteriori mixture parameters using the EM algorithm.

        @param data: DataSet object
        @param max_iter: maximum number of iterations
        @param delta: minimal difference in likelihood between two iterations before
        convergence is assumed.
        @param silent: 0/1 flag, toggles verbose output
        @param mix_pi: [internal use only] necessary for the reestimation of
        mixtures as components
        @param mix_posterior:[internal use only] necessary for the reestimation of
        mixtures as components
        @param tilt: 0/1 flag, toggles the use of a deterministic annealing in the training

        @return: tuple of posterior matrix and log-likelihood from the last iteration
        """
        assert self.G >= len(data.labels), 'Insufficent number of components for given labeling.'

        if hasattr(data, "__iter__"):
            raise TypeError, "DataSet object required."
        elif isinstance(data, DataSet):
            if data.internalData is None:
                if not silent:
                    sys.stdout.write("Parsing data set...")
                    sys.stdout.flush()
                data.internalInit(self)
                if not silent:
                    sys.stdout.write("done\n")
                    sys.stdout.flush()
        else:
            raise ValueError, "Unknown input type format: " + str(data.__class__)

        log_p_old = float('-inf')
        step = 0

        # if deterministic annealing is activated, increase number of steps by self.nr_tilt_steps
        if tilt:
            if not silent:
                sys.stdout.write("Running EM with " + str(self.nr_tilt_steps) + " steps of deterministic annealing.\n")
            max_iter += self.nr_tilt_steps

        # for lower hierarchy mixture we need the log of mix_posterior
        if mix_posterior is not None:
            log_mix_posterior = np.log(mix_posterior)
        else:
            log_mix_posterior = None

        while 1:
            log_p = 0.0
            # matrix of log posterior probs: components# * (sequence positions)
            log_l = np.zeros((self.G, data.N), dtype='Float64')
            #log_col_sum = np.zeros(data.N,dtype='Float64')  # array of column sums of log_l
            log_pi = np.log(self.pi)  # array of log mixture coefficients

            # computing log posterior distribution
            for i in range(self.G):
                log_l[i] = log_pi[i] + self.components[i].pdf(data)

            # Partially supervised training
            # For sequences with partial information use a weight vector s.t.
            # P[model|seq] = 1 if model = label[seq] and 0 else
            for i, cl in enumerate(data.labels): # for each class
                for o in cl: # for each observation in a class
                    v = log_l[i, o]
                    p_vec = np.zeros(self.G, dtype='Float64')
                    p_vec[:] = float('-inf')
                    p_vec[i] = v
                    log_l[:, o] = p_vec


            # computing data log likelihood as criteria of convergence
            # log_l is normalized in-place and likelihood is returned as log_p
            (log_l, log_p) = mixextend.get_normalized_posterior_matrix(log_l)

            # adjusting posterior for lower hierarchy mixtures
            if mix_posterior is not None:
                # multiplying in the posterior of upper hierarchy mixture
                log_l[:, j] = log_l[:, j] + log_mix_posterior[j]


            # we have to take the parameter prior into account to form the objective function
            # If we assume independence between parameters in different components, the prior
            # contribution is given by a product over the individual component and structure priors
            try:
                log_prior = self.prior.pdf(self)
            except ValueError:  # catch zero probability under prior

                raise ConvergenceFailureEM, "Zero probability under prior."

            # calculate objective function
            log_p += log_prior
            # checking for convergence
            diff = (log_p - log_p_old)

            if log_p_old != -1.0 and not silent and step > 0:
                if tilt and step <= self.nr_tilt_steps:
                    sys.stdout.write("TILT Step " + str(step) + ": log posterior: " + str(log_p) + "\n")
                else:
                    sys.stdout.write("Step " + str(step) + ": log posterior: " + str(log_p) + "   (diff=" + str(diff) + ")\n")

            if diff < 0.0 and step > 1 and abs(diff / log_p_old) > self.err_tol:
                #print log_p,log_p_old, diff,step,abs(diff / log_p_old)
                #print "WARNING: EM divergent."
                raise ConvergenceFailureEM, "Convergence failed, EM divergent: "

            if (not tilt or (tilt and step + 1 >= self.nr_tilt_steps)) and delta >= diff and max_iter != 1:
                if not silent:
                    sys.stdout.write("Convergence reached with log_p " + str(log_p) + " after " + str(step) + " steps.\n")
                if self.identFlag:
                    self.identifiable()
                return (log_l, log_p)

            log_p_old = log_p
            if step == max_iter:
                if not silent:
                    sys.stdout.write("Max_iter " + str(max_iter) + " reached -> stopping\n")

                if self.identFlag:
                    self.identifiable()
                return (log_l, log_p)


            # compute posterior likelihood matrix from log posterior
            l = np.exp(log_l)

            # deterministic annealing, shifting posterior toward uniform distribution.
            if tilt and step + 1 <= self.nr_tilt_steps and mix_posterior is None:
                h = self.heat - (step * (self.heat / (self.nr_tilt_steps))  )
                for j in range(data.N):
                    uni = 1.0 / self.G
                    tilt_l = (uni - l[:, j]) * h
                    l[:, j] += tilt_l

            # variables for component fixing
            fix_pi = 1.0
            unfix_pi = 0.0
            fix_flag = 0   # flag for fixed mixture components

            # update component parameters and mixture weights
            for i in range(self.G):
                if self.compFix[i] & 2:   # pi[i] is fixed
                    fix_pi -= self.pi[i]
                    fix_flag = 1
                    continue
                else:
                    # for mixtures of mixtures we need to multiply in the mix_pi[i]s
                    if mix_pi is not None:
                        self.pi[i] = ( l[i, :].sum() + self.prior.piPrior.alpha[i] - 1.0 ) / ((data.N * mix_pi) + self.prior.piPrior.alpha_sum - self.G )
                    else:
                        self.pi[i] = ( l[i, :].sum() + self.prior.piPrior.alpha[i] - 1.0 ) / (data.N + ( self.prior.piPrior.alpha_sum - self.G) )

                    unfix_pi += self.pi[i]

                if self.compFix[i] & 1:
                    continue
                else:
                    # Check for model structure
                    if not self.struct:
                        self.prior.compPrior.mapMStep(self.components[i], l[i, :], data, self.pi[i])

            # if there is a model structure we update the leader distributions only
            if self.struct:
                for j in range(len(self.components[0])):
                    for k in self.leaders[j]:
                        # compute group posterior
                        # XXX extension function for pooled posterior ?
                        g_post = copy.deepcopy(l[k, :])
                        g_pi = self.pi[k]
                        for memb in self.groups[j][k]:
                            g_post += l[memb, :]
                            g_pi += self.pi[memb]

                        if isinstance(self.components[k][j], MixtureModel):
                            self.prior.compPrior[j].mapMStep(self.components[k][j], g_post, data.singleFeatureSubset(j), g_pi)
                        else:
                            try:
                                self.prior.compPrior[j].mapMStep(self.components[k][j], g_post, data.getInternalFeature(j))
                            except InvalidPosteriorDistribution:
                                raise

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

            sys.stdout.flush()
            step += 1


    def updateStructureBayesian(self, data, objFunction='MAP', silent=1):
        """
        Updating structure by chosing optimal local merge with respect to the posterior.


        Features: - store merges in a learning history to prevent recomputation of merge parameters
                  - compute parameters of candidate structures from expected sufficient statistics of groups to be merged


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

        # apply labels to posterior matrix
        # XXX there should be a more efficient way to do this ... XXX
        for i, cl in enumerate(data.labels): # for each class
            for o in cl: # for each observation in a class
                ind = range(self.G)
                ind.pop(i)
                for ng in ind:
                    l[:, ng, o] = float('-inf')

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

            # extracting current feature from the DataSet
            if isinstance(self.components[0][j], MixtureModel): # XXX
                data_j = data.singleFeatureSubset(j)
            else:
                data_j = data.getInternalFeature(j)

            tau_pool = np.zeros(data.N, dtype='Float64')
            for lead in self.leaders[j]:
                el_dist = copy.copy(self.components[lead][j])

                tau_pool = copy.copy(tau[lead, :])
                pi_pool = self.pi[lead]
                for z in self.groups[j][lead]:
                    tau_pool += tau[z, :]
                    pi_pool += self.pi[z]

                stat = el_dist.sufficientStatistics(tau_pool, data_j)
                M = CandidateGroup(el_dist, np.sum(tau_pool), pi_pool, stat)
                L[(lead,) + tuple(self.groups[j][lead])] = M

            while not term:
                best_dist = None   # accepted candidate distributions
                best_post = float('-inf')   # corresponding posteriors
                best_indices = None
                best_l_j = l[j]
                best_log_prior_list_j = log_prior_list[j]

                for mc1 in range(len(temp_leaders[j])):
                    merge_cand1 = temp_leaders[j][mc1]
                    for mc2 in range(mc1 + 1, len(temp_leaders[j])):
                        merge_cand2 = temp_leaders[j][mc2]
                        if not silent:
                            print "-------------------"
                            print merge_cand1, " -> ", merge_cand2
                            print self.components[merge_cand1][j]
                            print self.components[merge_cand2][j]

                        nr_leaders_j = len(temp_leaders[j]) - 1
                        cand_group_j = temp_groups[j][merge_cand1] + [merge_cand2] + temp_groups[j][merge_cand2]

                        hist_ind_part1 = (merge_cand1,) + tuple(temp_groups[j][merge_cand1])
                        hist_ind_part2 = (merge_cand2,) + tuple(temp_groups[j][merge_cand2])
                        hist_ind_complete = hist_ind_part1 + hist_ind_part2

                        recomp = 0
                        if L.has_key(hist_ind_complete):
                            recomp = 1
                        if not silent:
                            print "\ncandidate model structure: XXX"
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

                        # applying labels
                        for i, cl in enumerate(data.labels): # for each class
                            for o in cl: # for each observation in a class
                                ind = range(self.G)
                                ind.pop(i)
                                for ng in ind:
                                    l_j_1[ng, o] = float('-inf')

                        # get updated unnormalized posterior matrix
                        g = g_wo_j + l_j_1

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
                            print 'Posterior:', post_1, '=', lk_1, '+', log_prior_1

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
                        print "--- Winner ---"
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


    def classify(self, data, labels=None, entropy_cutoff=None, silent=0, EStep=None, EStepParam=None):
        """
        Classification of input 'data'.
        Assignment to mixture components by maximum likelihood over
        the component membership posterior. No parameter reestimation.

        Classification of labelled samples is fixed a priori and overrrides the assignment by
        maximum posterior.

        @param data: ConstrainedDataSet object
        @param labels: optional sample IDs
        @param entropy_cutoff: entropy threshold for the posterior distribution. Samples which fall
        above the threshold will remain unassigned
        @param silent: 0/1 flag, toggles verbose output
        @param EStep: function implementing the EStep, by default self.EStep
        @param EStepParam: additional paramenters for more complex EStep implementations


        @return: list of class labels
        """

        assert isinstance(data, ConstrainedDataSet)

        # standard classification
        c = BayesMixtureModel.classify(self, data, labels=labels, entropy_cutoff=entropy_cutoff, silent=1, EStep=EStep, EStepParam=EStepParam)

        # apply labels
        for i, cl in enumerate(data.labels): # for each class
            for o in cl: # for each observation in a class
                c[o] = i

        if not silent:
            # printing out the clusters
            cluster = {}
            en = {}
            for j in range(-1, self.G, 1):
                cluster[j] = []

            for i in range(data.N):
                cluster[c[i]].append(data.sampleIDs[i])

            print "\n** Clustering **"
            for j in range(self.G):
                print "Cluster ", j, ', size', len(cluster[j])
                print cluster[j], "\n"

            print "Unassigend due to entropy cutoff:"
            print cluster[-1], "\n"

        return c
