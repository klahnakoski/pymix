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
import numpy as np
from ..util.errors import InvalidPosteriorDistribution
from ..util.constrained_dataset import ConstrainedDataSet
from ..util.dataset import DataSet
from ..util.maths import sum_logs
from .mixture import MixtureModel


class LabeledMixtureModel(MixtureModel):
    """
    Class for a mixture model containing the label constrained
    version of the E-Step See
    A. Schliep, C. Steinhoff,
    A. A. Schonhuth Robust inference of groups in gene expression
    time-courses using mixtures of HMMs Bioinformatics. 2004 Aug 4;20
    Suppl 1:I283-I289 (Proceedings of the ISMB 2004).
    for details
    """

    def __init__(self, G, pi, components, compFix=None, struct=0):
        MixtureModel.__init__(self, G, pi, components, compFix=compFix, struct=struct, identifiable=0)

    def EM(self, data, max_iter, delta, silent=False, mix_pi=None, mix_posterior=None, tilt=0):
        """
        Reestimation of mixture parameters using the EM algorithm.
        This method do some initial checking and call the EM from MixtureModel with the constrained labels E step

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

        assert isinstance(data, ConstrainedDataSet), 'Data set does not contain labels, Labeled EM can not be performed'
        assert data.labels != None, 'Data set does not contain labels, Labeled EM can not be performed'
        assert data.noLabels <= self.G, 'Number of components should be equal or greatedr then the number of label classes'

        return MixtureModel.EM(self, data, max_iter, delta, silent=silent, mix_pi=mix_pi,
            mix_posterior=mix_posterior, tilt=tilt, EStep=self.EStep, EStepParam=None)

    def EStep(self, data, mix_posterior=None, mix_pi=None, EStepParam=None):
        """Reestimation of mixture parameters using the EM algorithm.

        @param data: DataSet object
        @param mix_pi: [internal use only] necessary for the reestimation of
        mixtures as components
        @param mix_posterior:[internal use only] necessary for the reestimation of
        mixtures as components
        @param EStepParam: additional paramenters for more complex EStep implementations, in
        this implementaion it is ignored

        @return: tuple of log likelihood matrices and sum of log-likelihood of components
        """
        # computing log posterior distribution
        #[log_l,log_p] = MixtureModel.EStep(self,data,mix_posterior,mix_pi,None)

        log_l = np.zeros((self.G, data.N), dtype='Float64')
        log_col_sum = np.zeros(data.N, dtype='Float64')  # array of column sums of log_l
        log_pi = np.log(self.pi)  # array of log mixture coefficients

        # compute log of mix_posterior (if present)
        if mix_posterior is not None:
            log_mix_posterior = np.log(mix_posterior)

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

        log_col_sum = np.zeros(data.N, dtype='Float64')  # array of column sums of log_l
        for j in range(data.N):
            log_col_sum[j] = sum_logs(log_l[:, j]) # sum over jth column of log_l
            # if posterior is invalid, check for model validity
            if log_col_sum[j] == float('-inf'):
                # if self is at the top of hierarchy, the model is unable to produce the
                # sequence and an exception is raised. Otherwise normalization is not necessary.
                if mix_posterior is None and not mix_pi:
                    #print "\n---- Invalid -----\n",self,"\n----------"
                    #print "\n---------- Invalid ---------------"
                    #print "mix_pi = ", mix_pi
                    #print "x[",j,"] = ", data.getInternalFeature(j)
                    #print "l[:,",j,"] = ", log_l[:,j]
                    #print 'data[',j,'] = ',data.dataMatrix[j]
                    raise InvalidPosteriorDistribution, "Invalid posterior distribution."
            # for valid posterior, normalize and go on
            else:
                # normalizing log posterior
                log_l[:, j] = log_l[:, j] - log_col_sum[j]
                # adjusting posterior for lower hierarchy mixtures
                if mix_posterior is not None:
                    # multiplying in the posterior of upper hierarch mixture
                    log_l[:, j] = log_l[:, j] + log_mix_posterior[j]
        return log_l, np.sum(log_col_sum)

    def modelInitialization(self, data, rtype=1, missing_value=None):
        """
        Perform model initialization given a random assigment of the
        data to the models.

        @param data: DataSet object
        @param rtype: type of random assignments.
        0 = fuzzy assingment
        1 = hard assingment
        @param missing_value: missing symbol to be ignored in parameter estimation (if applicable)
        """
        assert isinstance(data, ConstrainedDataSet), 'Data set does not contain labels, Labeled EM can not be performed'
        assert data.labels != None, 'Data set does not contain labels, Labeled EM can not be performed'
        assert data.noLabels <= self.G, 'Number of components should be equal or greater than the number of label classes'

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

        # peform label assigments (non random!)
        for i, cl in enumerate(data.labels): # for each class
            for o in cl: # for each observation in a class
                p_vec = np.zeros(self.G, dtype='Float64')
                p_vec[i] = 1.0
                l[:, o] = p_vec

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
                last_index = 0
                for j in range(len(self.components[i])):
                    if isinstance(self.components[i][j], MixtureModel):
                        self.components[i][j].modelInitialization(data.getInternalFeature(j), rtype=rtype, missing_value=missing_value)
                    else:
                        loc_l = l[i, :]
                        if missing_value:
                            # masking missing values from parameter estimation
                            for k, d in enumerate(data.getInternalFeature(j)):
                                if d == missing_value:
                                    loc_l[k] = 0.0
                        self.components[i][j].MStep(loc_l, data.getInternalFeature(j))

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

    def classify(self, data, labels=None, entropy_cutoff=None, silent=0):
        """
        Classification of input 'data'.
        Assignment to mixture components by maximum likelihood over
        the component membership posterior. No parameter reestimation.

        @param data: DataSet object
        @param labels: optional sample IDs
        @param entropy_cutoff: entropy threshold for the posterior distribution. Samples which fall
        above the threshold will remain unassigned
        @param silent: 0/1 flag, toggles verbose output

        @return: list of class labels
        """
        return MixtureModel.classify(self, data, labels=labels, entropy_cutoff=entropy_cutoff, silent=silent, EStep=self.EStep, EStepParam=None)

