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

from ..distributions.prob import ProbDistribution


class PriorDistribution(ProbDistribution):
    """
    Prior distribution base class for the Bayesian framework
    """
    def pdf(self, m):
        """
        Returns the log-density of the ProbDistribution object(s) 'm' under the
        prior.

        @param m: single appropriate ProbDistribution object or list of ProbDistribution objects
        """
        raise NotImplementedError, "Needs implementation"

    def posterior(self,m,x):
        raise NotImplementedError, "Needs implementation"

    def marginal(self,x):
        raise NotImplementedError, "Needs implementation"

    def mapMStep(self, dist, posterior, data, mix_pi=None, dist_ind = None):
        """
        Maximization step of the maximum aposteriori EM procedure. Reestimates the distribution parameters
        of argument 'dist' using the posterior distribution, the data and a conjugate prior.

        MUST accept either numpy or DataSet object of appropriate values. numpys are used as input
        for the atomar distributions for efficiency reasons.

        @param dist: distribution whose parameters are to be maximized
        @param posterior: posterior distribution of component membership
        @param data: DataSet object or 'numpy' of samples
        @param mix_pi: mixture weights, necessary for MixtureModels as components.
        @param dist_ind: optional index of 'dist', necessary for ConditionalGaussDistribution.mapMStep (XXX)
        """
        raise NotImplementedError


    def mapMStepMerge(self, group_list):
        """
        Computes the MAP parameter estimates for a candidate merge in the structure
        learning based on the information of two CandidateGroup objects.

        @param group_list: list of CandidateGroup objects
        @return: CandidateGroup object with MAP parameters
        """
        raise NotImplementedError

    def mapMStepSplit(self, toSplitFrom, toBeSplit):
        """
        Computes the MAP parameter estimates for a candidate merge in the structure
        learning based on the information of two CandidateGroup objects.

        @return: CandidateGroup object with MAP parameters

        """
        raise NotImplementedError

    def updateHyperparameters(self, dists, posterior, data):
        """
        Update the hyperparameters in an empirical Bayes fashion.

        @param dists: list of ProbabilityDistribution objects
        @param posterior: numpy matrix of component membership posteriors
        @param data: DataSet object
        """
        raise NotImplementedError
