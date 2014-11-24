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

class ProbDistribution:
    """
    Base class for all probability distributions.
    """

    def __init__(self):
        """
        Constructor
        """
        pass

    def __eq__(self, other):
        """
        Interface for the '==' operation

        @param other: object to be compared
        """
        raise NotImplementedError

    def __str__(self):
        """
        String representation of the DataSet

        @return: string representation
        """
        raise NotImplementedError

    def __copy__(self):
        "Interface for the copy.copy function"
        raise NotImplementedError

    def pdf(self, data):
        """
        Density function.
        MUST accept either numpy or DataSet object of appropriate values. We use numpys as input
        for the atomar distributions for efficiency reasons (The cleaner solution would be to construct
        DataSet subset objects for the different features and we might switch over to doing that eventually).

        @param data: DataSet object or numpy array

        @return: log-value of the density function for each sample in 'data'
        """
        raise NotImplementedError


    def MStep(self, posterior, data, mix_pi=None):
        """
        Maximization step of the EM procedure. Reestimates the distribution parameters
        using the posterior distribution and the data.

        MUST accept either numpy or DataSet object of appropriate values. numpys are used as input
        for the atomar distributions for efficiency reasons

        @param posterior: posterior distribution of component membership
        @param data: DataSet object or 'numpy' of samples
        @param mix_pi: mixture weights, necessary for MixtureModels as components.

        """
        raise NotImplementedError


    def sample(self):
        """
        Samples a single value from the distribution.

        @return: sampled value
        """
        raise NotImplementedError, "Needs implementation"

    def sampleSet(self, nr):
        """
        Samples several values from the distribution.

        @param nr: number of values to be sampled.

        @return: sampled values
        """
        raise NotImplementedError, "Needs implementation"

    def sufficientStatistics(self, posterior, data):
        """
        Returns sufficient statistics for a given data set and posterior.

        @param posterior: numpy vector of component membership posteriors
        @param data: numpy vector holding the data

        @return: list with dot(posterior, data) and dot(posterior, data**2)
        """
        raise NotImplementedError, "Needs implementation"


    def isValid(self, x):
        """
        Checks whether 'x' is a valid argument for the distribution and raises InvalidDistributionInput
        exception if that is not the case.

        @param x: single sample in external representation, i.e.. an entry of DataSet.dataMatrix

        @return: True/False flag
        """
        raise NotImplementedError

    def formatData(self, x):
        """
        Formats samples 'x' for inclusion into DataSet object. Used by DataSet.internalInit()

        @param x: list of samples

        @return: two element list: first element = dimension of self, second element = sufficient statistics for samples 'x'
        """
        return [self.dimension, x]


    def flatStr(self, offset):
        """
        Returns the model parameters as a string compatible
        with the WriteMixture/ReadMixture flat file
        format.

        @param offset: number of '\t' characters to be used in the flatfile.
        """
        raise NotImplementedError

    def posteriorTraceback(self, x):
        """
        Returns the decoupled posterior distribution for each
        sample in 'x'. Used for analysis of clustering results.

        @param x: list of samples

        @return: decoupled posterior
        """
        raise NotImplementedError

    def update_suff_p(self):
        """
        Updates the .suff_p field.
        """
        return self.suff_p

    def merge(self, dlist, weights):
        """
        Merges 'self' with the distributions in'dlist' by an
        convex combination of the parameters as determined by 'weights'

        @param dlist: list of distribution objects of the same type as 'self'
        @param weights: list of weights, need not to sum up to one
        """
        raise NotImplementedError

