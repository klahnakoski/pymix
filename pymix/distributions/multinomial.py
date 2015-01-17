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
import numpy as np
from .prob import ProbDistribution
from ..util.errors import InvalidPosteriorDistribution, InvalidDistributionInput
from ..util.dataset import DataSet
from pymix.vendor.ghmm.emission_domain import IntegerRange
from pyLibrary.testing.fuzzytestcase import assertAlmostEqual


class MultinomialDistribution(ProbDistribution):
    """
    Multinomial Distribution
    """

    def __init__(self, dimension, M, phi, alphabet=None, parFix=None):
        """
        Constructor

        @param M: number of possible outcomes (0 to M-1)
        @param dimension: number of values in each sample
        @param phi:= discrete distribution of N objects
        @param alphabet: Alphabet object (optional)
        @param parFix: list of flags to determine if any elements of phi should be fixed
        """
        assert len(phi) == M, "Invalid number of parameters for MultinomialDistribution."
        try:
            assert abs((1.0 - sum(phi))) < 1e-12, str(phi) + ": " + str(1.0 - sum(phi))  #  check parameter validity
        except Exception, e:
            raise e

        self.dimension = dimension  # length of input vectors, corresponds to p in MixtureModel
        self.M = M
        self.suff_p = M  # length of the sufficient statistics, equal to size of alphabet

        # in case there is no alphabet specified IntegerRange is used
        if alphabet:
            assert len(alphabet) == self.M, "Size of alphabet and M does not match: " + str(len(alphabet)) + " != " + str(self.M)
            self.alphabet = alphabet
        else:
            self.alphabet = IntegerRange(0, self.M)

        if parFix == None:
            self.parFix = np.array([0] * self.M)
        else:
            assert len(parFix) == self.M, "Invalid length of parFix vector."
            self.parFix = np.array(parFix)

        # number of free parameters is M-1 minus the number of fixed entries in phi
        self.freeParams = M - 1 - sum(self.parFix)

        self.phi = np.array(phi, dtype='Float64')

        # minimal value for any component of self.phi, enforced in MStep
        self.min_phi = (1.0 / self.M) * 0.001

    def __eq__(self, other):
        res = False
        if isinstance(other, MultinomialDistribution):
            if other.dimension == self.dimension and other.M == self.M and np.allclose(other.phi, self.phi):
                res = True
        return res

    def __copy__(self):
        "Interface for the copy.copy function"
        return MultinomialDistribution(self.dimension, self.M, copy.deepcopy(self.phi), self.alphabet, parFix=self.parFix)

    def __str__(self):
        outstr = "Multinom(M = " + str(self.M) + ", N = " + str(self.dimension) + " ) : " + str(self.phi) #+"\n"
        #outstr += str(self.alphabet) + "\n"
        return outstr

    def pdf(self, data):
        # Note: The multinomial coefficient is omitted in the implementation.
        # Result is proportional to the true log densitiy which is sufficient for
        # the EM.
        # gsl computes the true density, including the multinomial coefficient normalizing constant
        # therefore it is less efficient than the implementation below
        if isinstance(data, DataSet):
            x = data.internalData
        elif hasattr(data, "__iter__"):
            x = data
        else:
            raise TypeError, "Unknown/Invalid input type."

        # switch to log scale for density computation
        log_phi = np.log(self.phi)

        # computing un-normalized density
        res = np.zeros(len(x), dtype='Float64')
        for j in range(len(x)):
            for i in range(self.M):
                res[j] += (log_phi[i] * x[j, i])

        res2 = np.sum(x * log_phi, axis=1)
        assertAlmostEqual(res, res2)

        return res

    def sample(self, native=False):
        sample = []
        for i in range(self.dimension):
            sum = 0.0
            p = random.random()
            for k in range(self.M):
                sum += self.phi[k]
                if sum >= p:
                    break
            sample.append(k)

        return map(self.alphabet.external, sample)

    def sampleSet(self, nr):
        return [self.sample() for i in range(nr)]

    def MStep(self, posterior, data, mix_pi=None):
        if isinstance(data, DataSet):
            x = data.internalData
        elif hasattr(data, "__iter__"):
            x = data
        else:
            raise TypeError, "Unknown/Invalid input to MStep."

        ind = np.where(self.parFix == 0)[0]
        fix_flag = 0
        fix_phi = 1.0
        dsum = 0.0

        # reestimating parameters
        for i in range(self.M):
            if self.parFix[i] == 1:
                fix_phi -= self.phi[i]
                fix_flag = 1
                continue
            else:
                est = np.dot(x[:, i], posterior)
                self.phi[i] = est
                dsum += est

        if dsum == 0.0:
            raise InvalidPosteriorDistribution, "Invalid posterior in MStep."

        # normalzing parameter estimates
        self.phi[ind] = (self.phi[ind] * fix_phi) / dsum

        adjust = 0  # adjusting flag
        for i in range(self.M):
            if self.parFix[i] == 0 and self.phi[i] < self.min_phi:
                adjust = 1
                self.phi[i] = self.min_phi

        # renormalizing the adjusted parameters if necessary
        if adjust:
            dsum = sum(self.phi[ind])
            self.phi[ind] = (self.phi[ind] * fix_phi) / dsum

    def isValid(self, x):
        if sum(map(self.alphabet.isAdmissable, x)) != self.dimension:
            raise InvalidDistributionInput, "\n\tInvalid data: " + str(x) + " in MultinomialDistribution(" + str(self.alphabet.listOfCharacters) + ")."

    def formatData(self, x):
        count = [0] * self.M #  np.zeros(self.M)

        # special case of p = 1
        if len(x) == 1:
            self.isValid(str(x[0]))
            count[self.alphabet.internal(str(x[0]))] = 1

            return [self.M, count]

        for i in range(self.M):
            self.isValid(x)
            count[i] = sum([1 for c in x if c == self.alphabet.listOfCharacters[i]])

        return [self.M, count]

    def flatStr(self, offset):
        offset += 1
        return "\t" * offset + ";Mult;" + str(self.dimension) + ";" + str(self.M) + ";" + str(self.phi.tolist()) + ";" + str(self.alphabet.listOfCharacters) + ";" + str(self.parFix.tolist()) + "\n"

    def posteriorTraceback(self, x):
        return self.pdf(x)

    def merge(self, dlist, weights):
        raise DeprecationWarning, 'Part of the outdated structure learning implementation.'

