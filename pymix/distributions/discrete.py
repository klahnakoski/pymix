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
from .multinomial import MultinomialDistribution
from pyLibrary.debugs.logs import Log
from ..util.errors import InvalidPosteriorDistribution, InvalidDistributionInput
from ..util.dataset import DataSet


class DiscreteDistribution(MultinomialDistribution):
    """
    This is the special case of a MultinomialDistribution with p = 1, that is a simple univariate discrete
    distribution. Certain key functions are overloaded for increased efficiency.
    """

    def __init__(self, M, phi, alphabet=None, parFix=None):
        """
        @param M: size of alphabet
        @param phi: distribution parameters
        @param alphabet: Alphabet object (optional)
        @param parFix: list of flags to determine if any elements of phi should be fixed
        """

        MultinomialDistribution.__init__(self, 1, M, phi, alphabet=alphabet, parFix=parFix)
        self.suff_p = 1

    def __str__(self):
        outstr = "DiscreteDist(M = " + str(self.M) + "): " + str(self.phi) #+"\n"
        #outstr += str(self.alphabet) + "\n"
        return outstr

    def __copy__(self):
        return DiscreteDistribution(self.M, copy.deepcopy(self.phi), self.alphabet, parFix=self.parFix)

    def pdf(self, data):
        if isinstance(data, DataSet):
            assert data.dimension == 1
            x = data.getInternalFeature(0)
        elif hasattr(data, "__iter__"):
            x = data
        else:
            raise TypeError, "Unknown/Invalid input type."

        # switch to log scale for density computation
        log_phi = np.log(self.phi)

        # computing un-normalized density
        res = log_phi[x[:, 0].astype('Int32')]
        return res

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
                i_ind = np.where(x == i)[0]
                est = np.sum(posterior[i_ind])
                self.phi[i] = est
                dsum += est

        if dsum == 0.0:
            print self
            print posterior

            raise InvalidPosteriorDistribution, "Invalid posterior in MStep."

        # normalzing parameter estimates
        self.phi[ind] = (self.phi[ind] * fix_phi) / dsum

        adjust = 0  # adjusting flag
        for i in range(self.M):
            if self.parFix[i] == 0 and self.phi[i] < self.min_phi:
                #print "---- enforcing minimal phi -----"
                adjust = 1
                self.phi[i] = self.min_phi

        # renormalizing the adjusted parameters if necessary
        if adjust:
            dsum = sum(self.phi[ind])
            self.phi[ind] = (self.phi[ind] * fix_phi) / dsum

    def sample(self, native=False):
        for i in range(self.dimension):
            sum = 0.0
            p = random.random()
            for k in range(self.M):
                sum += self.phi[k]
                if sum >= p:
                    break
        return self.alphabet.external(k)

    def sampleSet(self, nr):
        res = []
        for i in range(nr):
            res.append(self.sample())
        return res

    def sufficientStatistics(self, posterior, data):
        stat = np.zeros(self.M, dtype='Float64')
        for i in range(self.M):
            i_ind = np.where(data == i)[0]
            stat[i] = np.sum(posterior[i_ind])
        return stat

    def formatData(self, x):
        self.isValid(x)
        if type(x) == list:
            assert len(x) == 1
            internal = self.alphabet.internal(x[0])
        else:
            internal = self.alphabet.internal(x)
        return [1, [internal]]

    def isValid(self, x):
        if isinstance(x, (basestring, int, float)):
            if not self.alphabet.isAdmissable(x):
                Log.error("Invalid data: {{x}} in DiscreteDistribution({{chars}})", {
                    "chars": self.alphabet.listOfCharacters,
                    "x": repr(x)
                })
        elif type(x) == list and len(x) == 1:
            self.isValid(x[0])
        else:
            Log.error("Invalid data: {{x}} in DiscreteDistribution({{chars}})", {
                "chars": self.alphabet.listOfCharacters,
                "x": repr(x)
            })

    def flatStr(self, offset):
        offset += 1
        return "\t" * offset + ";Discrete;" + str(self.M) + ";" + str(self.phi.tolist()) + ";" + str(self.alphabet.listOfCharacters) + ";" + str(self.parFix.tolist()) + "\n"
