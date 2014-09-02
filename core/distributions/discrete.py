import copy
import random
import numpy as np
from core.distributions.multinomial import MultinomialDistribution
from core.pymix_util.errors import InvalidPosteriorDistribution, InvalidDistributionInput
from core.pymix_util.dataset import DataSet


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
            assert data.p == 1
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

    def sample(self):
        for i in range(self.p):
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
            internal = self.alphabet.internal(str(x[0]))
        else:
            internal = self.alphabet.internal(str(x))
        return [1, [internal]]

    def isValid(self, x):
        if type(x) == str or type(x) == int or type(x) == float:
            if not self.alphabet.isAdmissable(str(x)):
                raise InvalidDistributionInput, "\n\tInvalid data: " + str(x) + " in DiscreteDistribution(" + str(self.alphabet.listOfCharacters) + ")."
        else:
            if type(x) == list and len(x) == 1:
                self.isValid(x[0])
            else:
                raise InvalidDistributionInput, "\n\tInvalid data: " + str(x) + " in DiscreteDistribution(" + str(self.alphabet.listOfCharacters) + ")."

    def flatStr(self, offset):
        offset += 1
        return "\t" * offset + ";Discrete;" + str(self.M) + ";" + str(self.phi.tolist()) + ";" + str(self.alphabet.listOfCharacters) + ";" + str(self.parFix.tolist()) + "\n"
