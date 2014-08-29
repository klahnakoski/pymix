import copy
import random
import numpy
from core import assertAlmostEqual
from core.distributions.prob import ProbDistribution
from core.pymix_util.errors import InvalidPosteriorDistribution, InvalidDistributionInput
from core.pymix_util.alphabet import IntegerRange
from core.pymix_util.dataset import DataSet
from core.util.env.logs import Log


class MultinomialDistribution(ProbDistribution):
    """
    Multinomial Distribution
    """

    def __init__(self, p, M, phi, alphabet=None, parFix=None):
        """
        Constructor

        @param M: number of possible outcomes (0 to M-1)
        @param p: number of values in each sample
        @param phi:= discrete distribution of N objects
        @param alphabet: Alphabet object (optional)
        @param parFix: list of flags to determine if any elements of phi should be fixed
        """
        assert len(phi) == M, "Invalid number of parameters for MultinomialDistribution."
        assert abs((1.0 - sum(phi))) < 1e-12, str(phi) + ": " + str(1.0 - sum(phi)) # check parameter validity

        self.p = p # lenght of input vectors, corresponds to p in MixtureModel
        self.M = M
        self.suff_p = M  # length of the sufficient statistics, equal to size of alphabet

        # in case there is no alphabet specified IntegerRange is used
        if alphabet:
            assert len(alphabet) == self.M, "Size of alphabet and M does not match: " + str(len(alphabet)) + " != " + str(self.M)
            self.alphabet = alphabet
        else:
            self.alphabet = IntegerRange(0, self.M)

        if parFix == None:
            self.parFix = numpy.array([0] * self.M)
        else:
            assert len(parFix) == self.M, "Invalid length of parFix vector."
            self.parFix = numpy.array(parFix)

        # number of free parameters is M-1 minus the number of fixed entries in phi
        self.freeParams = M - 1 - sum(self.parFix)

        self.phi = numpy.array(phi, dtype='Float64')

        # minimal value for any component of self.phi, enforced in MStep
        self.min_phi = ( 1.0 / self.M ) * 0.001

    def __eq__(self, other):
        res = False
        if isinstance(other, MultinomialDistribution):
            if other.p == self.p and other.M == self.M and numpy.allclose(other.phi, self.phi):
                res = True
        return res

    def __copy__(self):
        "Interface for the copy.copy function"
        return MultinomialDistribution(self.p, self.M, copy.deepcopy(self.phi), self.alphabet, parFix=self.parFix)

    def __str__(self):
        outstr = "Multinom(M = " + str(self.M) + ", N = " + str(self.p) + " ) : " + str(self.phi) #+"\n"
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
        elif isinstance(data, numpy.ndarray):
            x = data
        else:
            raise TypeError, "Unknown/Invalid input type."

        # switch to log scale for density computation
        log_phi = numpy.log(self.phi)

        # computing un-normalized density
        res = numpy.zeros(len(x), dtype='Float64')
        for j in range(len(x)):
            for i in range(self.M):
                res[j] += (log_phi[i] * x[j, i])

        res2 = numpy.sum(x * log_phi, axis=1)
        assertAlmostEqual(None, res, res2)

        return res

    def sample(self):
        sample = []
        for i in range(self.p):
            sum = 0.0
            p = random.random()
            for k in range(self.M):
                sum += self.phi[k]
                if sum >= p:
                    break
            sample.append(k)

        return map(self.alphabet.external, sample)

    def sampleSet(self, nr):
        res = []

        for i in range(nr):
            res.append(self.sample())

        return res

    def MStep(self, posterior, data, mix_pi=None):
        if isinstance(data, DataSet):
            x = data.internalData
        elif isinstance(data, numpy.ndarray):
            x = data
        else:
            raise TypeError, "Unknown/Invalid input to MStep."

        ind = numpy.where(self.parFix == 0)[0]
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
                est = numpy.dot(x[:, i], posterior)
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
        if sum(map(self.alphabet.isAdmissable, x)) != self.p:
            raise InvalidDistributionInput, "\n\tInvalid data: " + str(x) + " in MultinomialDistribution(" + str(self.alphabet.listOfCharacters) + ")."

    def formatData(self, x):
        count = [0] * self.M #  numpy.zeros(self.M)

        # special case of p = 1
        if len(x) == 1:
            self.isValid(str(x[0]))
            count[self.alphabet.internal(str(x[0]))] = 1

            return [self.M, count]

        for i in range(self.M):
            self.isValid(x)
            f = lambda x: x == self.alphabet.listOfCharacters[i]
            count[i] = sum(map(f, x))

        return [self.M, count]

    def flatStr(self, offset):
        offset += 1
        return "\t" * offset + ";Mult;" + str(self.p) + ";" + str(self.M) + ";" + str(self.phi.tolist()) + ";" + str(self.alphabet.listOfCharacters) + ";" + str(self.parFix.tolist()) + "\n"

    def posteriorTraceback(self, x):
        return self.pdf(x)

    def merge(self, dlist, weights):
        raise DeprecationWarning, 'Part of the outdated structure learning implementation.'

