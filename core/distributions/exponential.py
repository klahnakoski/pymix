import math
import random
import numpy as np
from core.distributions.prob import ProbDistribution
from core.pymix_util.errors import InvalidDistributionInput
from core.pymix_util.dataset import DataSet


class ExponentialDistribution(ProbDistribution):
    """
    Exponential distribution
    """

    def __init__(self, lambd):
        """
        Constructor

        @param lambd: shape parameter lambda
        """

        self.p = self.suff_p = 1
        self.lambd = lambd  # lambd is a rate: 0.0 < lambd <= 1.0
        self.freeParams = 1

    def __copy__(self):
        return ExponentialDistribution(self.lambd)


    def __str__(self):
        return "Exponential:  [" + str(self.lambd) + "]"

    def __eq__(self, other):
        if not isinstance(other, ExponentialDistribution):
            return False
        if not self.lambd == other.lambd:
            return False
        return True


    def pdf(self, data):
        if isinstance(data, DataSet):
            assert data.internalData is not None, "Internal data not initialized."
            nr = len(data.internalData)
            assert data.internalData.shape == (nr, 1), 'shape = ' + str(data.internalData.shape)

            x = np.transpose(data.internalData)[0]

        elif hasattr(data, "__iter__"):
            nr = len(data)

            if data.shape == (nr, 1):  # data format needs to be changed
                x = np.transpose(data)[0]
            elif data.shape == (nr,):
                x = data
            else:
                raise TypeError, 'Invalid data shape: ' + str(data.shape)
        else:
            raise TypeError, "Unknown/Invalid input type:" + str(type(data))

        return math.log(self.lambd) + (-self.lambd * x)  # XXX pure Python implementation for now


    def sample(self):
        return random.expovariate(self.lambd)

    def MStep(self, posterior, data, mix_pi=None):
        # data has to be reshaped for parameter estimation
        if isinstance(data, DataSet):
            x = data.internalData[:, 0]
        elif hasattr(data, "__iter__"):
            x = data[:, 0]
        else:
            raise TypeError, "Unknown/Invalid input to MStep."

        self.lambd = posterior.sum() / np.dot(posterior, x)


    def isValid(self, x):
        "Checks whether 'x' is a valid argument for the distribution."
        try:
            float(x)
        except ValueError:
            #print "Invalid data: ",x,"in ExponentialDistribution."
            raise InvalidDistributionInput, "\n\tInvalid data: " + str(x) + " in ExponentialDistribution."

        if x < 0:
            raise InvalidDistributionInput, "\n\tInvalid data: negative float " + str(x) + " in ExponentialDistribution."

    def formatData(self, x):
        """
        """
        if type(x) == list and len(x) == 1:
            x = x[0]
        self.isValid(x)
        return [self.p, [x]]

    def flatStr(self, offset):
        offset += 1
        return "\t" * offset + ";Exp;" + str(self.lambd) + "\n"

    def posteriorTraceback(self, x):
        return self.pdf(x)
