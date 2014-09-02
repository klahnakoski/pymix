import random
import numpy as np
from core.distributions.prob import ProbDistribution
from core.pymix_util.errors import InvalidDistributionInput
from core.pymix_util.dataset import DataSet


class UniformDistribution(ProbDistribution):
    """
    Uniform distribution over a given intervall.
    """
    def __init__(self, start, end):
        """
        Constructor

        @param start: begin of interval
        @param end: end of interval
        """
        assert start < end

        self.p = self.suff_p = 1
        self.freeParams = 0

        self.start = start
        self.end = end
        self.density = np.log(1.0 / (end - start))   # compute log density value only once

    def __eq__(self,other):
        raise NotImplementedError

    def __str__(self):
        return "Uniform:  ["+str(self.start)+","+str(self.end)+"]"

    def __copy__(self):
        raise NotImplementedError

    def pdf(self,data):
        if isinstance(data, DataSet ):
            x = data.internalData
        elif hasattr(data, "__iter__"):
            x = data
        else:
            raise TypeError,"Unknown/Invalid input type."
        res = np.zeros(len(x),dtype='Float64')
        for i in range(len(x)):
            # density is self.density inside the interval and -inf (i.e. 0) outside
            if self.start <= x[i][0] <= self.end:
                res[i] = self.density
            else:
                res[i] = float('-inf')

        return res

    def MStep(self,posterior,data,mix_pi=None):
        # nothing to be done...
        pass

    def sample(self):
        return random.uniform( self.start, self.end)


    def sampleSet(self,nr):
        set = []
        for i in range(nr):
            set.append(self.sample())
        return set

    def isValid(self,x):
        try:
            float(x)
        except (ValueError):
            raise InvalidDistributionInput, "\n\tInvalid data in "+str(x)+" in UniformDistribution."

    def formatData(self,x):
        if isinstance(x,list) and len(x) == 1:
            x = x[0]
        self.isValid(x)  # make sure x is valid argument
        return [self.p,[x]]


    def flatStr(self,offset):
        raise NotImplementedError, "Boom !"

    def posteriorTraceback(self,x):
        raise NotImplementedError, "Kawoom !"

    def update_suff_p(self):
        return self.suff_p

    def merge(self,dlist, weights):
        raise NotImplementedError, "Kawoom !"

