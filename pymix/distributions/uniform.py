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
from .prob import ProbDistribution
from pymix.util.ghmm import random_mt
from ..util.errors import InvalidDistributionInput
from ..util.dataset import DataSet


class UniformDistribution(ProbDistribution):
    """
    Uniform distribution over a given intervall.
    """
    def __init__(self, start, end, *dummy_args):
        """
        Constructor

        @param start: begin of interval
        @param end: end of interval
        @param dummy_args: extra args for initialization using 4-tuples
        """
        assert start < end

        self.dimension = self.suff_p = 1
        self.freeParams = 0

        self.start = start
        self.end = end
        self.density = 1.0 / (end - start)
        self.log_density = np.log(self.density)   # compute log density value only once

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
                res[i] = self.log_density
            else:
                res[i] = float('-inf')

        return res

    def linear_pdf(self, x):
        if self.start <= x <= self.end:
            return self.density
        return 0

    def cdf(self, x):
        if x < self.start:
            return 0.0

        if x >= self.end:
            return 1.0

        return (x - self.start) / (self.end - self.start)

    def MStep(self, posterior, data, mix_pi=None):
        # nothing to be done...
        pass

    def sample(self, native=False):
        if native:
            return random.uniform(self.start, self.end)
        else:
            return (random_mt.float23()*(self.end-self.start))+self.start


    def sampleSet(self,nr):
        set = []
        for i in range(nr):
            set.append(self.sample())
        return set

    def isValid(self, x):
        try:
            float(x)
        except (ValueError):
            raise InvalidDistributionInput, "\n\tInvalid data in "+str(x)+" in UniformDistribution."

    def formatData(self,x):
        if isinstance(x,list) and len(x) == 1:
            x = x[0]
        self.isValid(x)  # make sure x is valid argument
        return [self.dimension,[x]]


    def flatStr(self,offset):
        raise NotImplementedError, "Boom !"

    def posteriorTraceback(self,x):
        raise NotImplementedError, "Kawoom !"

    def update_suff_p(self):
        return self.suff_p

    def merge(self,dlist, weights):
        raise NotImplementedError, "Kawoom !"

