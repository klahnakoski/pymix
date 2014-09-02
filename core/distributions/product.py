import copy
import numpy as np
from core.distributions.prob import ProbDistribution
from core.pymix_util.errors import InvalidDistributionInput
from core.pymix_util.dataset import DataSet


class ProductDistribution(ProbDistribution):
    """ Class for joined distributions for a vector of random variables with (possibly) different
        types. We assume indepenence between the features.
        Implements the naive Bayes Model.

    """

    def __init__(self, distList):
        """
        Constructor

        @param distList: list of ProbDistribution objects
        """
        # initialize attributes
        self.distList = distList
        self.p = 0
        self.freeParams = 0
        self.dataRange = []
        self.dist_nr = len(distList)

        # dimension and dataRange for sufficient statistics data
        self.suff_p = 0
        self.suff_dataRange = None
        for dist in distList:
            assert isinstance(dist, ProbDistribution)
            self.p += dist.p
            self.dataRange.append(self.p)
            self.freeParams += dist.freeParams
            self.suff_p += dist.suff_p

        # initializing dimensions for sufficient statistics data
        self.update_suff_p()

    def __eq__(self, other):
        if other.p != self.p or other.dist_nr != self.dist_nr:
            return False
        for i in range(self.dist_nr):
            if not (other.distList[i] == self.distList[i]):
                return False
        return True

    def __copy__(self):
        copyList = []
        for i in range(len(self.distList)):
            copyList.append(copy.copy(self.distList[i]))

        copy_pd = ProductDistribution(copyList)
        copy_pd.suff_p = self.suff_p
        copy_pd.suff_dataRange = copy.copy(self.suff_dataRange)
        return copy_pd

    def __str__(self):
        outstr = "ProductDist: \n"
        for dist in self.distList:
            outstr += "  " + str(dist) + "\n"
        return outstr

    def __getitem__(self, ind):
        if ind < 0 or ind > self.dist_nr - 1:
            raise IndexError
        else:
            return self.distList[ind]

    def __setitem__(self, ind, value):
        if ind < 0 or ind > self.dist_nr - 1:
            raise IndexError
        else:
            self.distList[ind] = value

    def __len__(self):
        return self.dist_nr

    def pdf(self, data):
        from core.models.mixture import MixtureModel
        assert self.suff_dataRange and self.suff_p, "Attributes for sufficient statistics not initialized."
        if isinstance(data, DataSet):
            res = np.zeros(data.N, dtype='Float64')
            for i in range(self.dist_nr):
                if isinstance(self.distList[i], MixtureModel): # XXX only necessary for mixtures of mixtures
                    res += self.distList[i].pdf(data.singleFeatureSubset(i))
                else:
                    res += self.distList[i].pdf(data.getInternalFeature(i, self.distList[i]))
            return res
        else:
            raise TypeError, 'DataSet object required, got ' + str(type(data))

    def sample(self):
        ls = []
        for i in range(len(self.distList)):
            s = self.distList[i].sample()
            if type(s) != list:
                ls.append(s)
            else:
                ls += s
        return ls

    def sampleSet(self, nr):
        res = []
        for i in range(nr):
            res.append(self.sample())
        return res

    def sampleDataSet(self, nr):
        """
        Returns a DataSet object of size 'nr'.

        @param nr: size of DataSet to be sampled

        @return: DataSet object
        """
        ls = []
        for i in range(nr):
            ls.append(self.sample())

        data = DataSet()
        data.dataMatrix = ls
        data.N = nr
        data.p = self.p
        data.sampleIDs = []

        for i in range(data.N):
            data.sampleIDs.append("sample" + str(i))

        for h in range(data.p):
            data.headers.append("X_" + str(h))

        data.internalInit(self)
        return data

    def MStep(self, posterior, data, mix_pi=None):
        from core.models.mixture import MixtureModel
        assert self.suff_dataRange and self.suff_p, "Attributes for sufficient statistics not initialized."
        assert isinstance(data, DataSet), 'DataSet required, got ' + str(type(data)) + '.'

        for i in range(self.dist_nr):
            if isinstance(self.distList[i], MixtureModel):
                self.distList[i].MStep(posterior, data.singleFeatureSubset(i), mix_pi)
            else:
                self.distList[i].MStep(posterior, data.getInternalFeature(i))

    def formatData(self, x):
        res = []
        last_index = 0
        for i in range(len(self.distList)):

            # XXX HACK: if distList[i] is an HMM feature there is nothing to be done
            # since all the HMM code was moved to mixtureHMM.py we check whether self.distList[i]
            # is an HMM by string matching  __class__ (for now).
            if self.distList[i].p == 1:
                strg = str(self.distList[i].__class__)
                if strg == 'mixtureHMM.HMM':
                    continue

            if self.dist_nr == 1:
                [new_p, dat] = self.distList[i].formatData(x)
                res += dat
            else:
                if self.distList[i].suff_p == 1:
                    [new_p, dat] = self.distList[i].formatData(x[self.dataRange[i] - 1])
                    res += dat
                else:
                    [new_p, dat] = self.distList[i].formatData(x[last_index:self.dataRange[i]])
                    res += dat
                last_index = self.dataRange[i]
        return [self.suff_p, res]

    def isValid(self, x):
        last_index = 0
        for i in range(len(self.distList)):
            if self.distList[i].p == 1:
                try:
                    self.distList[i].isValid(x[self.dataRange[i] - 1])
                except InvalidDistributionInput, ex:
                    ex.message = "\n\tin ProductDistribution.distList[" + str(i) + "]"
                    raise
            else:
                try:
                    self.distList[i].isValid(x[last_index:self.dataRange[i]])
                except InvalidDistributionInput, ex:
                    ex.message = "\n\tin ProductDistribution.distList[" + str(i) + "]"
                    raise
            last_index = self.dataRange[i]

    def flatStr(self, offset):
        offset += 1
        s = "\t" * offset + ";Prod;" + str(self.p) + "\n"
        for d in self.distList:
            s += d.flatStr(offset)
        return s

    def posteriorTraceback(self, x):
        res = []
        last_index = 0
        assert len(x) == self.suff_p, "Different number of dimensions in data and model."

        for i in range(len(self.distList)):
            if self.distList[i].suff_p == 1:
                res += self.distList[i].posteriorTraceback(x[:, self.suff_dataRange[i] - 1])
            else:
                res += self.distList[i].posteriorTraceback(x[:, last_index:self.suff_dataRange[i]])

            last_index = self.suff_dataRange[i]
        return res

    def update_suff_p(self):
        old_suff_p = None
        # in case suff_ variables have already been initialized,
        # store old values and compare as consistency check
        if self.suff_dataRange is not None and self.suff_p:
            old_suff_dataRange = self.suff_dataRange
            old_suff_p = self.suff_p

        self.suff_dataRange = []
        self.suff_p = 0
        for dist in self.distList:
            self.suff_p += dist.update_suff_p()
            self.suff_dataRange.append(self.suff_p)

        if old_suff_p:
            assert self.suff_p == old_suff_p, str(self.suff_p) + " != " + str(old_suff_p)
            assert old_suff_dataRange == self.suff_dataRange, str(old_suff_dataRange) + " != " + str(self.suff_dataRange)

        return self.suff_p
