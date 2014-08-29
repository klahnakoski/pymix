from core.distributions.product import ProductDistribution
from core.pymix_util.errors import InvalidDistributionInput
from core.models.mixture import MixtureModel
from core.priors.prior import PriorDistribution
from core.pymix_util.dataset import DataSet


class ProductDistributionPrior(PriorDistribution):
    """
    Prior for ProductDistribution objects. Basically only holds a list of priors for
    atomar distributions. Necessary for model hierarchy.
    """

    def __init__(self, priorList):
        """
        Constructor

        @param priorList: list of PriorDistribution objects
        """
        self.priorList = priorList
        self.dist_nr = len(priorList)

    def __getitem__(self, ind):
        if ind < 0 or ind > self.dist_nr - 1:
            raise IndexError, 'Index ' + str(ind)
        else:
            return self.priorList[ind]

    def __setitem__(self, ind, item):
        assert isinstance(item, PriorDistribution)

        if ind < 0 or ind > self.dist_nr - 1:
            raise IndexError
        else:
            self.priorList[ind] = item

    def __eq__(self, other):
        if not isinstance(other, ProductDistributionPrior):
            return False
        if self.dist_nr != other.dist_nr:
            return False
        for i in range(self.dist_nr):
            if not self.priorList[i] == other.priorList[i]:
                return False
        return True

    def pdf(self, dist):
        assert isinstance(dist, ProductDistribution)
        res = 0
        for i in range(self.dist_nr):
            res += self.priorList[i].pdf(dist.distList[i])

        return res

    def marginal(self, dist, posterior, data):
        assert isinstance(dist, ProductDistribution)
        res = 0
        for i in range(self.dist_nr):
            res += self.priorList[i].marginal(dist.distList[i], posterior, data.getInternalFeature(i))

        return res

    def mapMStep(self, dist, posterior, data, mix_pi=None, dist_ind=None):
        assert dist.suff_dataRange and dist.suff_p, "Attributes for sufficient statistics not initialized."
        assert isinstance(data, DataSet)
        assert isinstance(dist, ProductDistribution)
        assert dist.dist_nr == len(self.priorList)

        for i in range(dist.dist_nr):
            if isinstance(dist.distList[i], MixtureModel):
            # XXX use of isinstance() should be removed, singleFeatureSubset(i) to replace getInternalFeature(i)  ?
                self.priorList[i].mapMStep(dist.distList[i], posterior, data.singleFeatureSubset(i), mix_pi, dist_ind)
            else:
                self.priorList[i].mapMStep(dist.distList[i], posterior, data.getInternalFeature(i), mix_pi, dist_ind)


    def isValid(self, p):
        if not isinstance(p, ProductDistribution):
            raise InvalidDistributionInput, 'Not a ProductDistribution.'
        if p.dist_nr != self.dist_nr:
            raise InvalidDistributionInput, 'Different dimensions in ProductDistributionPrior and ProductDistribution: ' + str(p.dist_nr) + ' != ' + str(self.dist_nr)
        for j in range(p.dist_nr):
            try:
                self[j].isValid(p[j])
            except InvalidDistributionInput, ex:
                ex.message += "\n\tin ProductDistributionPrior.priorList[" + str(j) + "]"
                raise

