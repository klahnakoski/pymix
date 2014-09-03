from pymix.distributions.normal import NormalDistribution
from pymix.distributions.product import ProductDistribution
from pymix.models.mixture import MixtureModel
from pymix.util.dataset import DataSet

pr1 = ProductDistribution([NormalDistribution(-6.0, 0.5), NormalDistribution(-4.0, 0.5), NormalDistribution(-3.0, 0.5)])
pr2 = ProductDistribution([NormalDistribution(-5.0, 0.5), NormalDistribution(-3.3, 0.5), NormalDistribution(-2.3, 0.5)])

m = MixtureModel(2, [0.7, 0.3], [pr1, pr2])

seq = m.sampleSet(5)

#print seq
z = 0
m.printTraceback(DataSet().fromList(seq), z)
