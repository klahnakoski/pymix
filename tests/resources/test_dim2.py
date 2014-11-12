from random import random
from pyLibrary.times.timer import Timer
from pymix.distributions.multinomial import MultinomialDistribution
from pymix.distributions.product import ProductDistribution
from pymix.models.mixture import MixtureModel

pdList = []
for j in range(3):
    dList = []
    for i in range(10):
        par = [random(), random(), random(), random(), random(), random()]
        f = lambda x: x / sum(par)
        par = map(f, par)

        dList.append(MultinomialDistribution(6, 6, par))

    pdList.append(ProductDistribution(dList))

piList = [random(), random(), random()]
g = lambda x: x / sum(piList)
piList = map(g, piList)

mix = MixtureModel(3, piList, pdList)

dat = mix.sampleDataSet(1000)

pdList2 = []
for j in range(3):
    dList2 = []
    for i in range(10):
        par2 = [random(), random(), random(), random(), random(), random()]
        f = lambda x: x / sum(par2)
        par2 = map(f, par2)

        dList2.append(MultinomialDistribution(6, 6, par2))

    pdList2.append(ProductDistribution(dList2))

piList2 = [random(), random(), random()]
g = lambda x: x / sum(piList2)
piList2 = map(g, piList2)

mix2 = MixtureModel(3, piList2, pdList2)
with Timer("time"):
    mix2.EM(dat, 40, 0.1)


