from mixture import *
from random import *

N = 150

p = []
p2 = []
p5 = []
for i in range(N):
    p.append(random())
    p2.append(random())
    p5.append(random())

g = lambda x: x/sum(p)
p = map(g,p)

g2 = lambda x: x/sum(p2)
p2 = map(g2,p2)

g5 = lambda x: x/sum(p5)
p5 = map(g5,p5)


multi = MultinomialDistribution(80,N,p)
multi2 = MultinomialDistribution(80,N,p2)
multi5 = MultinomialDistribution(80,N,p5)

mix = MixtureModel(3,[0.5,0.25,0.25],[multi,multi2,multi5])
print mix

[true,s] = mix.labelled_sample(1000)

p3 = []
p4 = []
p6 = []
for i in range(N):
    p3.append(random())
    p4.append(random())
    p6.append(random())

g3 = lambda x: x/sum(p3)
p3 = map(g3,p3)

g4 = lambda x: x/sum(p4)
p4 = map(g4,p4)

g6 = lambda x: x/sum(p6)
p6 = map(g6,p6)


multi3 = MultinomialDistribution(80,N,p3)
multi4 = MultinomialDistribution(80,N,p4)
multi6 = MultinomialDistribution(80,N,p6)

mix2 = MixtureModel(3,[0.1,0.3,0.6],[multi3,multi4,multi6])

pred = mix2.cluster(s,40,0.2)

evaluate(pred, true)

