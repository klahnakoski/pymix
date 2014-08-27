from mixture import *
from random import *

N = 50
M = 40
G = 8

pi1 = []
mList1 = []
for j in range(G):
    p = []
    for i in range(N):
        p.append(random())

    g = lambda x: x/sum(p)
    p = map(g,p)
    
    pi1.append(random())
    mList1.append( MultinomialDistribution(M,N,p))

fpi =lambda x: x/sum(pi1)
pi1= map(fpi,pi1)

mix = MixtureModel(G,pi1,mList1)

[true,s] = mix.labelled_sample(75)

pi2 = []
mList2 = []
for j in range(G):
    p2 = []
    for i in range(N):
        p2.append(random())

    g2 = lambda x: x/sum(p2)
    p2 = map(g2,p2)
 
    pi2.append(random())
    mList2.append( MultinomialDistribution(M,N,p2))

fpi2 =lambda x: x/sum(pi2)
pi2= map(fpi2,pi2)
mix2 = MixtureModel(G,pi2,mList2)


pred = mix2.cluster(s,40,0.2,entropy_cutoff =1e-3)
#pred = mix2.cluster(s,40,0.2)
print "pred = ",pred
print "true = ",true

evaluate(pred, true)

