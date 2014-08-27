import random
from mixture import *

#d = DataSet(["test.txt","drd1.txt","pheno.txt"])
d = DataSet(["test.txt","drd1.txt","pheno.txt"])
print d



p1 = []
p2 = []
p3 = []
p4 = []
for i in range(25):
    p1.append(random.random())
    p2.append(random.random())
    p3.append(random.random())
    p4.append(random.random())
    
g1 = lambda x: x/sum(p1)
p1 = map(g1,p1)

g2 = lambda x: x/sum(p2)
p2 = map(g2,p2)

g3 = lambda x: x/sum(p3)
p3 = map(g3,p3)

g4 = lambda x: x/sum(p4)
p4 = map(g4,p4)


mult = MultinomialDistribution(6,25,p1,SNP)
mult2 = MultinomialDistribution(7,25,p2,SNP)
phi = normalDistribution(11.0, 4.0)
phi2 = normalDistribution(11.0, 6.0)
pd1 = ProductDistribution([mult,mult2,phi,phi2])

mult3 = MultinomialDistribution(6,25,p3,SNP)
mult4 = MultinomialDistribution(7,25,p4,SNP)
phi3 = normalDistribution(8.0, 5.0)
phi4 = normalDistribution(15.0, 5.0)
pd2 = ProductDistribution([mult,mult2,phi,phi2])


m = MixtureModel(2,[0.5,0.5], [ pd1, pd2])
m.EM(d,15,0.05)
