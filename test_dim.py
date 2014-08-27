from mixture import *
from random import *

dList = []
piList = [random(), random(), random(),random(), random(), random(),random(), random(), random(),random()]
g = lambda x: x / sum(piList)
piList = map(g,piList)
#print piList
#print sum(piList)


for i in range(10):
    par = [random(), random(), random()]
    f = lambda x: x / sum(par)
    par = map(f,par)
#    print par
    
    dList.append( MultinomialDistribution(6,3,par))
    
mix = MixtureModel(10,piList,dList)    

s = mix.sample(1000)


dList2 = []
piList2 = [random(), random(), random(),random(), random(), random(),random(), random(), random(),random()]
g = lambda x: x / sum(piList2)
piList2 = map(g,piList2)
#print piList
#print sum(piList)


for i in range(10):
    par2 = [random(), random(), random()]
    f = lambda x: x / sum(par2)
    par2 = map(f,par2)
#    print par
    
    dList2.append( MultinomialDistribution(6,3,par2))
    
mix2 = MixtureModel(10,piList2,dList2)    

mix2.EM(s,50,0.2)

print "----------- True Model--------- "
print mix
print "--------------------------------\n\n"

print "----------- Trained Model--------- "
print mix2
print "--------------------------------"


