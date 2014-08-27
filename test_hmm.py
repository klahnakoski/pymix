import mixture
import numpy
import random
import mixtureHMM

# building generating models
DIAG = mixture.Alphabet(['.','0','8','1'])

A  = [[0.3, 0.6,0.1],[0.0, 0.5, 0.5],[0.4,0.2,0.4]]
B  = [[0.5, 0.2,0.1,0.2],[0.5,0.4,0.05,0.05],[0.8,0.1,0.05,0.05]]
pi = [1.0, 0.0, 0.0]
h1 = mixtureHMM.getHMM(mixtureHMM.ghmm.IntegerRange(0,4), mixtureHMM.ghmm.DiscreteDistribution(mixtureHMM.ghmm.IntegerRange(0,4)), A, B, pi)

#seq = h1.hmm.sample(10,50)
#print seq

A2  = [[0.5, 0.4,0.1],[0.3, 0.2, 0.5],[0.3,0.2,0.5]]
B2  = [[0.1, 0.1,0.4,0.4],[0.1,0.1,0.4,0.5],[0.2,0.2,0.3,0.3]]
pi2 = [0.6, 0.4, 0.0]
h2 = mixtureHMM.getHMM(mixtureHMM.ghmm.IntegerRange(0,4), mixtureHMM.ghmm.DiscreteDistribution(mixtureHMM.ghmm.IntegerRange(0,4)), A2, B2, pi2)

n1 = mixture.NormalDistribution(2.5,0.5)
n2 = mixture.NormalDistribution(6.0,0.8)

mult1 = mixture.MultinomialDistribution(3,4,[0.23,0.26,0.26,0.25],alphabet = DIAG)
mult2 = mixture.MultinomialDistribution(3,4,[0.7,0.1,0.1,0.1],alphabet = DIAG)

c1 = mixture.ProductDistribution([n1,mult1,h1])
c2 = mixture.ProductDistribution([n2,mult2,h2])

mpi = [0.4, 0.6]
m = mixture.MixtureModel(2,mpi,[c1,c2])

#print m
#print "-->",m.components[0].suff_dataRange


# ----------- constructing complex DataSet ----------------

# mixture for sampling
gc1 = mixture.ProductDistribution([n1,mult1])
gc2 = mixture.ProductDistribution([n2,mult2])
gen = mixture.MixtureModel(2,mpi,[gc1,gc2])

dat = gen.sampleSet(100)
#print dat

# sampling hmm data
seq1 = h1.hmm.sample(40,10)
seq2 = h2.hmm.sample(60,10)

seq1.merge(seq2)



data = mixtureHMM.SequenceDataSet()


#data.fromGHMM(dat,[seq1])
data.fromGHMM(dat,[seq1])

data.internalInit(m)

#print data.getInternalFeature(0)
#print data.getInternalFeature(1)




# -------- construct model to be trained -------------

tA  = [[0.5, 0.2,0.3],[0.2, 0.3, 0.5],[0.1,0.5,0.4]]
tB  = [[0.2, 0.4,0.1,0.3],[0.5,0.1,0.2,0.2],[0.4,0.3,0.15,0.15]]
tpi = [0.3, 0.3, 0.4]
th1 = mixtureHMM.getHMM(mixtureHMM.ghmm.IntegerRange(0,4), mixtureHMM.ghmm.DiscreteDistribution(mixtureHMM.ghmm.IntegerRange(0,4)), tA, tB, tpi)


seq = h1.hmm.sample(10,50)
#print seq

tA2  = [[0.5, 0.4,0.1],[0.3, 0.2, 0.5],[0.3,0.2,0.5]]
tB2  = [[0.1, 0.1,0.4,0.4],[0.1,0.1,0.4,0.4],[0.2,0.1,0.6,0.1]]
tpi2 = [0.3, 0.4, 0.3]
th2 = mixtureHMM.getHMM(mixtureHMM.ghmm.IntegerRange(0,4), mixtureHMM.ghmm.DiscreteDistribution(mixtureHMM.ghmm.IntegerRange(0,4)), tA2, tB2, tpi2)

tn1 = mixture.NormalDistribution(-1.5,1.5)
tn2 = mixture.NormalDistribution(9.0,1.2)

tmult1 = mixture.MultinomialDistribution(3,4,[0.1,0.1,0.55,0.25],alphabet = DIAG)
tmult2 = mixture.MultinomialDistribution(3,4,[0.4,0.3,0.1,0.2],alphabet = DIAG)

tc1 = mixture.ProductDistribution([tn1,tmult1,th1])
tc2 = mixture.ProductDistribution([tn2,tmult2,th2])

tmpi = [0.7, 0.3]
tm = mixture.MixtureModel(2,tmpi,[tc1,tc2])

tm.EM(data,80,0.1,silent=0)

##for i,s in enumerate(seq1):
##    print "\n",i,":",s
##    print "h1:",h1.hmm.loglikelihoods(s)
##    print "h2:",h2.hmm.loglikelihoods(s)
#
#
##print tm
##print tm.components[0].distList[2].hmm
##print tm.components[1].distList[2].hmm












