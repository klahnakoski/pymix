from mixture import *
from time import clock  
from copy import copy


p1 = []
for i in range(25):
    p1.append(random.random())
   
g1 = lambda x: x/sum(p1)
p1 = map(g1,p1)

mult = MultinomialDistribution(5,25,p1,SNP)
#r = mult.applyAlphabet(['A/A','C/T','G/G','C/C','A/A'])

mult2 = MultinomialDistribution(5,4,[0.25]*4,DNA)
phi = NormalDistribution(11.0, 4.0)
phi2 = NormalDistribution(11.0, 6.0)


r = phi.applyAlphabet([11.23,12.23,9.7,12.423])

pd1 = ProductDistribution([mult,mult2,phi,phi2])

r2 = pd1.applyAlphabet(['A/A','C/T','G/G','C/C','A/A','a','c','g','t','t',11.0,12.0])




# ----------------------------- Example 1 ----------------------------- 
m = MixtureModel(3,[0.25,0.5,0.25],
                [ NormalDistribution(-3.0, 0.5),
                  NormalDistribution(1.0, 0.5),
                  NormalDistribution(6.0, 0.5)
                ] )


print "true",m

seq = m.sampleSet(100)

m2 = MixtureModel(3,[0.2,0.4,0.4],
                [ NormalDistribution(-3.5, 0.5),
                  NormalDistribution(0.5, 1.5),
                  NormalDistribution(4.0, 0.6)
                ] )

m2.randParams(seq)
t1 = clock()
m2.EM(seq,40,0.0)
t2 = clock()
print "time = ", t2-t1
print m2

#  ----------------------------- Example 2 ----------------------------- 
e1 = MixtureModel(2,[0.7,0.3],
                    [ NormalDistribution(0.0,0.4), ExponentialDistribution(0.5) ]
                    )

seq2 = e1.sample(500)

e2 = MixtureModel(2,[0.5,0.5],
                    [ NormalDistribution(2.0,0.4), ExponentialDistribution(0.1) ]
                    )

#e2.EM(seq2,60,5)


#  ----------------------------- Example 3 ----------------------------- 
m3 = MixtureModel(2,[0.3,0.7],
                [ NormalDistribution(0.0, 0.5),
                  NormalDistribution(1.3, 0.5)
                ] )

(true,seq3) = m3.sampleSetLabels(380)

m4 = MixtureModel(2,[0.5,0.5],
                [ NormalDistribution(-1.5, 1.5),
                  NormalDistribution(1.5, 1.5)
                ] )


dat = DataSet()
dat.fromArray(seq3)

print "vorher ------\n", m4
pred = m4.cluster(dat, nr_runs=5,nr_init=9, max_iter=30, delta=0.1, labels = None, entropy_cutoff = None)

classes = m4.classify(dat)


m4.shortInitEM(dat, 5, 5,5, 0.1)
m4.EM(seq3,20,0.1)
print "####Finish\n",m4

dat.printClustering(2,pred)

evaluate(pred, true)


#  ----------------------------- Example 4 ----------------------------- 
m5 = MixtureModel(1,[1.0],
                [ NormalDistribution(3.0, 2.5) ] )

seq4 = m5.sample(1800)

#print "var = ", variance(seq4)

m6 = MixtureModel(1,[1.0],
                [ NormalDistribution(-1.5, 2.5) ] )

#m6.EM(seq4,1,5)
#print m6

#seq5 = numarray.zeros(900,numarray.Float)
#for i in range(900):
#    seq5[i] = random.normalvariate(0.0,0.5)
    
#print "var = ", variance(seq5)

# -----------------------------  Example 5 ----------------------------- 


mc1 = MixtureModel(1,[1.0], [MultinomialDistribution(6,3,[0.0,0.25,0.75])] )
mc2 = MixtureModel(1,[1.0], [MultinomialDistribution(6,3,[0.5,0.3,0.2  ])] )

m7 = MixtureModel(2,[0.5,0.5], [ mc1, mc2])

seq6 = m7.sampleSet(150)

mc3 = MixtureModel(1,[1.0], [MultinomialDistribution(6,3,[0.4,0.5,0.1] )])
mc4 = MixtureModel(1,[1.0], [MultinomialDistribution(6,3,[0.2,0.1,0.7])] )

m8 = MixtureModel(2,[0.1,0.9], [  mc3,mc4   ])
m8.EM(seq6,30,0.3)
#print m8


# -----------------------------  Example 6 ----------------------------- 

mult = MultinomialDistribution(6,3,[0.25,0.25,0.5])
phi = NormalDistribution(2.0, 0.5)
phi2 = NormalDistribution(0.5, 1.5)
pd = ProductDistribution([mult,phi,phi2])

mult2 = MultinomialDistribution(6,3,[0.1,0.7,0.2])
phi3 = NormalDistribution(0.0, 1.5)
phi4 = NormalDistribution(-0.0, 0.5)
pd2 = ProductDistribution([mult2,phi3,phi4])

pd_mix = MixtureModel(2,[0.5,0.5], [pd,pd2])


#[classes, pd_seq] = pd_mix.labelled_sample(1000)

pd_seq = pd_mix.sampleSet(100)

mult3 = MultinomialDistribution(6,3,[0.6,0.1,0.3])
phi5 = NormalDistribution(1.0, 1.5)
phi6 = NormalDistribution(-1.0, 1.0)
pd3 = ProductDistribution([mult3,phi5,phi6])

mult4 = MultinomialDistribution(6,3,[0.4,0.3,0.3])
phi7 = NormalDistribution(-2.0, 0.2)
phi8 = NormalDistribution(1.5, 3.5)
pd4 = ProductDistribution([mult4,phi7,phi8])

pd_mix2 = MixtureModel(2,[0.2,0.8], [pd3,pd4])
pd_mix2.randParams(pd_seq)


cluster = pd_mix2.cluster(pd_seq,50,0.2)

evaluate(cluster, classes)
print "----------- True Model--------- "
print pd_mix
print "--------------------------------\n\n"
print "----------- Trained Model--------- "
print pd_mix2
print "--------------------------------" 
pd_mix2.EM(pd_seq,30,0.0)
print pd_mix
print pd_mix2


# -----------------------------  Example 5 ----------------------------- 

mult1 = MultinomialDistribution(3,4,[0.25,0.25,0.25,0.25])
mix1= MixtureModel(2,[0.3,0.7],[NormalDistribution(-2.0,1.0),NormalDistribution(-2.0,1.0)],compFix=[0,2])
mix2= MixtureModel(2,[0.8,0.2],[NormalDistribution(-3.0,1.0),NormalDistribution(3.0,1.0)],compFix=[0,2])
pd1 = ProductDistribution([mix1,mix2,mult1])

mult2 = MultinomialDistribution(3,4,[0.2,0.1,0.5,0.2])
mix3= MixtureModel(2,[0.5,0.5],[NormalDistribution(1.0,1.0),NormalDistribution(0.5,1.0)],compFix=[0,2])
mix4= MixtureModel(2,[0.1,0.9],[NormalDistribution(2.0,1.0),NormalDistribution(1.5,1.0)],compFix=[0,2])
pd2 = ProductDistribution([mix3,mix4,mult2])

m7 = MixtureModel(2,[0.4,0.6], [ pd1,pd2])

seq6 = m7.sampleSet(80)

p=random_vector(4)
mult3 = MultinomialDistribution(3,4,p)
mix5= MixtureModel(2,[0.3,0.7],[NormalDistribution(-1.0,1.0),NormalDistribution(-2.0,1.0)],compFix=[0,2])
mix6= MixtureModel(2,[0.8,0.2],[NormalDistribution(-2.0,1.0),NormalDistribution(2.0,1.0)],compFix=[0,2])
pd3 = ProductDistribution([mix5,mix6,mult3])

p=random_vector(4)
mult4 = MultinomialDistribution(3,4,p)
mix7= MixtureModel(2,[0.5,0.5],[NormalDistribution(-1.0,1.0),NormalDistribution(-3.0,1.0)],compFix=[0,2])
mix8= MixtureModel(2,[0.1,0.9],[NormalDistribution(2.5,1.0),NormalDistribution(0.5,1.0)],compFix=[0,2])
pd4 = ProductDistribution([mix3,mix4,mult4])


m8= MixtureModel(2,[0.5,0.5], [ pd3,pd4])
m8.randParams(seq6)

m8.EM(seq6,15,0.3)
#print m8

#print m8


SNP = Alphabet(["A/A","A/C","A/G","A/T",
        "C/A","C/C","C/G","C/T",
        "G/A","G/C","G/G","G/T",
        "T/A","T/C","T/G","T/T",
        "T/N","G/N","C/N", "A/N", "N/A",
        "N/C","N/G","N/T","N/N"])


DIAG = Alphabet(['.','0','8','1'])

m = MultinomialDistribution(6,4,[0.25,0.25,0.25,0.25],DIAG)
m_1 = MultinomialDistribution(1,4,[0.25,0.25,0.25,0.25],DIAG)
n = NormalDistribution(2.5,1.0)
pd = ProductDistribution([m_1,n,m])

m2 = MultinomialDistribution(6,4,[0.2,0.1,0.5,0.2],DIAG)
m_2 = MultinomialDistribution(1,4,[0.25,0.25,0.25,0.25],DIAG)
n2 = NormalDistribution(1.0,0.5)
pd2 = ProductDistribution([m_2,n2,m2])

mix = MixtureModel(2,[0.2,0.8], [ pd,pd2 ])
seq = mix.sampleSet(500)

#print seq[0]
##suff = mix.sufficientStatistics(seq[0])
#suff = mix.sufficientStatistics(seq[1])
#print suff

p3 = random_vector(4)
m3 = MultinomialDistribution(6,4,[0.3,0.3,0.1,0.3],DIAG)
m_3 = MultinomialDistribution(1,4,[0.25,0.25,0.25,0.25],DIAG)
n3 = NormalDistribution(1.5,1.0)
pd3 = ProductDistribution([m_3,n3,m3])

p4= random_vector(4)
m4 = MultinomialDistribution(6,4,[0.2,0.1,0.3,0.4],DIAG)
n4= NormalDistribution(0.0,0.5)
pd4 = ProductDistribution([m_3,n4,m4])

mix2 = MixtureModel(2,[0.25,0.75], [ pd3,pd4 ])

data = DataSet()
data.dataMatrix= seq
#data.fromFiles(["filt_dat1.txt"])
data.N = 500
mix2.EM(data,30,0.1)

print mix
print mix2













