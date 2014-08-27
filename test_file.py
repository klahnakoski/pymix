import mixture

#m = MixtureModel(3,[0.25,0.15,0.6],
#                [ NormalDistribution(-6.0, 1.5),
#                  NormalDistribution(1.0, 0.5),
#                  NormalDistribution(6.0, 1.5)
#                ] )
#                
#                
##writeMixture(m,'test.txt')
##
##read_m = readMixture("test.txt")
##
##print "Reading:"
##print read_m
##x = read_m.sample()
##print x
##p = read_m.pdf(x)
##print p
#
##-----------------------------------------------------------------------------------
#
#
#mc1 = MultinomialDistribution(6,3,[0.0,0.25,0.75])
#mc2 = MultinomialDistribution(6,3,[0.5,0.3,0.2  ])
#
#mult = MixtureModel(2,[0.5,0.5], [ mc1, mc2])
#
##writeMixture(mult,'test.txt')
##
##read_mult = readMixture("test.txt")
##
##print "Reading:"
##print read_mult
##x = read_mult.sample()
##print x
##p = read_mult.pdf(x)
##print p
#
#
#
##-----------------------------------------------------------------------------------
#mult = MultinomialDistribution(6,3,[0.25,0.25,0.5])
#phi = NormalDistribution(3.0, 0.5)
#phi2 = NormalDistribution(0.5, 1.5)
#pd = ProductDistribution([mult,phi,phi2])
#
#mult2 = MultinomialDistribution(6,3,[0.1,0.7,0.2])
#phi3 = NormalDistribution(0.0, 1.5)
#phi4 = NormalDistribution(-2.5, 0.5)
#pd2 = ProductDistribution([mult2,phi3,phi4])
#
#pd_mix = MixtureModel(2,[0.5,0.5], [pd,pd2])
#
##writeMixture(pd_mix,'test.txt')
##
##read_pd_mix = readMixture("test.txt")
##print "Reading:"
##print read_pd_mix
##x = read_pd_mix.sample()
##print x
##p = read_pd_mix.pdf(x)
##print p
#
##-----------------------------------------------------------------------------------
#
#
#mmc1 = MixtureModel(2,[0.5,0.5],
#       [  MultinomialDistribution(6,3,[0.75,0.25,0.0]),
#          MultinomialDistribution(6,3,[0.4,0.3,0.3]) 
#       ])
#
#mmc2 = MixtureModel(2,[0.1,0.9],
#       [  MultinomialDistribution(6,3,[0.3,0.6,0.1]),
#          MultinomialDistribution(6,3,[0.2,0.1,0.7]) 
#       ])
#
#mm1 = MixtureModel(2,[0.2,0.8],[mmc1,mmc2],compFix=[2,2])
#
#writeMixture(mm1,'test.txt')
##
#read_mm1 = readMixture("test.txt")
##
##print "Reading:"
##print read_mm1
##print read_mm1.compFix
##x = read_mm1.sample()
##print x
##p = read_mm1.pdf(x)
##print p
#
#
##-----------------------------------------------------------------------------------
#
##
##mc1 = MixtureModel(1,[1.0], [MultinomialDistribution(6,3,[0.0,0.25,0.75])] )
##mc2 = MixtureModel(1,[1.0], [MultinomialDistribution(6,3,[0.5,0.3,0.2  ])] )
##
##mult = MixtureModel(2,[0.5,0.5], [ mc1, mc2])
##
##writeMixture(mult,'test.txt')
##
##read_mult = readMixture("test.txt")
##
##print "Reading:"
##print read_mult
##x = read_mult.sample()
##print x
##p = read_mult.pdf(x)
##print p
##
#
#
##-----------------------------------------------------------------------------------
#
##adhd = readMixture('ADHD_model.mix')
##print adhd

#-----------------------------------------------------------------------------------

G = 3
p = 4
# Bayesian Mixture with three components and four discrete features
piPrior = mixture.DirichletDistribution(G,[1.0]*G)

compPrior= []
for i in range(2):
    compPrior.append( mixture.DirichletDistribution(4,[1.02,1.02,1.02,1.02]) )
for i in range(2):
    compPrior.append( mixture.NormalGammaDistribution( 1.0,2.0,3.0,4.0 ) )

mixPrior = mixture.MixturePrior(0.7,0.7,piPrior, compPrior)

DNA = mixture.Alphabet(['A','C','G','T'])
comps = []
for i in range(G):
    dlist = []
    for j in range(2):
       phi = mixture.random_vector(4)
       dlist.append( mixture.DiscreteDistribution(4,phi,DNA))
    for j in range(2):
       mu = j+1.0
       sigma = j+0.5
       dlist.append( mixture.NormalDistribution(mu,sigma))


    comps.append(mixture.ProductDistribution(dlist))
pi = mixture.random_vector(G)

m = mixture.BayesMixtureModel(G,pi, comps, mixPrior, struct = 1)

mixture.writeMixture(m, 'test.bmix')


m2 = mixture.readMixture('test.bmix')


print m2
print m2.prior
