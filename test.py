#import ghmm
#import mixture
#import numpy
#import random
#import math
#import copy


import _C_mixextend
import numpy

a = numpy.array([1,2,3,4])
r = _C_mixextend.wrap_gsl_dirichlet_sample(a,4)
print r







#-----------------------------------------------


#import randomMixtures
#g = randomMixtures.getRandomCSIMixture_conditionalDists(3, 3, 0.0, 1000.0, M=8, dtypes='disc')
#print g

#-----------------------------------------------
#import mixture
#
#
#
#d = mixture.DirichletPrior(4, [0.2] *4)
#
#for i in range(10):
#    dist = d.sample()
#    print dist.phi   



##-------------------------------------------------
#import _C_mixextend as C
#import numpy
#
#
#M =5
#alpha = numpy.array([1.0]*M)
#
#for i in range(1):
#    p = C.wrap_gsl_dirichlet_sample(alpha,M)
#    print p,p.sum()
#    p = p * 0.8
#    print p,p.sum()
#
#
#-------------------------------------------------

#import numpy as n
#import _C_mixextend as C
#a = n.arange(12,dtype='Float32').reshape((3,4))
#a[0,1] = -1e500
#
#b = n.arange(12,dtype='Float32').reshape((3,4)) *2
#b[0,1] = - 1e500
#
#print a
#print b
#c = C.substract_matrix(b,a)
#
#print c


#-------------------------------------------------



#a = numpy.array([[ random.randint(1,100) for i in range(5)] for j in range(5)])
#print a
#t = mixture._C_mixextend.get_two_largest_elements(a)
#print '\n',t
#
#g = [ random.randint(1,100) for i in range(5)]
#l = [ random.randint(1,100) for i in range(5)]
#
#print '\n',g
#print l
#
#mixture._C_mixextend.update_two_largest_elements(t,g,l)
#
#print '\n',t


#-------------------------------------------------











#import time
#import _C_extend
#
#import pygsl.rng
#
#def vector_len(p):
#    r = 0.0
#    for i in p:
#        r += i**2
#    return math.sqrt(r)
#
#def get_angle_degrees(v,u):
#    transform =   180.0 / math.pi
#
#    dp = numpy.dot(v,u)
#
#    ct = dp / (vector_len(v) * vector_len(u))
#
#    return ct
#
#
#print get_angle_degrees( [1.0,0.0,0.0], [0.3,0.3,1.0] )
#print get_angle_degrees( [0.0,1.0,0.0], [0.6,0.6,1.0] )
#print get_angle_degrees( [0.0,0.0,1.0], [0.3,0.3,1.0] )
#
#











#-----------------------------------------------------------------------------------------

#r = numpy.rec.fromrecords([['vv', 3, 'lo'],['bb', 4, 'du'] ], names='col1,col2,col3')
#print r
#print r['col1']
#
#
#a1 = numpy.array([0.0,0.0])
#a2 = numpy.array([1.0,1.0])
#r = numpy.rec.fromrecords([[a1, 3, 'lo'],[a2, 4, 'du'] ], names='col1,col2,col3')
#print r
#print r['col1']

#a1 = numpy.array([0.0,0.0,0.0])
#a2 = numpy.array([1.0,1.0])
#a = numpy.array([ a1, a2 ])
#print a




#n1 = mixture.NormalDistribution(0.0,1.0)
#n2 = mixture.NormalDistribution(1.0,2.0)
#n3 = mixture.NormalDistribution(-3.0,1.0)
#n4 = mixture.NormalDistribution(-6.0,2.0)
#ao = numpy.array([[n1,n2],[n3,n4]])
#print type(ao)
#print ao.dtype
#print ao
#print ao[0]
#print ao[0,1].mu
#print ao[:,0]

#---------------------------------------------------------------------------------

#print '*** res=',res

#l = [ [4.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0] ]
#a = numpy.array(l, dtype='Float64')


#def get_a(N,G):
#    #N = 100
#    #G = 6
#    a = numpy.zeros((G,N))
#    for i in range(G):
#        for j in range(N):
#            a[i,j] = random.uniform(0.1,10)
#
#    a[:,0 ] = numpy.zeros(G)
#    a = numpy.log(a)
#    return a

#a = get_a(100,6)

#print a

#a = numpy.log( numpy.array([ [10.0,2.0, 3.0],
#                             [4.0,15.0,16.0],
#                             [7.0, 8.0, 9.0] ]))

#print a
#---------------------------------------------------------------------------------

rep = 6

#t0 = time.time()
#for r in range(rep):
#    res = []
#    for i in range(len(a[0])):
#        res.append(mixture.sumlogs_purepy(a[:,i]))
#    #print 'purepy:',res
#t1 = time.time()
#print 'purepy:',t1-t0, 'seconds'
#
#
##print type((a,))
#
##s = mixextend.test2(a)
##print '*** s=',s
#
#t0 = time.time()
#for r in range(rep):
#    r = mixture.matrixSumlogs(a)
#    #print 'old:',r
#t1 = time.time()
#print 'mixextend:',t1-t0, 'seconds'
#
##r = mixture.matrixSumlogs_numpy(a)
##print 'mixextend:',r
#
#t0 = time.time()
#for r in range(rep):
#    r = _C_extend.matrix_sum_logs(a)
#    #print 'C_extend:',r
#t1 = time.time()
#print 'C_extend:',t1-t0, 'seconds'


#a = get_a(100,1000)
#t0 = time.time()
#for c in a:
#    r =  mixture.sumlogs(c)
#t1 = time.time()
#print 'mixextend:',t1-t0, 'seconds'
#
#t0 = time.time()
#for c in a:
#    r = mixture.sumlogs_new(c)
#t1 = time.time()
#print 'C_extend:',t1-t0, 'seconds'

# -----------------------------------------------------------------------


#print a
#r = _C_extend.get_normalized_posterior_matrix(a)
#print r
#if r == float('-inf'):
#    raise ValueError

#G = 5
#p_disc = 0
#p_norm = 2000
#N = 1000
#delta = 0.1
#
## Bayesian Mixture with three components and four discrete features
#piPrior = mixture.DirichletPrior(G,[1.0]*G)
#
#compPrior= []
#for i in range(p_disc):
#    compPrior.append( mixture.DirichletPrior(4,[1.02,1.02,1.02,1.02]) )
#for i in range(p_norm):
#    compPrior.append(  mixture.NormalGammaPrior(1.5, 0.1, 3.0, 1.0) )
#
#mixPrior = mixture.MixtureModelPrior(0.004,0.004,piPrior, compPrior)
#mixPrior.structPriorHeuristic(delta, N)
#
#DNA = mixture.Alphabet(['A','C','G','T'])
##random.seed(3586662)
#comps = []
#for i in range(G):
#    dlist = []
#    for j in range(p_disc):
#       phi = mixture.random_vector(4)
#       dlist.append( mixture.DiscreteDistribution(4,phi, DNA))
#
#    for j in range(p_norm):
#       mu = random.uniform(-3.0, 3.0)
#       sigma = random.uniform(0.1,2.0)
#       dlist.append( mixture.NormalDistribution(mu,sigma))
#
#
#    comps.append(mixture.ProductDistribution(dlist))

#print comps
#print [comps[i].distList[0] for i in range(len(comps))]

#pi = mixture.random_vector(G)
#m = mixture.BayesMixtureModel(G,pi, comps, mixPrior, struct = 1)



#t0 = time.time()
#r2 =mixPrior.pdf_old(m)
#t1 = time.time()
#print 'old:',t1-t0, 'seconds'
#
#t0 = time.time()
#r1=  mixPrior.pdf(m)
#t1 = time.time()
#print 'new:',t1-t0, 'seconds'

#print r1
#print r2




#ng = mixture.NormalGammaPrior(1.5, 0.1, 3.0, 1.0)
#n = mixture.NormalDistribution(100.0, 0.8)
#
#t0 = time.time()
#for i in range(1):
#    r = ng.pdf(n)
#    print r
#t1 = time.time()
#print 'time:',t1-t0, 'seconds'


#data = m.sampleDataSet(N)

#t0 = time.time()
#ll, lp = m.EStep_old(data)
#t1 = time.time()
##print 'old:'
##print ll
##print lp
#print 'mixextend:',t1-t0, 'seconds'
#
#t0 = time.time()
#ll2, lp2 = m.EStep(data)
#t1 = time.time()
##print 'new:'
##print ll
##print lp
#print 'C_extend: ',t1-t0, 'seconds'
#
#print lp
#assert str(lp) == str(lp2), str(lp)+', '+str(lp2)

#---------------------------------------------------------------------------------




#
#n1 = mixture.ProductDistribution([mixture.NormalDistribution(-2.5,0.5)])
#n2 = mixture.ProductDistribution([mixture.NormalDistribution(6.0,0.8)])
#
#pi = [0.4, 0.6]
#gen = mixture.MixtureModel(2,pi,[n1,n2])
#
#random.seed(3586662)        
#data = gen.sampleDataSet(10)
#
#print data.internalData
#
#f0 = data.getInternalFeature(0)
#
#
#print f0
#print data._internalData_views[0]











#---------------------------------------------------------------------------------

# gamma pdf in gsl
#def my_inv_gamma(sigma, dof, scale ):
#    return (sigma**2) ** (-(dof+2) / 2 ) * math.exp(- scale / (2 * sigma**2) )
#
#def gsl_gamma(sigma, a, b):
#    return pygsl.rng.gamma_pdf(1.0/sigma, a, 1.0/b)
#
#
#res1 = my_inv_gamma(1.0, 3.0, 1.0 )
#res2 = gsl_gamma(1.0, 3.0, 1.0 )
#print res1/res2
#
#res1 = my_inv_gamma(1.6, 3.0, 1.0 )
#res2 = gsl_gamma(1.6, 3.0, 1.0 )
#print res1/res2
#
#
#res1 = my_inv_gamma(2.0, 3.0, 1.0 )
#res2 = gsl_gamma(2.0, 3.0, 1.0 )
#print res1/res2





#---------------------------------------------------------------------------------

#l = [1,1,1,1] * 50
#a = numarray.array([1,1,1,1] * 50)
#
#rep = 5000
#
#timing.start()
#for i in range(rep):
#    sum(l)
#timing.finish()
#print  'sum(list)', float(timing.micro()) / 1000000
#
#timing.start()
#for i in range(rep):
#    sum(a)
#timing.finish()
#print  'sum(arr)', float(timing.micro()) / 1000000
#
#
#timing.start()
#for i in range(rep):
#    numarray.sum(l)
#timing.finish()
#print 'na.sum(list)', float(timing.micro()) / 1000000
#
#timing.start()
#for i in range(rep):
#    numarray.sum(a)
#timing.finish()
#print  'na.sum(arr)', float(timing.micro()) / 1000000


#G = 4
#p = 1
## Bayesian Mixture with three components and four discrete features
#piPrior = mixture.DirichletPrior(G,[1.0]*G)
#
#compPrior= []
#for i in range(p):
#    compPrior.append( mixture.DirichletPrior(4,[1.02,1.02,1.02,1.02]) )
#
#mixPrior = mixture.MixtureModelPrior(0.004,0.004,piPrior, compPrior)
#
#DNA = mixture.Alphabet(['A','C','G','T'])
##random.seed(3586662)
#comps = []
#for i in range(G):
#    dlist = []
#    for j in range(p):
#       phi = mixture.random_vector(4)
#       dlist.append( mixture.DiscreteDistribution(4,phi, DNA))
#    comps.append(mixture.ProductDistribution(dlist))
#pi = mixture.random_vector(G)
#model = mixture.BayesMixtureModel(G,pi, comps, mixPrior, struct = 1)
#
#data = model.sampleDataSet(30)
#
#model.modelInitialization(data)
#
#model.mapEM( data, 40,0.1,silent=1)
#
#
#print model
#
#
#
## building data likelihood factor matrix for the current group structure      
#l = numarray.zeros( (model.dist_nr, model.G, data.N),dtype='Float64' )
#for j in range(model.dist_nr):
#    for lead_j in model.leaders[j]:
#        l_row = model.components[lead_j][j].pdf(data.getInternalFeature(j) )  
#        l[j,lead_j,:] = l_row
#        for v in model.groups[j][lead_j]:
#            l[j,v,:] = l_row
#
## g is the matrix of log posterior probabilities of the components given the data
#g = numarray.sum(l) 
#for k in range(model.G):
#    g[k,:] += numarray.log(model.pi[k])
#
#sum_logs = numarray.zeros(data.N,dtype='Float64')    
#g_norm = numarray.zeros((model.G, data.N),dtype='Float64')
#for n in range(data.N):
#    sum_logs[n] = mixture.sumlogs(g[:,n])
#    # normalizing log posterior
#    g_norm[:,n] = g[:,n] - sum_logs[n]
#
#tau = numarray.exp(g_norm)
#
#print "tau:"
#for tt in tau:
#    print tt.tolist()
#print
#
#lk = sum(sum_logs) 
#
#print '\nlk = ',lk
#
#
#
#print "\ng:"
#for tt in g:
#    print tt.tolist()
#print
#
#tau_pool = tau[0,:] + tau[1,:]
#pi_pool = model.pi[0] + model.pi[1]
#
#print 'tau_pool =',tau_pool
#
#data_j = data.getInternalFeature(0)
#candidate_dist = copy.copy(model.components[0][0])
#model.prior.compPrior[j].mapMStep(candidate_dist, tau_pool, data_j,  pi_pool)  
#
#print '  ->',candidate_dist
#
#l_row = candidate_dist.pdf(data_j)   
#
#print 'l_row =',l_row
#
#g_1 = copy.copy(g)
#
#g_1[0, :] = l_row
#g_1[1, :] = l_row
#
#print "\ng_1:"
#for tt in g_1:
#    print tt.tolist()
#print
#
#sum_logs = numarray.zeros(data.N,dtype='Float64')    
#for n in range(data.N):
#    sum_logs[n] = mixture.sumlogs(g_1[:,n])
#lk_1 = sum(sum_logs)
#
#print 'new lk=',lk_1
#
##print '\n',g[0:2,:].tolist()
##print '\n',g[2:4,:].tolist()
#
#print '\n\n ----- TEST -----'
#
#
#old_part_g = g[0:2,:]
#old_sum_logs = numarray.zeros(data.N,dtype='Float64')    
#for n in range(data.N):
#    old_sum_logs[n] = mixture.sumlogs(old_part_g[:,n])
#old_lk_term = sum(old_sum_logs)
#
#new_part_g = g_1[0:2,:]
#new_sum_logs = numarray.zeros(data.N,dtype='Float64')    
#for n in range(data.N):
#    new_sum_logs[n] = mixture.sumlogs(new_part_g[:,n])
#new_lk_term = sum(new_sum_logs)
#
#test_lk = lk - old_lk_term + new_lk_term
#
#print 'test_lk =',test_lk
#
#lk_test2 = 0.0
#for j in range(data.N):
#    l_ij = numarray.exp( g[:,j] )
#    #print   l_ij 
#
#    sum_l_ij = sum(l_ij)
#    lk_test2 += numarray.log( sum_l_ij  )
#
#
#print 'lk_test2 =',lk_test2












#d = mixture.DiscreteDistribution(4,[0.1,0.2,0.3,0.4])
#p = mixture.ProductDistribution([d])
#
#m = mixture.MixtureModel(1,[1.0],[p])
#
#l = [['0'],['1'],['2'],['3']]
#
#data = mixture.DataSet()
#data.fromList(l)
#
#print data
#print m
#
#data.internalInit(m)






#mult = mixture.MultinomialDistribution(6,3,[0.25,0.25,0.5])
#phi = mixture.NormalDistribution(2.0, 0.5)
#phi2 = mixture.NormalDistribution(0.5, 1.5)
#pd = mixture.ProductDistribution([mult,phi,phi2])
#
#
#
#for d in pd:
#    print d
#    


#def my_log(arr):
#    res = numarray.zeros(len(arr),dtype='Float64')
#    for i,x in enumerate(arr):
#        if x == 0.0:
#            res[i] = float('-inf')
#        else:    
#            res[i] = math.log(x)
#
#    return res        
#
#def getData(N,nr_0):
#
#
#    dat = numarray.zeros(N,dtype='Float64')
#    for i in range(N):
#        dat[i] = random.random()
#
#
#    ind = numarray.array(random.sample(range(N), nr_0))
#    dat[ind] = 0.0
#
#    return dat
#
#N= 10
#nr_0 = 4
#
#
#dat = getData(N,nr_0)
#timing.start()
#res = my_log(dat)
#timing.finish()
#print "time", float(timing.micro()) / 1000000
#
#
#dat = getData(N,nr_0)
#timing.start()
#ind1 = numarray.where(dat == 0.0)[0]       
#dat[ind1] = float('inf')  
## computing log likelihood 
#res = numarray.log(dat)  # XXX use mylog function
#res[ind1] = float('-inf')  
#
#timing.finish()
#print "time", float(timing.micro()) / 1000000
#
#
#dat = getData(N,nr_0)
#timing.start()
#print dat
#
#ind1 = numarray.where(dat == 0.0)[0]       
#print "ind1",ind1
#ind2 = numarray.where(dat != 0.0)[0]       
##dat[ind1] = float('inf')
#print "ind2",ind2
## computing log likelihood 
#dat[ind2] = numarray.log(dat[ind2]) 
#
#print dat
#
## replace inf values with -inf to complete calculations
##ind2 = numarray.where(dat == float('inf'))[0]
#dat[ind1] = float('-inf')  
#
#timing.finish()
#print "time", float(timing.micro()) / 1000000





#e = mixture.ExponentialDistribution(2.0)
#a = numarray.array([1.0],dtype='Float64')
#print e.pdf(a)


#A  = [[0.3, 0.6,0.1],[0.0, 0.5, 0.5],[0.4,0.2,0.4]]
#B  = [[0.5, 0.5],[0.5,0.5],[1.0,0.0]]
#pi = [1.0, 0.0, 0.0]
#I = ghmm.IntegerRange(0,2)
#D = ghmm.DiscreteDistribution(I)
#m = ghmm.HMMFromMatrices(I,D,A,B,pi)
#
##print m
#
#seq = m.sample(10,50)
#m.baumWelch(seq,10,0.1)
#
#timing.start()
#s = str(m)
#timing.finish()
#print "time", float(timing.micro()) / 1000000
#timing.start()
#s2 =  m.str2()
#timing.finish()
#print "time", float(timing.micro()) / 1000000

#
#p = [0.4,0.4,0.2]
#alpha = [1.0,1.0,3.0]
#d = mixture.DirichletDistribution(alpha)ss
#
#print "My pdf:"
#print d.pdf(p)
#
#
#print "PyGsl:"
#print pygsl.rng.dirichlet_pdf(alpha,[p] )
#

#curr_K = 3
#weights = [0.1,0.1,0.1,0.7]

# sample component
#r = random.random()
#p_sum = 0
#for k in range(curr_K+1):
#    p_sum += weights[k]
#    if r <= p_sum:               
#        break
#print "        chosen comp",k

#----------------------------------------------------------------------


#import crp
#
#scale = 4.0
#shape = 4.0
#
#
##  def __init__(self, shape, scale, mu, tau  ):
#p = crp.NormalGammaPrior(shape,scale,0.0,2.0)
#
#nr = 80000
#ps = 0.0
#pl = numpy.zeros(nr,dtype='Float64')
#
#inv_pl = numpy.zeros(nr,dtype='Float64')
#inv_ps = 0.0
#
#for i in range(nr):
#    grand = p.sampleG()
#    pl[i] = grand
#    ps += grand
#    
#    inv_grand = 1.0 / grand
#    inv_pl[i] = inv_grand 
#    inv_ps += inv_grand 
#
#
#print "shape =", shape
#print "scale =", scale,"\n"
#
#print "Gamma:"   
#print "mean = ",ps / nr, " -> theroetical: ", shape,"*", scale, "= ", shape * scale
#print "var =", mixture.variance(pl)," -> theroetical:", shape,"*", scale**2   , "= ", shape*scale**2
#
#inv_shape = shape
#inv_scale = 1.0 / scale
#print "\ninv_shape =", inv_shape
#print "inv_scale =", inv_scale
#
#print "\nInverse Gamma:"   
#print "mean = ",inv_ps / nr," -> theroetical:", inv_scale,"/ (", inv_shape,"- 1) = ",inv_scale / (inv_shape-1)  
#print "var =", mixture.variance(inv_pl), " -> theroetical:", inv_scale,"**2 / ( (",inv_shape," - 1)**2 (",inv_shape,"- 2) ) = ", inv_scale**2 / ((inv_shape-1)**2 * (inv_shape-2))
#
