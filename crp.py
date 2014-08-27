import mixture
import random
import copy
import math
import pygsl.sf
import numpy
import pygsl.rng
import sys

class NormalGammaPrior:

    def __init__(self, shape, scale, mu, tau  ):

        #print "shape ", shape/2.0
        #print "scale ", scale/2.0
        #print "mu ", mu
        #print "tau ", tau
        
        
        # parameters on inverse gamma prior on variance
        self.shape = shape    # shape parameter of the inverse Gamma  (s)
        self.scale = scale    # scale parameter of the inverse Gamma  (S)
        
        # parameters on normal prior on the mean
        self.mu = mu
        self.tau = tau
        
        
    def sample(self,returnType='tuple'):
    
        assert returnType in ['tuple','object']
        
        grand = random.gammavariate(self.shape, self.scale )
        
        #return grand
        
        sigma = 1.0 / grand
        
        #print sigma
        mu = random.normalvariate(self.mu, math.sqrt(self.tau*sigma) )
        #mu = random.normalvariate(self.mu, self.tau*sigma )
        
        if returnType == 'tuple':
            return (mu,sigma)
        elif returnType == 'object':
            return mixture.NormalDistribution(mu,math.sqrt(sigma))                        

    def sampleG(self):
        grand = random.gammavariate(self.shape, self.scale )
        return grand



class ChineseRestaurantProcess:

    def __init__(self, alpha, G0):
        self.alpha = alpha      #   proportional to the probability of opening a new table
        self.G0 = G0    # base measure over the parameters of the component distributions, conjugate prior
                        # e.g NormalGammaPrior

    
    def sample(self, nr):
        nr_comp = 0
        comp_memb = []
        comp_list = []
        
        for i in range(0,nr,1):
            print "----------- iteration "+str(i)+" -----------"
            
            print "nr_comp = ", nr_comp
            # calculation of the component assignment distribution
            crp_dist = copy.copy(comp_memb)
            crp_dist.append(self.alpha)
            for j in range(nr_comp+1):
                crp_dist[j] = float(crp_dist[j]) / float((i-1+self.alpha) )
            
          
            print "crp_dist ", crp_dist
            
            # sample new component assignment
            r = random.random()
            p_sum = 0
            for j in range(nr_comp+1):
                
                p_sum += crp_dist[j]
                if r <= p_sum:
                    #print "   assignment to ", j
                    
                    # new component has been openend
                    if (j+1) > nr_comp:
                        print "  ->  New table opened."
                        
                        comp_list.append(self.G0.sample())
                        
                        comp_memb.append(1)
                        nr_comp += 1
                    else:
                        comp_memb[j]  += 1   

                    break

            print "comp_memb = ", comp_memb

        print "\nNumber of components: ",nr_comp
        return [comp_list, comp_memb]


class MultivariateChineseRestaurantProcess:
    
    def __init__(self, alpha_list, G0_list):
        self.alpha_list = alpha_list      #   proportional to the probability of opening a new table
        self.G0_list = G0_list    # base measure over the parameters of the component distributions, conjugate prior
                                  # e.g NormalGammaPrior






class GibbsSampler:
    def __init__(self, CRP):

        self.CRP = CRP   # ChineseRestaurantProcess object
        #self.compList = []   #  XXX list of component parameters tuples, to be sampled from base measure
                             # in runSampler XXX

        self.dist_dict = {}  # dictionary of distinct Gaussians in the process
        self.data_assignment_dict = {}    #  dictionary of number of data points associated 
                                          #  with a distributions in self.dist_dict
        
        self.dist_index_list = []   # list of keys into self.dist_dict which give assignment of data points to distributions

    def runSampler(self, burn_in ,nr_steps, data):
        """
        Implementation of "Bayesian density estimation and Inference using Mixtures", Escobar & West, 1995
        
        """
        
        
        # XXX data is a list of observations, should be DataSet object later
        
        data_nr = len(data)
        print "data_nr = ",data_nr
        
        # XXX initialise data item component assignment with draws from the CPR        
        #self.compList, comp_memb = self.CRP.sample(data_nr)
        #curr_K = len(comp_memb)
        
        # XXX initialisation by sampling a component for each data item
        
        self.dist_index_list = numpy.zeros(data_nr,dtype='Int32')
        
        dist_id = 0
        for d in range(data_nr):
            self.dist_dict[dist_id] =  self.CRP.G0.sample() 
            self.data_assignment_dict[dist_id] = 1  # one data point per distribution
            self.dist_index_list[d] = dist_id
            dist_id += 1
        
        curr_K = data_nr
        
        #print "Initial component assignment: ", comp_memb
                
        
        #print "Initial parameters: "
        #print self.dist_dict
        #raise RuntimeError
        #for p in self.compList:
        #    print p
        
        print "Starting sampling for "+ str(nr_steps)+ " iterations. (Burn in is "+str(burn_in)+" steps)"


        # XXX
        # first compute factor c_shape
        #                                    s                                                s                                                   s  
        c_shape_log = pygsl.sf.lngamma(  (1 + self.CRP.G0.shape ) / 2.0  )[0] - pygsl.sf.lngamma( self.CRP.G0.shape / 2.0 )[0] + math.log(1.0/math.sqrt(self.CRP.G0.shape)) 
        c_shape = math.exp(c_shape_log)

        # compute M
        #               tau                  S                    s  
        M = (1 + self.CRP.G0.tau) * ( self.CRP.G0.scale / self.CRP.G0.shape)

        
        
        # Gibbs sampling main loop
        for s in range(burn_in + nr_steps):

            #print 'step',s

            if s == burn_in:
                sys.stdout.write("\nBurn in complete.\n")
                sys.stdout.flush()

            
            #print "-------------- step "+str(s)+" --------------------------"
            if (s % 100) == 0:
                sys.stdout.write(".")
                sys.stdout.flush()


            # in each step sample new values for each component conditioned on all other components
            for i in range(data_nr):
                
                
#                print '---------------------------------------'
#                print 'keys:'
#                print self.dist_dict.keys(), len(self.dist_dict.keys())
#                print self.data_assignment_dict.keys(), len(self.dist_dict.keys())
#                
#                print
#                for kk in self.dist_dict.keys():
#                    print kk,self.dist_dict[kk],self.data_assignment_dict[kk]
#                
#                print self.dist_index_list
#                print 'nr components:',curr_K
#                print '\n'
                
                
                
                
                # compute weigths
                weights = []
                weight_keys = []   # keys into self.dist_dict for weights

                for j in self.dist_dict.keys():
                    # XXX ??
                    
                    if self.dist_index_list[i] == j:
                        n_j = self.data_assignment_dict[j] -1
                    else:   
                        n_j = self.data_assignment_dict[j]    
                    
                    
                    
                    # compute the weights for the current components
                    #      q_j          n_j                       y_i             mu_j                        sigma_j                                      sigma_j
                    # XXX comp_memb[j] * XXX

                    w = n_j  * math.exp( - (data[i][0] - self.dist_dict[j][0])**2 / (2 *  self.dist_dict[j][1]))  * ( (2 * self.dist_dict[j][1] )**-0.5  )

                    weights.append( w )
                    weight_keys.append( j )

                    #print "w(",j,") =", w
                    #res = ( 1.0 / (math.sqrt(self.compList[j][1]) * math.sqrt(2 * math.pi)) ) * math.exp( (- (data[i]-self.compList[j][0])**2) /( 2*self.compList[j][1]) )
                    #print res
                    #print pygsl.rng.gaussian_pdf(data[i]- self.compList[j][0], self.compList[j][1])


                # compute weight for the uninitialized components:
                   
#                # first compute factor c_shape
#                #                                    s                                                s                                                   s  
#                c_shape_log = pygsl.sf.lngamma(  (1 + self.CRP.G0.shape ) / 2.0  )[0] - pygsl.sf.lngamma( self.CRP.G0.shape / 2.0 )[0] + math.log(1.0/math.sqrt(self.CRP.G0.shape)) 
#                c_shape = math.exp(c_shape_log)
#
#                print c_shape
#
#                # compute M
#                #               tau                  S                    s  
#                M = (1 + self.CRP.G0.tau) * ( self.CRP.G0.scale / self.CRP.G0.shape)
#
#                print M

                #                       alpha                          y_i           m      
                weights.append( (self.CRP.alpha * c_shape * ( 1 + ((data[i][0] - self.CRP.G0.mu)**2 / (self.CRP.G0.shape * M)) )**-((1+self.CRP.G0.shape)/2)) * (1.0 / math.sqrt(M)))
                
                        
                #print "q"+str(i), weights                    
                
                # normalize weight vector:
                w_sum = sum(weights)
                f = lambda x: x/w_sum    
                weights = map(f,weights)
                
#                print '***'
#                for iii,ww in enumerate(weights):
#                    if iii < len(weights)-1:
#                        print '  ', iii,ww, self.dist_dict[weight_keys[iii]]
                
                
                # sample component
                r = random.random()
                p_sum = 0
                for k in range(len(weights)):
                    p_sum += weights[k]
                    if r <= p_sum:               
                        break
                
                #print "        chosen comp",k, "curr_K ", curr_K
                
                #print "\n   data point[",i,"] =",data[i],"  -> ", self.compList[i]
                #print "    q"+str(i), weights  
                #print "        chosen comp",k, "curr_K ", curr_K
                
                # previously initialised component was chosen and we assign the corresponding parameters
                if k < curr_K-1:
                    new_index = weight_keys[k]
                    old_index = self.dist_index_list[i]
                    
                    #self.compList[i] = self.compList[k]
                    
                    #            self.data_assignment_dict[dist_id] = 1  # one data point per distribution
                    #            self.dist_index_list[d] = dist_id
                    #            dist_id += 1
                    
                    # reassign distribution
                    self.dist_index_list[i] = new_index
                    self.data_assignment_dict[new_index] += 1  #  update data point assignment count
                    
                    if self.data_assignment_dict[old_index] == 1:
                        # remove empty component
                        self.dist_dict.pop(old_index)
                        self.data_assignment_dict.pop(old_index)
                        curr_K -= 1
                    else:
                        self.data_assignment_dict[old_index] -= 1
                    
                    
                    #print i,"  -> assigned ",old_index,' to ',new_index
                    
                    # adjust index for changes in comp_memb
                    #if k >= i:
                    #    k += 1
                    
                    # XXX update component membership vector  XXX
                    #comp_memb[k] += 1
                    #comp_memb[i] -= 1
               
                    #if comp_memb[i] == 0:
                    #    empty = 1

               
                else:
                    # for new component sample from prior with updated parameters
                    #   def __init__(self, shape, scale, mu, tau  ):
                    shape_i = (1.0 + self.CRP.G0.shape) / 2.0
                    scale_i = ((self.CRP.G0.scale) + (( data[i][0] - self.CRP.G0.mu )**2.0  / ( 1 + self.CRP.G0.tau ))) / 2.0

                    mu_i = ( self.CRP.G0.mu + self.CRP.G0.tau * data[i][0] ) / ( 1.0 + self.CRP.G0.tau )
                    tau_i =  self.CRP.G0.tau / ( 1 + self.CRP.G0.tau )
                    prior_i = NormalGammaPrior( shape_i, scale_i, mu_i, tau_i )
                    
                    
#                    sys.stdout.write("   \n-----------")
#                    sys.stdout.write("  data["+str(i)+"] = "+str( data[i])+ ", current comp= "+str(self.dist_dict[self.dist_index_list[i]])+"\n")
#                    sys.stdout.write("  New sigma sampled from G("+str(shape_i)+", "+str(scale_i)+")\n")
#                    print '      Expectation mean: ',shape_i * scale_i,' expectation var:',shape_i*scale_i**2
#                    sys.stdout.write("  New mu sampled from N("+str(mu_i)+","+str(tau_i)+"*V_i)\n")
                    
                    new_p = prior_i.sample() 
                    
#                    print '     -> ',new_p


                    #sys.stdout.flush()
                    dist_id += 1  # increment distribution id 


                    new_index = dist_id
                    old_index = self.dist_index_list[i]

                    #print "New component sampled: "+str(new_p)+'(',old_index,'->',new_index,'), id=' ,new_index


                    self.dist_index_list[i] = new_index
                    
                    # update component parameters
                    self.dist_dict[new_index] = new_p 
                    self.data_assignment_dict[new_index] = 1

                    if self.data_assignment_dict[old_index] == 1:
                        # remove empty component
                        self.dist_dict.pop(old_index)
                        self.data_assignment_dict.pop(old_index)
                        curr_K -= 1
                        
                    else:
                        self.data_assignment_dict[old_index] -= 1
                    
                    
                    curr_K += 1
                    
                    #comp_memb.append(1)
       
                #print "comp_memb = ",comp_memb
                #for c in self.compList:
                #    print c
                #print "-----\n"
                
                
                # XXX debug checks
                assert len(self.dist_dict.keys()) == curr_K, str(len(self.dist_dict.keys()))+' , '+  str(curr_K)
                assert self.dist_dict.keys() == self.data_assignment_dict.keys()
                for kk in self.dist_dict.keys():
                    assert self.data_assignment_dict[kk] == len(numpy.where(self.dist_index_list == kk)[0]), str(self.data_assignment_dict[kk])+' , '+str(len(numpy.where(self.dist_index_list == kk)[0]))
                assert sum( [ self.data_assignment_dict[kk] for kk in self.data_assignment_dict.keys() ] ) == data_nr, str(sum( [ self.data_assignment_dict[kk] for kk in self.data_assignment_dict.keys() ]))+' != '+ str(data_nr)


            
            if s > burn_in:
       
                # post processing
                print '\n------------ step '+str(s - burn_in+1)+ '---------------'
                for e in self.dist_dict.keys():
                    print self.dist_dict[e]," (",self.data_assignment_dict[e],")"


class GibbsSamplerVariant2:
    def __init__(self, CRP):

        self.CRP = CRP   # ChineseRestaurantProcess object
        #self.compList = []   #  XXX list of component parameters tuples, to be sampled from base measure
                             # in runSampler XXX

        self.dist_dict = {}  # dictionary of distinct Gaussians in the process
        self.data_assignment_dict = {}    #  dictionary of number of data points associated 
                                          #  with a distributions in self.dist_dict
        
        self.dist_index_list = None   # list of keys into self.dist_dict which give assignment of data points to distributions



    def runSampler(self, burn_in ,nr_steps, data):
        """
        Implementation of "Hierarchical priors and mixture models with applications 
        in regression and density estimation", Escobar, Mueller & West, 1994
        """
        
        # XXX data is a list of observations, should be DataSet object later
        
        data_nr = len(data)
        self.dist_index_list = numpy.zeros(data_nr)
        
        print "data_nr = ",data_nr
        
        # XXX initialise data item component assignment with draws from the CPR        
        #self.compList, comp_memb = self.CRP.sample(data_nr)
        #curr_K = len(comp_memb)
        
        # XXX initialisation by sampling a component for each data item
        
        self.dist_index_list = numpy.zeros(data_nr,dtype='Int32')
        
        dist_id = 0
        for d in range(data_nr):
            self.dist_dict[dist_id] =  self.CRP.G0.sample(returnType = 'object') 
            self.data_assignment_dict[dist_id] = 1  # one data point per distribution
            self.dist_index_list[d] = dist_id
            dist_id += 1
        
        curr_K = data_nr
        
        #print "Initial component assignment: ", comp_memb
                
        
        #print "Initial parameters: "
        #print self.dist_dict
        #raise RuntimeError
        #for p in self.compList:
        #    print p
        
        print "Starting sampling for "+ str(nr_steps)+ " iterations. (Burn in is "+str(burn_in)+" steps)"


        # XXX
        # first compute factor c_shape
        c_shape_log = pygsl.sf.lngamma(  (1 + self.CRP.G0.shape ) / 2.0  )[0] - pygsl.sf.lngamma( self.CRP.G0.shape / 2.0 )[0] + math.log(1.0/math.sqrt(self.CRP.G0.shape)) 
        c_shape = math.exp(c_shape_log)

        # compute M
        #               tau                  S                    s  
        M = (1 + self.CRP.G0.tau) * ( self.CRP.G0.scale / self.CRP.G0.shape)

        
        
        # Gibbs sampling main loop
        for s in range(burn_in + nr_steps):

            #print 'step',s

            if s == burn_in:
                sys.stdout.write("\nBurn in complete.\n")
                sys.stdout.flush()

            
            #print "-------------- step "+str(s)+" --------------------------"
            if (s % 100) == 0:
                sys.stdout.write(".")
                sys.stdout.flush()


            # in each step sample new values for each component conditioned on all other components
            for i in range(data_nr):
                
                
#                print '---------------------------------------'
#                print 'keys:'
#                print self.dist_dict.keys(), len(self.dist_dict.keys())
#                print self.data_assignment_dict.keys(), len(self.dist_dict.keys())
#                
#                print
#                for kk in self.dist_dict.keys():
#                    print kk,self.dist_dict[kk],self.data_assignment_dict[kk]
#                
#                print self.dist_index_list
#                print 'nr components:',curr_K
#                print '\n'
                
                prev_dist_index = self.dist_index_list[i]
                
                
                # compute weigths
                weights = []
                weight_keys = []   # keys into self.dist_dict for weights

                for j in self.dist_dict.keys():
#                    if self.dist_index_list[i] == j:
#                        n_j = self.data_assignment_dict[j] -1
#                    else:   
#                        n_j = self.data_assignment_dict[j]    
                    
                    n_j = self.data_assignment_dict[j]                        
                    
                    # compute the weights for the current components
                    #      q_j          n_j                       y_i             mu_j                        sigma_j                                      sigma_j
                    # XXX comp_memb[j] * XXX

                    w = n_j  * math.exp( self.dist_dict[j].pdf(data[i]) )

                    #print n_j, self.dist_dict[j].pdf(data[i]),math.exp( self.dist_dict[j].pdf(data[i]) )

                    weights.append( w )
                    weight_keys.append( j )

                    #print "w(",j,") =", w
                    #res = ( 1.0 / (math.sqrt(self.compList[j][1]) * math.sqrt(2 * math.pi)) ) * math.exp( (- (data[i]-self.compList[j][0])**2) /( 2*self.compList[j][1]) )
                    #print res
                    #print pygsl.rng.gaussian_pdf(data[i]- self.compList[j][0], self.compList[j][1])


                # compute weight for the uninitialized components:
                   
#                # first compute factor c_shape
#                #                                    s                                                s                                                   s  
#                c_shape_log = pygsl.sf.lngamma(  (1 + self.CRP.G0.shape ) / 2.0  )[0] - pygsl.sf.lngamma( self.CRP.G0.shape / 2.0 )[0] + math.log(1.0/math.sqrt(self.CRP.G0.shape)) 
#                c_shape = math.exp(c_shape_log)
#
#                print c_shape
#
#                # compute M
#                #               tau                  S                    s  
#                M = (1 + self.CRP.G0.tau) * ( self.CRP.G0.scale / self.CRP.G0.shape)
#
#                print M

                #                       alpha                          y_i           m      
                weights.append( (self.CRP.alpha * c_shape * ( 1 + ((data[i][0] - self.CRP.G0.mu)**2 / (self.CRP.G0.shape * M)) )**-((1+self.CRP.G0.shape)/2)) * (1.0 / math.sqrt(M)))
                
                        
                #print "q"+str(i), weights                    

                # normalize weight vector:
                w_sum = sum(weights)
                f = lambda x: x/w_sum    
                weights = map(f,weights)

#                print '***'
#                for iii,ww in enumerate(weights):
#                    if iii < len(weights)-1:
#                        print '  ', iii,ww, self.dist_dict[weight_keys[iii]]
                
                #print weights

                # sample component
                r = random.random()
                p_sum = 0
                for k in range(len(weights)):
                    p_sum += weights[k]
                    if r <= p_sum:               
                        break
                
                # sample new distribution if necessary and initialize data structures
                if k == len(weights)-1:
                    self.dist_dict[dist_id] =  self.CRP.G0.sample(returnType = 'object') 
                    weight_keys.append(dist_id)
                    self.data_assignment_dict[dist_id] = 0  
                    dist_id += 1



                self.dist_index_list[i] = weight_keys[k]
                self.data_assignment_dict[prev_dist_index] -= 1 
                self.data_assignment_dict[weight_keys[k]] += 1    
                
                #print i, prev_dist_index, weight_keys[k]
                
                # remove distribution without assigned samples
                if self.data_assignment_dict[prev_dist_index] == 0:
                    self.data_assignment_dict.pop(prev_dist_index)
                    self.dist_dict.pop(prev_dist_index)
                
           
            
            for key in self.dist_dict.keys():
                # compute sufficient statistics
                key_ind = numpy.where(self.dist_index_list == key)[0]  # find sample indices assigned to distribution 'key'
                nr_dat = len(key_ind)
                T1 = data[key_ind].sum()
                #tt = data[key_ind]**2
                #T2 = tt.sum()
                
                mean = T1 / nr_dat
                SSE = numpy.sum((data[key_ind] - mean)**2)
                
                # update distribution parameters by sampling from posterior
                #   def __init__(self, shape, scale, mu, tau  ):
                shape_i = (nr_dat + self.CRP.G0.shape)  / 2.0
                scale_i = ((self.CRP.G0.scale) + (0.5 * SSE ) + ( nr_dat* self.CRP.G0.tau * ( mean - self.CRP.G0.mu )**2.0 ) / ( 2 * (self.CRP.G0.tau + nr_dat ))) /2.0
                
                mu_i = ( self.CRP.G0.tau * self.CRP.G0.mu + nr_dat * mean ) / ( nr_dat + self.CRP.G0.tau )
                tau_i =  self.CRP.G0.tau + nr_dat
                prior_i = NormalGammaPrior( shape_i, scale_i, mu_i, tau_i )



                update_dist = prior_i.sample(returnType = 'object') 


#                sys.stdout.write("   \n-----------")
#                sys.stdout.write("  data = "+str( data[key_ind]))
#                print '  sufficent. stat:',T1,T2,mean, SSE
#                sys.stdout.write("  New sigma sampled from G("+str(shape_i)+", "+str(scale_i)+")\n")
#                print '      Expectation mean: ',1.0/(shape_i * scale_i),' expectation var:',shape_i*scale_i**2
#                sys.stdout.write("  New mu sampled from N("+str(mu_i)+","+str(tau_i)+"*V_i)\n")
#                print '    ->', update_dist
                
                self.dist_dict[key] = update_dist
                
    
    
#            print self.dist_index_list
#            print self.data_assignment_dict
#            print self.dist_dict

        
                # XXX debug checks
                #assert len(self.dist_dict.keys()) == curr_K, str(len(self.dist_dict.keys()))+' , '+  str(curr_K)
                assert self.dist_dict.keys() == self.data_assignment_dict.keys()
                for kk in self.dist_dict.keys():
                    assert self.data_assignment_dict[kk] == len(numpy.where(self.dist_index_list == kk)[0]), str(self.data_assignment_dict[kk])+' , '+str(len(numpy.where(self.dist_index_list == kk)[0]))
                assert sum( [ self.data_assignment_dict[kk] for kk in self.data_assignment_dict.keys() ] ) == data_nr, str(sum( [ self.data_assignment_dict[kk] for kk in self.data_assignment_dict.keys() ]))+' != '+ str(data_nr)


            if s > burn_in:
       
                # post processing
                print '\n------------ step '+str(s - burn_in+1)+'  ( dists: '+str(len(self.dist_dict)) +') ---------------'
                for e in self.dist_dict.keys():
                    print self.dist_dict[e]," (",self.data_assignment_dict[e],")"



class MultivariateGibbsSampler:
    def __init__(self, CRP_list):
        self.CRP_list = CRP_list
        self.compList = []   #  XXX list of component parameters tuples XXX


    def runSampler(self, burn_in ,nr_steps, data):
        
        # XXX data is a list of observations, should be DataSet object later
        
        # data dimensionality and number CRPs must agree
        assert len(data[0]) == len(CRP_list)

        data_nr = len(data)
        print "data_nr = ",data_nr

        # XXX initialisation by sampling a component for each data item
        for d in range(data_nr):
            self.compList.append( self.CRP.G0.sample() )
        comp_memb = [1] * data_nr
        curr_K = data_nr
        


























