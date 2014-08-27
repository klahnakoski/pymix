################################################################################
# 
#       This file is part of the Python Mixture Package
#
#       file:   fullEnumerationExhaustive.py 
#       author: Benjamin Georgi 
#  
#       Copyright (C) 2004-2009 Benjamin Georgi
#       Copyright (C) 2004-2009 Max-Planck-Institut fuer Molekulare Genetik,
#                               Berlin
#
#       Contact: georgi@molgen.mpg.de
#
#       This library is free software; you can redistribute it and/or
#       modify it under the terms of the GNU Library General Public
#       License as published by the Free Software Foundation; either
#       version 2 of the License, or (at your option) any later version.
#
#       This library is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#       Library General Public License for more details.
#
#       You should have received a copy of the GNU Library General Public
#       License along with this library; if not, write to the Free
#       Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
#
#
################################################################################import mixture

"""

This file provides functions for the CSI structure learning with complete enumeration of all
possible structures.

Since the number of possible structures is exponential both in the number of features and the number of
components, this is only feasible for quite small data sets.

"""

import numpy
import copy
import _C_mixextend
import setPartitions



def updateStructureBayesianFullEnumeration(model,data,objFunction='MAP',silent=1):
    """
    CSI structure learning with full enumeration of the structure space.
    
    @param model: BayesMixtureModel object
    @param data: DataSet object
    @param objFunction: objective function of the optimization, only 'MAP' so far
    @param silent: verbosity flag, default is True
    """
    
    P = setPartitions.generate_all_partitions(model.G,order='reverse') # XXX too slow for large G

    max_ind = len(P)-1

    curr_indices = [0] * model.dist_nr
    curr_indices[0] = -1
    
    lpos = model.dist_nr-1

    nr = 1
    term = 0
    prev_indices = [-1] * model.dist_nr

    # initial structure is full CSI matrix
    best_structure = [[ (i,) for i in range(model.G)]] * model.dist_nr   #  [ tuple(range(model.G)) ] * model.dist_nr  



    # building data likelihood factor matrix for the current group structure      
    l = numpy.zeros( (model.dist_nr, model.G, data.N),dtype='Float64' )
    for j in range(model.dist_nr):
        for lead_j in range(model.G):
            l_row = model.components[lead_j][j].pdf(data.getInternalFeature(j) )  
            l[j,lead_j,:] = l_row

    # g is the matrix of log posterior probabilities of the components given the data
    g = numpy.sum(l, axis=0) 
    for k in range(model.G):
        g[k,:] += numpy.log(model.pi[k])


    sum_logs = mixture.matrixSumlogs(g)
    g_norm = g - sum_logs
    tau = numpy.exp(g_norm)

    if not silent:
        print "\ntau="
        for tt in tau:
            print tt.tolist()
        print


    # computing posterior as model selection criterion
    temp = mixture.DiscreteDistribution(model.G,model.pi)
    pi_prior = model.prior.piPrior.pdf(temp)
    log_prior = pi_prior
    log_prior_list = [0.0] * model.dist_nr
    for j in range(model.dist_nr):
        for r in range(model.G):
               log_prior_list[j] += model.prior.compPrior[j].pdf( model.components[r][j] )

#    log_prior += sum(log_prior_list)
#
#    # prior over number of components
#    log_prior += model.prior.nrCompPrior * model.G 
#    # prior over number of distinct groups
#    for j in range(model.dist_nr):
#        log_prior += model.prior.structPrior * len(model.leaders[j])   
#
#    # get posterior
#    lk = numpy.sum(sum_logs) 
#    best_post = lk + log_prior
#    if not silent:
#        print best_structure,':'
#        print "0: ",  lk ,"+", log_prior,"=", best_post
#        #print log_prior_list
    
    best_post = float('-inf')

    # initialize merge histories
    L = [{} for j in range(model.dist_nr)]
    for j in range(model.dist_nr):

        # extracting current feature from the DataSet
        if isinstance(model.components[0][j], mixture.MixtureModel): # XXX
            data_j = data.singleFeatureSubset(j)
        else:
            data_j = data.getInternalFeature(j)

    
        for lead in range(model.G):  # every component is a leader for initialization
           
            el_dist = copy.copy(model.components[lead][j])
            tau_pool = copy.copy(tau[lead, :])
            pi_pool = model.pi[lead]

            if objFunction == 'MAP':
                model.prior.compPrior[j].mapMStep(el_dist, tau_pool, data_j,  pi_pool)  
            else:
                # should never get here...
                raise TypeError


            stat = el_dist.sufficientStatistics(tau_pool, data_j)

            M = mixture.CandidateGroup(el_dist, numpy.sum(tau_pool), pi_pool, stat)
            
            l_row = el_dist.pdf(data_j)  
            M.l = l_row
            M.dist_prior = model.prior.compPrior[j].pdf( el_dist ) 

            L[j][(lead,)] = M

    g_wo_j = numpy.zeros((model.G, data.N),dtype='Float64')
    best_indices = copy.copy(curr_indices)
    while 1:

        if not silent:
            print '\n----------------------------------'

        curr_indices[0] += 1
        
        if curr_indices[0] > max_ind:
            
            #curr_indices[lpos] = 0
            for e in range(model.dist_nr):
                if e == model.dist_nr-1:
                    if curr_indices[e] > max_ind:
                        term = 1
                        break
                    
                if curr_indices[e] > max_ind:
                    curr_indices[e] = 0
                    curr_indices[e+1] += 1

        if term:
            break



        #print '\nprev:',prev_indices
        if not silent:
            print nr, ':',curr_indices  ,'->', [P[jj] for jj in curr_indices]

        g_wo_prev = copy.copy(g)
        g_this_struct = numpy.zeros((model.G, data.N))
        for j in range(model.dist_nr):
            if prev_indices[j] == curr_indices[j]:
                #print '   -> unchanged',j,curr_indices[j], P[curr_indices[j]]
                break
            else:
                #print '\n--------\nChanged',j,curr_indices[j], P[curr_indices[j]]
                curr_struct_j = P[curr_indices[j]]
                
                # unnormalized posterior matrix without the contribution of the jth feature
                try:
                    g_wo_prev  = g_wo_prev - l[j]
                except FloatingPointError:
                    # if there was an exception we have to compute each
                    # entry in g_wo_j seperately to set -inf - -inf = -inf
                    g_wo_prev = _C_mixextend.substract_matrix(g_wo_prev,l[j])
                
                # extracting current feature from the DataSet
                if isinstance(model.components[0][j], mixture.MixtureModel): # XXX
                    data_j = data.singleFeatureSubset(j)
                else:
                    data_j = data.getInternalFeature(j)


                l_j_1 = numpy.zeros( (model.G, data.N ) )  # XXX needs only be done once                                    
                
                
                #print '\n\n***', curr_struct_j
                
                for cs_j in curr_struct_j:
                    
                    #print '    ->',cs_j
                    
                    if L[j].has_key(cs_j):
                        #print '  ** REcomp',cs_j

                        # retrieve merge data from history
                        candidate_dist = L[j][cs_j].dist                            

                        if not silent:
                            print j,"  R  candidate:", cs_j, candidate_dist

                        l_row = L[j][cs_j].l
                        #cdist_prior = L[j][cs_j].dist_prior

                    else:
                        #print '  ** comp',cs_j
                        
                        M = model.prior.compPrior[j].mapMStepMerge( [L[j][(c,)]  for c in cs_j]  )



                        #print '\n   *** compute:',hist_ind1,hist_ind2

                        candidate_dist = M.dist

                        if not silent:
                            print j,"  C  candidate:", cs_j,candidate_dist

                        l_row = candidate_dist.pdf(data_j)  

                        #print '   l_row=',l_row

                        #cdist_prior = 

                        M.l = l_row
                        M.dist_prior = model.prior.compPrior[j].pdf( candidate_dist ) 

                        L[j][cs_j] = M
                        
                    for c in cs_j:
                        l_j_1[c,:] = l_row
                        #print '            ->',c
                        #g_this_struct[c,:] += l_row
                        
                g_this_struct += l_j_1
                l[j] = l_j_1


                # compute parameter prior for the candidate merge parameters
                log_prior_list_j = 0.0
                for r in curr_struct_j:
                    log_prior_list_j += L[j][r].dist_prior * len(r)
                log_prior_list[j] = log_prior_list_j


        # get updated unnormalized posterior matrix
        g_1 = g_wo_prev + g_this_struct

#                print '\ng_wo_j:'
#                for gg in g_wo_j:
#                    print gg.tolist()
#
#                print '\nl_j_1:'
#                for gg in l_j_1:
#                    print gg.tolist()
#
#
#        print '\ng_1:'
#        for gg in g_1:
#            print gg.tolist()


        sum_logs = mixture.matrixSumlogs(g_1)
        lk_1 = numpy.sum(sum_logs)

        #print '\n  *** likelihood =', lk_1

        # computing posterior as model selection criterion
        log_prior_1 = pi_prior

            #print r, L[r].dist_prior * len(r),L[r].dist_prior


        #print '\nlog_prior_list_j =',log_prior_list_j
        #print 'log_prior_list',log_prior_list

        log_prior_1 += sum(log_prior_list)

        #print '2:',log_prior_1


        # prior over number of components
        log_prior_1 += model.prior.nrCompPrior * model.G 
        # prior over number of distinct groups
        for z in range(model.dist_nr):
            log_prior_1 += model.prior.structPrior * len(P[curr_indices[z]]) 

        #print '3:',log_prior_1

        post_1 = lk_1 + log_prior_1

        if not silent:
            print '\nPosterior:',post_1 ,'=', lk_1 ,'+', log_prior_1            
            
        if post_1 >= best_post: # current candidate structure is better than previous best
            if not silent:
                print "*** New best candidate", post_1 ,">=", best_post
                
            if post_1 == best_post:
                print '******* Identical maxima !'
                print 'current:',curr_indices
                print 'best:',best_indices
            
                
            best_indices = copy.copy(curr_indices)
            best_post = post_1
        
        
        nr += 1
        g = g_1  # set likelihood matrix for the next candidate structure
        
        # XXX DEBUG XXX
        #if nr > 500:
        #    term = 1
        
        prev_indices = copy.copy(curr_indices)
        

    # setting updated structure in model
    for j in range(model.dist_nr):
        lead = []
        groups = {}

        #print j,best_indices[j]

        best_partition = P[ best_indices[j] ]
        for gr in best_partition:
            gr_list = list(gr)
            gr_lead = gr_list.pop(0)
            lead.append(gr_lead)
            groups[gr_lead] = gr_list
            
            # assigning distributions according to new structure
            model.components[gr_lead][j] = L[j][gr].dist
            for d in gr_list:
                model.components[d][j] = model.components[gr_lead][j]



        model.leaders[j] = lead
        model.groups[j] = groups



#    print '** G=',model.G
#    print '** p=',model.dist_nr
#    print '** nr =',nr

    if not silent:
        print '\n*** Globally optimal structure out of',nr,'possible:'
        print  [P[ best_indices[j]] for j in range(model.dist_nr)]









#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

def updateStructureBayesianFullEnumeration_AIC_BIC(model,data,objFunction='MAP',penalty='POST',silent=1):
    """
    """
    
    assert penalty in ['AIC', 'BIC','POST']
    
    P = mixture.generate_all_partitions(model.G,order='reverse') # XXX too slow for large G

    max_ind = len(P)-1

    curr_indices = [0] * model.dist_nr
    curr_indices[0] = -1
    
    lpos = model.dist_nr-1

    nr = 1
    term = 0
    prev_indices = [-1] * model.dist_nr

    # initial structure is full CSI matrix
    best_structure = [[ (i,) for i in range(model.G)]] * model.dist_nr   #  [ tuple(range(model.G)) ] * model.dist_nr  



    # building data likelihood factor matrix for the current group structure      
    l = numpy.zeros( (model.dist_nr, model.G, data.N),dtype='Float64' )
    for j in range(model.dist_nr):
        for lead_j in range(model.G):
            l_row = model.components[lead_j][j].pdf(data.getInternalFeature(j) )  
            l[j,lead_j,:] = l_row

    # g is the matrix of log posterior probabilities of the components given the data
    g = numpy.sum(l, axis=0) 
    for k in range(model.G):
        g[k,:] += numpy.log(model.pi[k])


    sum_logs = mixture.matrixSumlogs(g)
    g_norm = g - sum_logs
    tau = numpy.exp(g_norm)

    if not silent:
        print "\ntau="
        for tt in tau:
            print tt.tolist()
        print


    # computing posterior as model selection criterion
    temp = mixture.DiscreteDistribution(model.G,model.pi)
    pi_prior = model.prior.piPrior.pdf(temp)
    log_prior = pi_prior
    log_prior_list = [0.0] * model.dist_nr
    for j in range(model.dist_nr):
        for r in range(model.G):
               log_prior_list[j] += model.prior.compPrior[j].pdf( model.components[r][j] )

#    log_prior += sum(log_prior_list)
#
#    print '1:',log_prior

#    # get posterior
#    lk = numpy.sum(sum_logs) 


#    if penalty == 'POST':
#        # prior over number of components
#        log_prior += model.prior.nrCompPrior * model.G 
#        # prior over number of distinct groups
#        for j in range(model.dist_nr):
#            log_prior += model.prior.structPrior * len(model.leaders[j])   
#
#    elif penalty == 'BIC':
#        freeParams = model.G-1
#        for z in range(model.dist_nr):
#            #print 'Initial:',P[curr_indices[z]],'->',len(P[curr_indices[z]]) , model.components[0][j].freeParams
#            freeParams += model.components[0][z].freeParams * len(P[curr_indices[z]]) 
#
#        print '*** free params=',freeParams
#        print lk, freeParams,numpy.log(data.N)
#
#        log_prior += lk + (freeParams * numpy.log(data.N))  # BIC
#
#    elif penalty == 'AIC':
#        freeParams = model.G-1
#        for z in range(model.dist_nr):
#            #print 'Initial:',P[curr_indices[z]],'->',len(P[curr_indices[z]]) , model.components[0][j].freeParams
#            freeParams += model.components[0][z].freeParams * len(P[curr_indices[z]]) 
#        log_prior += lk + (2 * freeParams *  numpy.log(data.N))  # AIC
#    else:
#        raise TypeError



    #best_post = lk + log_prior
    best_post = float('-inf')
    
#    if not silent:
#        print best_structure,':'
#        print "0: ",  lk ,"+", log_prior,"=", best_post
#        #print log_prior_list


    # initialize merge histories
    L = [{} for j in range(model.dist_nr)]
    for j in range(model.dist_nr):

        # extracting current feature from the DataSet
        if isinstance(model.components[0][j], mixture.MixtureModel): # XXX
            data_j = data.singleFeatureSubset(j)
        else:
            data_j = data.getInternalFeature(j)

    
        for lead in range(model.G):  # every component is a leader for initialization
           
            el_dist = copy.copy(model.components[lead][j])
            tau_pool = copy.copy(tau[lead, :])
            pi_pool = model.pi[lead]

            if objFunction == 'MAP':
                model.prior.compPrior[j].mapMStep(el_dist, tau_pool, data_j,  pi_pool)  
            else:
                # should never get here...
                raise TypeError

            stat = el_dist.sufficientStatistics(tau_pool, data_j)

            M = mixture.CandidateGroup(el_dist, numpy.sum(tau_pool), pi_pool, stat)
            
            l_row = el_dist.pdf(data_j)  
            M.l = l_row
            M.dist_prior = model.prior.compPrior[j].pdf( el_dist ) 

            L[j][(lead,)] = M

    g_wo_j = numpy.zeros((model.G, data.N),dtype='Float64')
    best_indices = copy.copy(curr_indices)
    while 1:

        if not silent:
            print '\n----------------------------------'

        curr_indices[0] += 1
        
        if curr_indices[0] > max_ind:
            
            #curr_indices[lpos] = 0
            for e in range(model.dist_nr):
                if e == model.dist_nr-1:
                    if curr_indices[e] > max_ind:
                        term = 1
                        break
                    
                if curr_indices[e] > max_ind:
                    curr_indices[e] = 0
                    curr_indices[e+1] += 1

        if term:
            break



        #print '\nprev:',prev_indices
        if not silent:
            print nr, ':',curr_indices  ,'->', [P[jj] for jj in curr_indices]

        g_wo_prev = copy.copy(g)
        g_this_struct = numpy.zeros((model.G, data.N))
        for j in range(model.dist_nr):
            if prev_indices[j] == curr_indices[j]:
                #print '   -> unchanged',j,curr_indices[j], P[curr_indices[j]]
                break
            else:
                #print '\n--------\nChanged',j,curr_indices[j], P[curr_indices[j]]
                curr_struct_j = P[curr_indices[j]]
                
                # unnormalized posterior matrix without the contribution of the jth feature
                try:
                    g_wo_prev  = g_wo_prev - l[j]
                except FloatingPointError:
                    # if there was an exception we have to compute each
                    # entry in g_wo_j seperately to set -inf - -inf = -inf
                    g_wo_prev = _C_mixextend.substract_matrix(g_wo_prev,l[j])
                
                # extracting current feature from the DataSet
                if isinstance(model.components[0][j], mixture.MixtureModel): # XXX
                    data_j = data.singleFeatureSubset(j)
                else:
                    data_j = data.getInternalFeature(j)


                l_j_1 = numpy.zeros( (model.G, data.N ) )  # XXX needs only be done once                                    
                
                
                #print '\n\n***', curr_struct_j
                
                for cs_j in curr_struct_j:
                    
                    #print '    ->',cs_j
                    
                    if L[j].has_key(cs_j):
                        #print '  ** REcomp',cs_j

                        # retrieve merge data from history
                        candidate_dist = L[j][cs_j].dist                            

                        if not silent:
                            print j,"  R  candidate:", cs_j, candidate_dist

                        l_row = L[j][cs_j].l
                        #cdist_prior = L[j][cs_j].dist_prior

                    else:
                        #print '  ** comp',cs_j
                        
                        M = model.prior.compPrior[j].mapMStepMerge( [L[j][(c,)]  for c in cs_j]  )



                        #print '\n   *** compute:',hist_ind1,hist_ind2

                        candidate_dist = M.dist

                        if not silent:
                            print j,"  C  candidate:", cs_j,candidate_dist

                        l_row = candidate_dist.pdf(data_j)  

                        #print '   l_row=',l_row

                        #cdist_prior = 

                        M.l = l_row
                        M.dist_prior = model.prior.compPrior[j].pdf( candidate_dist ) 

                        L[j][cs_j] = M
                        
                    for c in cs_j:
                        l_j_1[c,:] = l_row
                        #print '            ->',c
                        #g_this_struct[c,:] += l_row
                        
                g_this_struct += l_j_1
                l[j] = l_j_1


                # compute parameter prior for the candidate merge parameters
                log_prior_list_j = 0.0
                for r in curr_struct_j:
                    log_prior_list_j += L[j][r].dist_prior * len(r)
                log_prior_list[j] = log_prior_list_j


        # get updated unnormalized posterior matrix
        g_1 = g_wo_prev + g_this_struct

#                print '\ng_wo_j:'
#                for gg in g_wo_j:
#                    print gg.tolist()
#
#                print '\nl_j_1:'
#                for gg in l_j_1:
#                    print gg.tolist()
#
#
#        print '\ng_1:'
#        for gg in g_1:
#            print gg.tolist()


        sum_logs = mixture.matrixSumlogs(g_1)
        lk_1 = numpy.sum(sum_logs)

        #print '\n  *** likelihood =', lk_1

        # computing posterior as model selection criterion
        log_prior_1 = pi_prior

            #print r, L[r].dist_prior * len(r),L[r].dist_prior


        #print '\nlog_prior_list_j =',log_prior_list_j
        #print 'log_prior_list',log_prior_list

        log_prior_1 += sum(log_prior_list)

        #print '2:',log_prior_1


        if penalty == 'POST':

            # prior over number of components
            log_prior_1 += model.prior.nrCompPrior * model.G 
            # prior over number of distinct groups
            for z in range(model.dist_nr):
                log_prior_1 += model.prior.structPrior * len(P[curr_indices[z]]) 
            post_1 = lk_1 + log_prior_1
            if not silent:
                print '\n'+penalty+':',post_1 ,'=', lk_1 ,'+', log_prior_1            



        elif penalty == 'BIC':
            freeParams = model.G-1
            for z in range(model.dist_nr):
                #print P[curr_indices[z]],'->',len(P[curr_indices[z]]) , model.components[0][j].freeParams

                freeParams += model.components[0][z].freeParams * len(P[curr_indices[z]]) 

            #print '*** free params=',freeParams
            #print lk, freeParams,numpy.log(data.N)

            log_prior_1 +=  (freeParams *  numpy.log(data.N))  # BIC
            post_1 = -((-2*lk_1) + log_prior_1)
            if not silent:
                print '\n'+penalty+':',post_1 ,'=', lk_1 ,'+', log_prior_1            


        
        elif penalty == 'AIC':
            freeParams = model.G-1
            for z in range(model.dist_nr):
                #print P[curr_indices[z]],'->',len(P[curr_indices[z]]) , model.components[0][j].freeParams
                freeParams += model.components[0][z].freeParams * len(P[curr_indices[z]]) 

            log_prior_1 +=  (2 * freeParams )  # AIC
            post_1 = -((-2*lk_1) + log_prior_1)

            if not silent:
                print '\n'+penalty+':',post_1 ,'= - ', (-2*lk_1) ,'+', log_prior_1            


        else:
            raise TypeError


        #print '3:',log_prior_1


            
        if post_1 >= best_post: # current candidate structure is better than previous best
            if not silent:
                print "*** New best candidate", post_1 ,">=", best_post
                
            if post_1 == best_post:
                print '******* Identical maxima !'
                print 'current:',[P[jj] for jj in curr_indices]
                print 'best:',[P[jj] for jj in best_indices]
                
            best_indices = copy.copy(curr_indices)
            best_post = post_1
        
        
        nr += 1
        g = g_1  # set likelihood matrix for the next candidate structure
        
        # XXX DEBUG XXX
        #if nr > 500:
        #    term = 1
        
        prev_indices = copy.copy(curr_indices)
        

    # setting updated structure in model
    for j in range(model.dist_nr):
        lead = []
        groups = {}

        

        best_partition = P[ best_indices[j] ]
        
        #print '#####',j,best_indices[j],best_partition
        
        for gr in best_partition:
            gr_list = list(gr)
            gr_lead = gr_list.pop(0)
            lead.append(gr_lead)
            groups[gr_lead] = gr_list
            
            # assigning distributions according to new structure
            model.components[gr_lead][j] = L[j][gr].dist
            for d in gr_list:
                model.components[d][j] = model.components[gr_lead][j]



        model.leaders[j] = lead
        model.groups[j] = groups



#    print '** G=',model.G
#    print '** p=',model.dist_nr
#    print '** nr =',nr

    if not silent:
        print '\n*** Globally optimal structure out of',nr,'possible:'
        print  [P[ best_indices[j]] for j in range(model.dist_nr)]














































