from mixture import *
import numarray
from Graph import *



def compStructure(v1,v2):
    nr_v = len(v1)
    
    G1 = Graph()
    G2 = Graph()
    
    
    for p in range(len(v1[0])):
    
        for i in range(1,nr_v+1):
            G1.AddVertex()
            G2.AddVertex()
        
        # finding number of unique entries in v1 and v2
        d_v1 = {}
        d_v2 = {}
        
        v1_ind = 0
        v2_ind = 0
        for i in range(nr_v):
            if not d_v1.has_key( v1[i,p] ):   
                t = numarray.where( v1[:,p] == v1[i,p] )
   #             d_v1[v1[i,p]] = 
            
                v1_ind +=1
            if not d_v2.has_key( v2[i,p] ):   
                d_v2[v2[i,p]] = v2_ind
                v2_ind += 1

  #      for k in d_v1.keys():
            
        
        
        #print d_v1
        #print d_v2
    
    
    
    
#    for i in range(len(matrix)):
#        for j in range(len(matrix)):
#            if (i == j):
#                continue
#            G.AddEdge(i+1,j+1,matrix[i][j])
#    return G
    


G = numarray.array([ [3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3] ,
[3, 4, 4, 4, 4, 4, 4, 4, 3, 1, 4, 4, 4, 4, 1, 4, 4, 3, 1, 4, 4, 4, 4, 4, 4] ,
[4, 5, 5, 3, 3, 4, 3, 4, 3, 1, 3, 5, 4, 4, 1, 4, 4, 3, 1, 5, 5, 4, 4, 4, 4] ,
[5, 3, 3, 5, 5, 4, 3, 3, 4, 1, 5, 5, 4, 4, 1, 3, 5, 3, 1, 5, 5, 5, 4, 4, 4] ,
[5, 6, 6, 5, 3, 5, 4, 4, 5, 1, 5, 6, 5, 4, 1, 5, 4, 3, 1, 6, 6, 4, 4, 4, 4] ,
[3, 3, 5, 6, 4, 5, 4, 4, 4, 1, 3, 4, 5, 4, 1, 4, 4, 3, 1, 6, 6, 4, 4, 4, 4] ,
[6, 4, 7, 4, 6, 4, 3, 4, 5, 1, 4, 6, 5, 4, 1, 5, 4, 4, 1, 5, 5, 4, 4, 4, 5] ])


T = numarray.array([ [3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3] ,
[3, 4, 4, 4, 4, 4, 4, 4, 3, 1, 4, 4, 4, 4, 1, 4, 4, 3, 1, 4, 4, 4, 4, 4, 4] ,
[4, 5, 5, 3, 3, 4, 3, 4, 3, 1, 3, 5, 4, 4, 1, 4, 4, 3, 1, 5, 5, 4, 4, 4, 4] ,
[5, 3, 3, 5, 5, 4, 3, 3, 4, 1, 5, 5, 4, 4, 1, 3, 5, 3, 1, 5, 5, 5, 4, 4, 4] ,
[5, 6, 6, 5, 3, 5, 4, 4, 5, 1, 5, 6, 5, 4, 1, 5, 4, 3, 1, 6, 6, 4, 4, 4, 4] ,
[3, 3, 5, 6, 4, 5, 4, 4, 4, 1, 3, 4, 5, 4, 1, 4, 4, 3, 1, 6, 6, 4, 4, 4, 4] ,
[6, 4, 7, 4, 6, 4, 3, 4, 5, 1, 4, 6, 5, 4, 1, 5, 4, 4, 1, 5, 5, 4, 4, 4, 5] ])


#compStructure(G,T)
#raise RuntimeError




DIAG = Alphabet(['.','0','8','1'])


noise1 = MultinomialDistribution(1,4,[0.5,0.15,0.15,0.2],alphabet = DIAG)
noise2 = NormalDistribution(0,1.0)
noise3 = MultinomialDistribution(1,4,[0.1,0.4,0.4,0.1],alphabet = DIAG)
noise4 = MultinomialDistribution(1,4,[0.7,0.1,0.1,0.1],alphabet = DIAG)
noise5 = NormalDistribution(2.5,0.5)
noise6 = NormalDistribution(9.0,1.0)
noise7 = NormalDistribution(19.0,1.0)
noise8 = NormalDistribution(2.0,1.0)

nlist = [noise1,noise2,noise3,noise4,noise5,noise6,noise7]


n1 = NormalDistribution(2.5,0.5)
n2 = NormalDistribution(3.2,0.5)
n3 = NormalDistribution(2.5,0.5)
n4 = NormalDistribution(-5.5,0.5)
n5 = NormalDistribution(-3.0,0.5)
d1 = MultinomialDistribution(1,4,[0.25,0.25,0.25,0.25],alphabet = DIAG)
d2 = MultinomialDistribution(1,4,[0.6,0.2,0.1,0.1],alphabet = DIAG)
mark1 = NormalDistribution(0,0.1)
pd = ProductDistribution([n1,n2,n3,n4,n5,d1,d2,mark1] + nlist )

n6 = NormalDistribution(2.52,0.495)
n7 = NormalDistribution(2.52,0.495)
n8 = NormalDistribution(3.52,0.495)
n9 = NormalDistribution(5.52,0.495)
n10 = NormalDistribution(1.0,0.5)
d3 = MultinomialDistribution(1,4,[0.23,0.27,0.23,0.27],alphabet = DIAG)
d4 = MultinomialDistribution(1,4,[0.5,0.15,0.15,0.2],alphabet = DIAG)
mark2 = NormalDistribution(20,0.1)
pd2 = ProductDistribution([n6,n7,n8,n9,n10,d3,d4,mark2]+ nlist )

n11 = NormalDistribution(9.0,1.0)
n12 = NormalDistribution(2.50,0.490)
n13 = NormalDistribution(2.52,0.495)
n14 = NormalDistribution(5.52,0.495)
n15 = NormalDistribution(1.2,0.5)
d5 = MultinomialDistribution(1,4,[0.21,0.27,0.27,0.25],alphabet = DIAG)
d6 = MultinomialDistribution(1,4,[0.1,0.4,0.4,0.1],alphabet = DIAG)
mark3 = NormalDistribution(40,0.1)
pd3 = ProductDistribution([n11,n12,n13,n14,n15,d5,d6,mark3]+ nlist )

n16 = NormalDistribution(9.0,1.0)
n17 = NormalDistribution(2.50,0.490)
n18 = NormalDistribution(2.52,0.495)
n19 = NormalDistribution(5.52,0.495)
n20 = NormalDistribution(1.2,0.5)
d7 = MultinomialDistribution(1,4,[0.21,0.27,0.27,0.25],alphabet = DIAG)
d8 = MultinomialDistribution(1,4,[0.1,0.4,0.4,0.1],alphabet = DIAG)
mark4 = NormalDistribution(60,0.1)
pd4 = ProductDistribution([n16,n17,n18,n19,n20,d7,d8,mark4]+ nlist )

n21 = NormalDistribution(4.0,1.0)
n22 = NormalDistribution(2.50,0.490)
n23 = NormalDistribution(2.52,0.495)
n24 = NormalDistribution(5.52,0.495)
n25 = NormalDistribution(9.2,0.5)
d9 = MultinomialDistribution(1,4,[0.21,0.27,0.27,0.25],alphabet = DIAG)
d10 = MultinomialDistribution(1,4,[0.7,0.1,0.1,0.1],alphabet = DIAG)
mark5 = NormalDistribution(80,0.1)
pd5 = ProductDistribution([n21,n22,n23,n24,n25,d9,d10,mark5]+ nlist )

n26 = NormalDistribution(4.0,1.0)
n27 = NormalDistribution(2.50,0.490)
n28 = NormalDistribution(2.52,0.495)
n29 = NormalDistribution(5.52,0.495)
n30 = NormalDistribution(-4.95,0.5)
d11 = MultinomialDistribution(1,4,[0.21,0.27,0.27,0.25],alphabet = DIAG)
d12 = MultinomialDistribution(1,4,[0.7,0.1,0.1,0.1],alphabet = DIAG)
mark6 = NormalDistribution(100,0.1)
pd6 = ProductDistribution([n26,n27,n28,n29,n30,d11,d12,mark6]+ nlist )



mix = MixtureModel(6,[0.1,0.1,0.1,0.2,0.2,0.3], [ pd,pd2,pd3,pd4,pd5,pd6 ],struct =1)

data = mix.sampleDataSet(500)
#print mix

mix.updateStructureGlobal(data)

#print mix
#print mix.groups
#print mix.leaders

#writeMixture(mix, "test.mix")

#mix.evalStructure(data.headers)



plot = numarray.zeros(( mix.G,mix.dist_nr ) )
for i in range(mix.dist_nr):
    
    for l in mix.leaders[i]:    
        plot[l,i] = l+1
        for g in mix.groups[i][l]:
           plot[g,i] = l+1

print "Generating", get_loglikelihood(mix, data.internalData)
#print "G ="
#print range(0,len(plot[0]))
for p in plot:
    print p

for k in range(5):
    m = MixtureModel(6,[0.1,0.1,0.1,0.2,0.2,0.3], [ pd,pd2,pd3,pd4,pd5,pd6 ],struct =1)
    m.modelInitialization(data.internalData,rtype=1)


    #print "vorher: "
    #print m
    #print m.leaders
    #print m.groups
    
    logp = m.randMaxEM(data,15,40,0.1,tilt=0,silent=1)
    ch = m.updateStructureGlobal(data)


    #print "nachher:"
    #print m
    #print m.leaders
    #print m.groups


    
    
    plot2 = numarray.zeros(( m.G,m.dist_nr ) )
    for i in range(m.dist_nr):
    
        for l in m.leaders[i]:    
            plot2[l,i] = l+1
            for g in m.groups[i][l]:
               plot2[g,i] = l+1

    #print mix.groups
    #print m.groups




    l = []
    for g in range(m.G):
        l.append( (m.components[g].distList[7].mu,g) )
        #print  (m.components[g].distList[7])

    #print l
    l.sort()
    print l

    plot_sort = numarray.zeros(( mix.G,mix.dist_nr ) )
    print "\nTrained ",logp,":"
    #print "T =\n"
    #print range(0,len(plot2[0]))
    for i in range(len(l)):
        print plot2[l[i][1]]
        plot_sort[i] = plot2[l[i][1]]

print "\nComparison:"
for i in range(len(plot[0])):
    print "g("+str(i)+"): ",plot[:,i]
    print "t("+str(i)+"): ",plot_sort[:,i],"\n"

print m

#for i in range(8):
#    [sens,spez] = evaluate(plot[:,i],plot2[:,i])
#    print "\nFeature:",i
#    print "Sens:",sens
#    print "Spez",spez

