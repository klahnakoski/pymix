import mixture
import numpy
import random


def testdtree():

        tree = {}
        tree[0] = -1
        tree[1] = 0
        tree[2] = 1

        
        n1 = mixture.ProductDistribution([mixture.ConditionalGaussDistribution(3,[0, 1, 0],
                                                                                 [0, -0.1, 0.1],
                                                                                 [0.5,0.5,0.5],tree)])
        tree2 = {}
        tree2[0] = -1
        tree2[1] = 0
        tree2[2] = 0	
        n2 = mixture.ProductDistribution([mixture.ConditionalGaussDistribution(3,[-1, 0, 1],
                                                                                 [0, 0.1, -0.1],
                                                                                 [0.5,0.5,0.5],tree2)])
         
        pi = [0.4, 0.6]
        gen = mixture.MixtureModel(2,pi,[n1,n2])

        random.seed(1)        
        data = gen.sampleDataSet(1000)

        print data


        
        n1 = mixture.ProductDistribution([mixture.DependenceTreeDistribution(3,[0.1, 1.1, 0.1],
                                                                                 [0, 0, 0],
                                                                                 [1.0,1.0,1.0])])
        n2 = mixture.ProductDistribution([mixture.DependenceTreeDistribution(3,[-1, 0, -0.1],
                                                                                 [0, 0, 0],
                                                                                 [1.0,1.0,1.0])])
	
        
        n1 = mixture.ProductDistribution([mixture.ConditionalGaussDistribution(3,[0, 1, 0],
                                                                                 [0.0, 0.1, 0.1],
                                                                                 [0.1,0.1,0.1],tree)])
        n2 = mixture.ProductDistribution([mixture.ConditionalGaussDistribution(3,[-1, 0, 1],
                                                                                 [0.0, 0.1, 0.1],
                                                                                 [0.1,0.1,0.1],tree2)])
        
        train = mixture.MixtureModel(2,pi,[n1,n2])
	train.modelInitialization(data)
        train.EM(data,100,0.01,silent=1)
        


def testLymphData():
	
	k = 5
	d = 11

	aux = [0]*d

	models = []

	for i in range(k):
	    aux1 = [0]*d
	    aux2 = [0]*d
	    aux3 = [0]*d	    
  	    models.append(mixture.ProductDistribution([mixture.DependenceTreeDistribution(d,aux1,aux2,aux3)]))

        pi = [1.0]*k
	pi = numpy.array(pi)/k
	
        
        train = mixture.MixtureModel(k,pi,models)

        data = mixture.DataSet()
	data.fromFiles(['data/ltree2_2fold.txt'],)
		
	train.modelInitialization(data)
	
        train.EM(data,100,0.01,silent=1)


	

testdtree()


#testLymphData()
