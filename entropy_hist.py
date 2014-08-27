from mixture import *
from random import *

m = MixtureModel(3,[0.2,0.3,0.5],
                [ normalDistribution(1.0, 0.30),
                  normalDistribution(3.0, 0.80),
                  normalDistribution(2.0, 0.80)
                ] )

#print "Simple:"
m.entropyHistogram(5000,50)




m2 = MixtureModel(2,[0.5,0.5],
                [ normalDistribution(1.1, 1.0),
                  normalDistribution(2.5, 0.5)
                ] )

#print "Hard:"
#m2.entropyHistogram(10)
