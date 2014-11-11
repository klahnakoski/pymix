from pymix.parse import readMixture
from pymix.util.dataset import DataSet
from pymix.vendor.ghmm.emission_domain import Alphabet

VNTR = Alphabet(['.', '2/4', '2/7', '3/4', '3/7', '4/4', '4/6', '4/7', '4/8', '4/9', '7/7'])
DIAG = Alphabet(['.', '0', '8', '1'])

data = DataSet()

# iq.txt = iq and achievement test fields from pheno.txt
# drd4_len.txt = drd4 vntr types, only number of repeats
data.fromFiles(["filt_WISC_WIAT_DISC_134.txt"]) # ,"DRD4_134_len.txt"

m = readMixture('pheno_best.py')

print "Without deterministic anealing:"
m.randMaxEM(data, 100, 30, 0.1, tilt=0, silent=0)

print "\nWith deterministic annealing:"
m.randMaxEM(data, 100, 30, 0.1, tilt=1, silent=0)
