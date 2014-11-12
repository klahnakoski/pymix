import random
from pymix.distributions.normal import NormalDistribution

from pymix import mixture
from pymix.models.mixture import MixtureModel
from pymix.util.dataset import DataSet
from pymix.vendor.ghmm.emission_domain import Alphabet


VNTR = Alphabet(['.', '2/4', '2/7', '3/4', '3/7', '4/4', '4/6', '4/7', '4/8', '4/9', '7/7'])
DIAG = Alphabet(['.', '0', '8', '1'])

data = DataSet()

# iq.txt = iq and achievement test fields from pheno.txt
# drd4_len.txt = drd4 vntr types, only number of repeats
data.fromFiles(["iq.txt", "phys.txt", "drd4_len.txt"])

COMOR = 11
G = 8
components = []
for i in range(G):


    # intelligence and achivement tests as univariate normal distributions. (TEST)
    bd_mu = float(random.randint(3, 16))
    bd_sigma = random.uniform(1.0, 8.0)
    missing_bd = NormalDistribution(-9999.9, 0.00001)
    dist_bd = NormalDistribution(bd_mu, bd_sigma)
    mix_bd = MixtureModel(2, [0.999, 0.001], [dist_bd, missing_bd], compFix=[0, 2])

    voc_mu = float(random.randint(3, 16))
    voc_sigma = random.uniform(1.0, 8.0)
    missing_voc = NormalDistribution(-9999.9, 0.00001)
    dist_voc = NormalDistribution(voc_mu, voc_sigma)
    mix_voc = MixtureModel(2, [0.999, 0.001], [dist_voc, missing_voc], compFix=[0, 2])

    read_mu = float(random.randint(80, 120))
    read_sigma = random.uniform(1.0, 28.0)
    missing_read = NormalDistribution(-9999.9, 0.00001)
    dist_read = NormalDistribution(read_mu, read_sigma)
    mix_read = MixtureModel(2, [0.999, 0.001], [dist_read, missing_read], compFix=[0, 2])

    math_mu = float(random.randint(80, 120))
    math_sigma = random.uniform(1.0, 28.0)
    missing_math = NormalDistribution(-9999.9, 0.00001)
    dist_math = NormalDistribution(math_mu, math_sigma)
    mix_math = MixtureModel(2, [0.999, 0.001], [dist_math, missing_math], compFix=[0, 2])

    spelling_mu = float(random.randint(80, 120))
    spelling_sigma = random.uniform(1.0, 28.0)
    missing_spelling = NormalDistribution(-9999.9, 0.00001)
    dist_spelling = NormalDistribution(spelling_mu, spelling_sigma)
    mix_spelling = MixtureModel(2, [0.999, 0.001], [dist_spelling, missing_spelling], compFix=[0, 2])

    # diagnoses for cormobidit disorders
    #"ODD"	"CONDUCT"	"SOC PHO"	"SEP ANX"	"SPEC PHO"	"ENUR NOC"	"ENUR DIU"	"ENCOPRES"	"TOURET"	"TIC CRON"	"TIC TRAN"
    comor = []
    for j in range(COMOR):
        p_comor = [0.0] + mixture.random_vector(3)
        comor_missing = mixture.MultinomialDistribution(1, 4, [1.0, 0.0, 0.0, 0.0], DIAG)
        comor_mult = mixture.MultinomialDistribution(1, 4, p_comor, DIAG)
        comor_mix = MixtureModel(2, [0.999, 0.001], [comor_mult, comor_missing], compFix=[0, 2])
        comor.append(comor_mix)
    pd_comor = mixture.ProductDistribution(comor)


    # the drd4 VNTR are represented as a discrete distribution over the observed lengths,
    # the specific repeat sequence tpyes are not considered at this time
    p_drd4_vntr_len = [0.0] + mixture.random_vector(10)

    dist_drd4_vntr_len = mixture.MultinomialDistribution(1, 11, p_drd4_vntr_len, VNTR)
    vntr_missing = mixture.MultinomialDistribution(1, 11, [1.0] + [0.0] * 10, VNTR)
    mix_drd4_vntr_len = MixtureModel(2, [0.999, 0.001], [dist_drd4_vntr_len, vntr_missing], compFix=[0, 2])

    components.append(mixture.ProductDistribution([mix_bd, mix_voc, mix_read, mix_math, mix_spelling, pd_comor, mix_drd4_vntr_len]))

m_pi = mixture.random_vector(G)
m = MixtureModel(G, m_pi, components)



#z = m.cluster( data, nr_runs=2,nr_init=3, max_iter=30, delta=0.1, labels = None, entropy_cutoff = None,tilt=0)

m.nr_tilt_steps = 10
l = m.randMaxEM(data, 1, 30, 0.1, tilt=0)
#z =m.classify(data)

#mixture.writeMixture(m,"iq_tilt_gsl.mix")

#l = m.EM(data,10,0.1,tilt=0)

