## function sumlogs is borrowed from GQLMixture.py
import math
import numpy as np
from core.pymix_util import mixextend
from core.pymix_util.maths import sum_logs


def kl_dist(d1, d2):
    """
    Kullback-Leibler divergence for two distributions. Only accept MultinomialDistribution and
    NormalDistribution objects for now.

    @param d1: MultinomialDistribution or NormalDistribution instance
    @param d2: MultinomialDistribution or NormalDistribution instance

    @return: Kullback-Leibler divergence between input distributions
    """
    # Kullback = sum[1..P](ln(SIGMA2/SIGMA1))
    # + sum[1..P](SIGMA1^2 / (2*(SIGMA2^2)))
    # + sum[1..P]((MU1-MU2)^2 / (2*(SIGMA2^2))) - P/2
    from core.distributions.multinomial import MultinomialDistribution
    from core.distributions.normal import NormalDistribution
    from core.distributions.product import ProductDistribution

    if isinstance(d1, NormalDistribution) and isinstance(d2, NormalDistribution):
        res = ( (0.5 * np.log(d2.sigma ** 2 / d1.sigma ** 2)) - 0.5 + d1.sigma ** 2 / (2 * d2.sigma ** 2)
                + (abs(d2.mu - d1.mu) ** 2) / (2 * d2.sigma ** 2) )
        return res
    elif isinstance(d1, MultinomialDistribution) and isinstance(d2, MultinomialDistribution):
        assert d1.M == d2.M
        en = 0
        for i in range(d1.M):
            en += d1.phi[i] * np.log(d1.phi[i] / d2.phi[i])
        return en
    elif isinstance(d1, ProductDistribution) and isinstance(d2, ProductDistribution):
        assert d1.dist_nr == d2.dist_nr == 1
        return kl_dist(d1[0], d2[0])

    else:
        raise TypeError, "Type mismatch or distribution not yet supported by kl_dist: " + str(d1.__class__) + ", " + str(d2.__class__)


def sym_kl_dist(d1, d2):
    """
    Symmetric Kullback-Leibler divergence for two distributions. Only accept MultinomialDistribution and
    NormalDistribution objects for now.

    @param d1: MultinomialDistribution or NormalDistribution instance
    @param d2: MultinomialDistribution or NormalDistribution instance

    @return: Symmetric Kullback-Leibler divergence between input distributions
    """
    from core.distributions.multinomial import MultinomialDistribution
    from core.distributions.normal import NormalDistribution
    from core.models.mixture import MixtureModel

    if ( (isinstance(d1, NormalDistribution) and isinstance(d2, NormalDistribution))
    or (isinstance(d1, MultinomialDistribution) and isinstance(d2, MultinomialDistribution)) ):
        d12 = kl_dist(d1, d2)
        d21 = kl_dist(d2, d1)
        dist = (d12 + d21 ) / 2.0
    elif isinstance(d1, MixtureModel) and isinstance(d2, MixtureModel):
        assert d1.G == d2.G, "Unequal number of components"
        d12 = 0
        d21 = 0
        for i in range(d1.G):
            d12 += d1.pi[i] * kl_dist(d1.components[i], d2.components[i])
            d21 += d2.pi[i] * kl_dist(d2.components[i], d1.components[i])
        dist = (d12 + d21 ) / 2.0
    else:
        raise TypeError, str(d1.__class__) + " != " + str(d2.__class__)

    if dist < 0.0:
        #raise ValueError,"Negative distance in sym_kl_dist."
        #print 'WARNING: Negative distance in sym_kl_dist.'
        return 0.0
    else:
        return dist


def computeErrors(classes, clusters):
    """
    For an array of class labels and an array of cluster labels
    compute true positives, false negatives, true negatives and
    false positives.

    Assumes identical order of objects.

    Class and cluster labels can be arbitrary data types supporting
    '==' operator.

    @param classes: list of class labels (true labels)
    @param clusters: list of cluster labels (predicted labels)

    @return: Ratios for true positives, false negatives, true
    negatives, false postitives   (tp, fn, tn, fp)
    """

    #print 'true:',classes
    # print 'pred:', clusters

    assert len(classes) == len(clusters)
    tp = fn = tn = fp = 0

    classList = []
    clustList = []
    # samples with cluster or class label -1 are excluded
    for i in xrange(len(classes)):
        if clusters[i] != -1 and classes[i] != -1:
            classList.append(classes[i])
            clustList.append(clusters[i])

            #else:
            #    print i,'discarded:' ,  classes[i] , clusters[i]

    # For all unordered pairs
    for i in xrange(len(classList)):
        for j in xrange(i + 1, len(classList)):
            if classList[i] == classList[j]: # (i,j) is a positive
                if clustList[i] == clustList[j]:
                    tp += 1
                else:
                    fn += 1

            else: # (i,j) is a negative
                if clustList[i] == clustList[j]:
                    fp += 1
                else:
                    tn += 1

    return (tp, fn, tn, fp)


def accuracy(classes, clusters):
    """
    Computes accuracy of a clustering solution

    @param classes: list of true class labels
    @param clusters: list of cluster labels
    @return: accuracy
    """
    (tp, fn, tn, fp) = computeErrors(classes, clusters)
    if (tp + tn) != 0:
        return float(tp + tn) / (tp + fp + tn + fn)
    else:
        return 0.0


def sensitivity(classes, clusters):
    """
    Computes sensitivity of a clustering solution

    @param classes: list of true class labels
    @param clusters: list of cluster labels
    @return: sensitivity
    """
    (tp, fn, tn, fp) = computeErrors(classes, clusters)
    if (tp + fn) != 0:
        return float(tp) / (tp + fn)
    else:
        return 0.0


def specificity(classes, clusters):
    """
    Computes specificity of a clustering solution

    @param classes: list of true class labels
    @param clusters: list of cluster labels
    @return: specificity
    """

    (tp, fn, tn, fp) = computeErrors(classes, clusters)
    if (tp + fp) != 0.0:
        return float(tp) / (tp + fp)
    else:
        return 0.0


def random_vector(nr, normal=1.0):
    """
    Returns a random probability vector of length 'nr'.
    Can be used to generate random parametrizations of a multinomial distribution with
    M = 'nr'.

    @param nr: lenght of output vector
    @param normal: sum over output vector, default 1.0

    @return: list with random entries sampled from a uniform distribution on [0,1] and normalized to 'normal'
    """

    alpha = np.array([1.0] * nr)

    p = mixextend.wrap_gsl_dirichlet_sample(alpha, nr)

    if float(normal) != 1.0:
        p = p * normal
    return p.tolist()


def variance(data):
    mean = data.mean()
    s = 0.0
    for i in range(len(data)):
        s += (data[i] - mean) ** 2
    return s / (len(data) - 1)


def entropy(p):
    """
    Returns the Shannon entropy for the probilistic vector 'p'.

    @param p: 'numpy' vector that sums up to 1.0
    """
    res = 0.0
    for i in range(len(p)):
        if p[i] != 0.0:
            res += p[i] * math.log(p[i], 2)
    return -res


def get_loglikelihood(mix_model, data):  # old implementation XXX
    # matrix of posterior probs: components# * (sequence positions)#
    l = np.zeros((mix_model.G, len(data)), dtype='Float64')
    col_sum = np.zeros(len(data), dtype='Float64')
    for i in range(mix_model.G):
        l[i] = np.log(mix_model.pi[i]) + mix_model.components[i].pdf(data)
    for j in range(len(data)):
        col_sum[j] = sum_logs(l[:, j]) # sum over jth column of l
    log_p = np.sum(col_sum)
    return log_p


def get_posterior(mix_model, data, logreturn=True):
    # matrix of posterior probs: components# * (sequence positions)#
    log_l = np.zeros((mix_model.G, len(data)), dtype='Float64')

    # computing log posterior distribution
    for i in range(mix_model.G):
        log_l[i] = math.log(mix_model.pi[i]) + mix_model.components[i].pdf(data)


    # computing data log likelihood as criteria of convergence
    # log_l is normalized in-place and likelihood is returned as log_p
    (log_l, log_p) = mixextend.get_normalized_posterior_matrix(log_l)

    if logreturn == True:
        return log_l
    else:
        return np.exp(log_l)
