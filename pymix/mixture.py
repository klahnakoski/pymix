################################################################################
#
#       This file is part of the Modified Python Mixture Package, original
#       source code is from https://svn.code.sf.net/p/pymix/code.  Also see
#       http://www.pymix.org/pymix/index.php?n=PyMix.Download
#
#       Changes made by: Kyle Lahnakoski (kyle@lahnakoski.com)
#
################################################################################
#
#       This file is part of the Python Mixture Package
#
#       file:    mixture.py
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
################################################################################


"""

PyMix - Python Mixture Package

The PyMix library implements algorithms and data structures for data mining
with finite mixture models. The framework is object oriented and
organized in a hierarchical fashion.



"""
import sys
import logging
import numpy as np

from pymix.util.stats import get_loglikelihood
from pymix.util import setPartitions


# ToDo: use logging package for verbosity control, makes all 'silent' parameters superfluous,
# needs to trickle down into the whole library

log = logging.getLogger("PyMix")

# creating StreamHandler to stderr
hdlr = logging.StreamHandler(sys.stderr)

# setting message format
#fmt = logging.Formatter("%(name)s %(asctime)s %(filename)s:%(lineno)d %(levelname)s %(thread)-5s - %(message)s")
fmt = logging.Formatter("%(name)s %(filename)s:%(lineno)d - %(message)s")
hdlr.setFormatter(fmt)

# adding handler to logger object
log.addHandler(hdlr)

# Set the minimal severity of a message to be shown. The levels in
# increasing severity are: DEBUG, INFO, WARNING, ERROR, CRITICAL
log.setLevel(logging.ERROR)


# By default numpy produces a warning whenever we call  np.log with an array
# containing zero values. Usually this will happen a lot, so we change  numpys error handling
# to ignore this error. Since  np.log returns -inf for zero arguments the computations run
# through just fine.
np.seterr(divide="ignore", invalid="raise")


def remove_col(matrix, index):
    """
    Removes a column in a Python matrix (list of lists)

    @param matrix: Python list of lists
    @param index: index of column to be removed

    @return: matrix with column deleted
    """
    for i in range(len(matrix)):
        del (matrix[i][index])
    return matrix


#--------------------------- TEST ------------------------------------------------
class CandidateGroupHISTORY:  # XXX reomve ? ...
    def __init__(self, l, dist_prior, dist):
        #self.indices = indices  # leader indices for this merge
        #self.post = post   # total posterior of this merge
        self.l = l       # vector of likelihoods of the merge for each sample in a single feature
        self.dist_prior = dist_prior  # prior density of candidate distribution
        self.dist = dist  # candidate distribution

        #self.lead = lead  # candidate leaders
        #self.groups = groups  # candidate groups

        #self.l_j_1 = l_j_1
        #self.log_prior_list_j = log_prior_list_j



#---------------------------------- Partial Supervised Learning -------------------------------------------------




#---------------------------------- Miscellaneous -------------------------------------------------


def structureAccuracy(true, m):
    """
    Returns the accuracy of two model structures with respect to
    the component partition they define.

    @param true: MixtureModel object with CSI structure
    @param m: MixtureModel object with CSI structure

    @return: agreement of the two structures as measure by the accuracy
    """

    tp = fn = tn = fp = 0
    for j in range(len(true.leaders)):

        ltrue = setPartitions.encode_partition(true.leaders[j], true.groups[j], true.G)
        lm = setPartitions.encode_partition(m.leaders[j], m.groups[j], m.G)

        # For all unordered pairs
        for i in range(m.G):
            for j in range(i + 1, m.G):

                if ltrue[i] == ltrue[j]: # (i,j) is a positive
                    if lm[i] == lm[j]:
                        tp += 1
                    else:
                        fn += 1

                else: # (i,j) is a negative
                    if lm[i] == lm[j]:
                        fp += 1
                    else:
                        tn += 1

    acc = float(tp + tn) / (tp + fn + tn + fp)

    return acc


def modelSelection(data, models, silent=False, NEC=1):
    """
    Computes model selection criterias NEC, BIC and AIC for a list of models.

    @param data: DataSet object
    @param models: list of MixtureModel objects order with ascending number of components.

    @return: list of optimal components number according to [NEC, BIC, AIC], in that order.
    """
    #assert models[0].G == 1, "One component model needed for NEC."
    if models[0].G != 1:
        print "warning: One component model needed for NEC."

    m_1 = models[0]
    data.internalInit(m_1)
    L_1 = get_loglikelihood(m_1, data)
    #P_1 = L_1 + m_1.prior.pdf(m_1)

    G_list = [1]
    NEC = [1.0]
    BIC = [-2 * L_1 - (m_1.freeParams * np.log(data.N))]
    AIC = [-2 * L_1 + ( 2 * m_1.freeParams )]
    #bBIC = [ -2*P_1 - (m_1.freeParams * np.log(data.N)) ]  # test: BIC with MAP instead of ML
    for i in range(1, len(models)):
        m_i = models[i]
        G_list.append(m_i.G)
        (log_l, L_G) = m_i.EStep(data)
        l = np.exp(log_l)

        if m_i.G == 1:
            NEC.append(1.0)  # if G=1, then NEC = 1.0 by definition
        else:

            # entropy term of the NEC
            E_g = 0
            for g in range(m_i.G):
                for n in range(data.N):
                    if log_l[g, n] != float('-inf'):
                        E_g += l[g, n] * log_l[g, n]
            E_g = -E_g
            NEC_G = E_g / ( L_G - L_1 )
            NEC.append(NEC_G)

        BIC_G = -2 * L_G + (m_i.freeParams * np.log(data.N))
        BIC.append(BIC_G)

        AIC_G = -2 * L_G + ( 2 * m_i.freeParams )
        AIC.append(AIC_G)

    if not silent:
        print "NEC = ", NEC
        print "BIC = ", BIC
        print "AIC = ", AIC

    NEC_min = np.argmin(np.array(NEC, dtype='Float64'))
    AIC_min = np.argmin(np.array(AIC, dtype='Float64'))
    BIC_min = np.argmin(np.array(BIC, dtype='Float64'))

    if not silent:
        print G_list
        print '**', NEC_min
        print "Minimal NEC at G = " + str(G_list[NEC_min]) + " with " + str(NEC[NEC_min])
        print "Minimal BIC at G = " + str(G_list[BIC_min]) + " with " + str(BIC[BIC_min])
        print "Minimal AIC at G = " + str(G_list[AIC_min]) + " with " + str(AIC[AIC_min])

    return (NEC, BIC, AIC)

