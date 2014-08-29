import copy
import sys

import numpy

from core.pymix_util.errors import ConvergenceFailureEM, InvalidPosteriorDistribution
from core.pymix_util.dataset import DataSet
from core.pymix_util.maths import sumlogs
from core.pymix_util.stats import sym_kl_dist


class CandidateGroup:
    """
    CandidateGroup is a simple container class.
    It holds the parameters and sufficient statistics for a candidate grouping
    in the CSI structure. It is used as part of the structure learning.
    """

    def __init__(self, dist, post_sum, pi_sum, req_stat, l=None, dist_prior=None):
        """
        Constructor

        @param dist: candidate distribution
        @param post_sum: sum over component membership posterior for candidate distribution
        @param pi_sum:  sum of pi's corresponding to candidate distribution
        @param req_stat:  additional statistics required  for paramter updates
        @param l: vector of likelihoods induced by the candidate distribution for each sample in a single feature
        @param dist_prior:  prior density over candidate distribution
        """

        self.dist = dist  # candidate distribution
        self.post_sum = post_sum  # sum over posterior for candidate merge
        self.pi_sum = pi_sum   # sum of pi's corresponding to candidate merge
        self.req_stat = req_stat  # additional required statistics for paramter updates by merge

        self.l = l       # vector of likelihoods of the merge for each sample in a single feature
        self.dist_prior = dist_prior  # prior density of candidate distribution

    #----------------------------------------------------------------------------------


    # --------------------------------------------------------------------------------------------------------------------

