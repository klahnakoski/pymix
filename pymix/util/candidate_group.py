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

