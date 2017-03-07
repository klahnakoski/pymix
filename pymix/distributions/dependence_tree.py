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

import numpy as np
from .conditional_gauss import ConditionalGaussDistribution
from ..util.dataset import DataSet


class DependenceTreeDistribution(ConditionalGaussDistribution):
    """
    This class implemements a tree of conditional Gaussians, including the
    tree topology learning.
    """

    def __init__(self, p, mu, w, sigma):
        """
        Constructor

        @param p: dimensionality of the distribution
        @param mu: mean parameter vector
        @param w: covariance weights (representing off-diagonal entries in the full covariance matrix)
        @param sigma: standard deviations (diagonal entries of the covariance matrix)
        """

        parents = self.randomStructure(p)
        # linear initialization of tree structure
        struct = {}
        for i in range(p - 1):
            struct[i + 1] = i
        struct[0] = -1
        ConditionalGaussDistribution.__init__(self, p, mu, w, sigma, struct)


    def MStep(self, posterior, data, mix_pi=None):
        if isinstance(data, DataSet):
            x = data.internalData
        elif hasattr(data, "__iter__"):
            x = data
        else:
            raise TypeError, "Unknown/Invalid input to MStep."

        post = posterior.sum() # sum of posteriors
        self.mean = np.dot(posterior, x) / post

        # centered input values (with new mus)
        centered = np.subtract(x, np.repeat([self.mean], len(x), axis=0));


        # estimating correlation factor
        sigma = np.dot(np.transpose(np.dot(np.identity(len(posterior)) * posterior, centered)), centered) / post # sigma/covariance matrix

        diagsigma = np.diagflat(1.0 / np.diagonal(sigma)) # vector with diagonal entries of sigma matrix
        correlation = np.dot(np.dot(diagsigma, np.multiply(sigma, sigma)), diagsigma) # correlation matrix with entries sigma_xy^2/(sigma^2_x * sigma^2_y)

        correlation = correlation - np.diagflat(np.diagonal(correlation)) # making diagonal entries = 0

        # XXX - check this
        parents = self.maximunSpanningTree(correlation) # return maximun spanning tree from the correlation matrix
        self.parents = self.directTree(parents, 0) # by default direct tree from 0


        # XXX note that computational time could be saved as these functions share same suficient statistics
        ConditionalGaussDistribution.MStep(self, posterior, data, mix_pi)


    def maximunSpanningTree(self, weights):
        """
        Estimates the MST given a fully connected graph defined by the symetric matrix weights.
        using Prim`s algorithm.

        @param weights: edge weights
        """

        # start with an empty tree and random vertex
        edgestree = {}
        for i in range(self.dimension):
            edgestree[i] = []
        verticestree = [0]

        while len(verticestree) < self.dimension:
            # possible edges = only ones form vertices at the current tree
            candidates = weights[verticestree, :]

            # look for maximal candidate edges
            indices = np.argmax(candidates, 1) # max neighboors in verticestrrees
            values = np.max(candidates, 1)
            uaux = np.argmax(values)
            u = verticestree[uaux]
            v = indices[uaux]

            # add (u,v) att tree
            edgestree[u].append(v)
            edgestree[v].append(u)
            #edgestree[v] = u

            #zeroing all vertices between v and verticestree
            for i in verticestree:
                weights[v, i] = 0
                weights[i, v] = 0

            # add (v) at tree
            verticestree.append(v)

            return edgestree

    def directTree(self, tree, root):
        parent = {}
        queue = []
        # directing the tree from the root
        parent[root] = -1
        visited = np.zeros((self.dimension, 1))
        for u in tree[root]:
            queue.append((root, u))
            visited[root] = 1
        while len(queue) > 0:
            (u, v) = queue.pop()
            parent[v] = u
            for newv in tree[v]:
                if not visited[newv]:
                    queue.append((v, newv))
            visited[v] = 1
            return parent

    def randomStructure(self, p):
        # linear initialization of tree structure
        struct = {}
        for i in range(p - 1):
            struct[i + 1] = i
        struct[0] = -1
        return struct

    def __str__(self):
        return 'Dependence Tree: \nmu=' + str(self.mean) + ' \nsigma=' + str(self.variance) + '\nw=' + str(self.w) + '\nparents=' + str(self.parents)


