import numpy
from core.distributions.conditional_gauss import ConditionalGaussDistribution
from core.pymix_util.dataset import DataSet


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
        elif isinstance(data, numpy.ndarray):
            x = data
        else:
            raise TypeError, "Unknown/Invalid input to MStep."

        post = posterior.sum() # sum of posteriors
        self.mu = numpy.dot(posterior, x) / post

        # centered input values (with new mus)
        centered = numpy.subtract(x, numpy.repeat([self.mu], len(x), axis=0));


        # estimating correlation factor
        sigma = numpy.dot(numpy.transpose(numpy.dot(numpy.identity(len(posterior)) * posterior, centered)), centered) / post # sigma/covariance matrix

        diagsigma = numpy.diagflat(1.0 / numpy.diagonal(sigma)) # vector with diagonal entries of sigma matrix
        correlation = numpy.dot(numpy.dot(diagsigma, numpy.multiply(sigma, sigma)), diagsigma) # correlation matrix with entries sigma_xy^2/(sigma^2_x * sigma^2_y)

        correlation = correlation - numpy.diagflat(numpy.diagonal(correlation)) # making diagonal entries = 0

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
        for i in range(self.p):
            edgestree[i] = []
        verticestree = [0]

        while len(verticestree) < self.p:
            # possible edges = only ones form vertices at the current tree
            candidates = weights[verticestree, :]

            # look for maximal candidate edges
            indices = numpy.argmax(candidates, 1) # max neighboors in verticestrrees
            values = numpy.max(candidates, 1)
            uaux = numpy.argmax(values)
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
        visited = numpy.zeros((self.p, 1))
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
        return 'Dependence Tree: \nmu=' + str(self.mu) + ' \nsigma=' + str(self.sigma) + '\nw=' + str(self.w) + '\nparents=' + str(self.parents)


