#*
#      Emission bundles all emission parameters
#


class Emission:
    def __init__(self):
        #* specify the type of the density
        self.type = 0
        #* dimension > 1 for multivariate normals
        self.dimension = None  # int
        #* mean for output functions (pointer to mean vector
        #        for multivariate)
        self.mean = None
        #* variance or pointer to a covariance matrix
        #        for multivariate normals
        self.variance = None
        #* pointer to inverse of covariance matrix if multivariate normal
        #        else None
        self.sigmainv = None  # double *
        #* determinant of covariance matrix for multivariate normal
        self.det = None  # double
        #* Cholesky decomposition of covariance matrix A,
        #        if A = GG' sigmacd only holds G
        self.sigmacd = None  # double *
        #* minimum of uniform distribution
        #       or left boundary for rigth-tail gaussians
        self.min = None  # double
        #* maximum of uniform distribution
        #       or right boundary for left-tail gaussians
        self.max = None  # double
        #* if fixed != 0 the parameters of the density are fixed
        self.fixed = 0  # int


    def setDensity(self, type):
        self.type = type
