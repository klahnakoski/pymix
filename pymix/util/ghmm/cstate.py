#
#    State class for continuous emission HMMs.
#
from pymix.util.ghmm.wrapper import ARRAY_CALLOC, ighmm_cmatrix_alloc


class ghmm_cstate:
    def __init__(self, M, N, cos):
        #  Number of output densities per state
        self.M = M  # int
        #  weight vector for output function components
        self.c = ARRAY_CALLOC(M)  # double *
        #  vector of Emission (type and parameters of output function components)
        self.e = [None]*M  # Emission *


        #  initial prob.
        self.pi = None  # double

        #   matrix in case of mult. transition matrices (COS > 1)
        #  number of  successor states
        self.out_a = ighmm_cmatrix_alloc(cos, N)

        #  transition probs to successor states. It is a
        #  transition probs from predecessor states. It is a
        #   matrix in case of mult. transition matrices (COS > 1)
        self.in_a = None  # double **
        #  number of  predecessor states
        self.in_a = ighmm_cmatrix_alloc(cos, N)

        #  flag for fixation of parameter. If fix = 1 do not change parameters of
        #      output functions, if fix = 0 do normal training. Default is 0.
        self.fix = 0  # int

        #  contains a description of the state
        self.desc = None


    def setDensity(self, i, type):
        self.e[i].type = type

    def getEmission(self, i):
        return self.e[i]

    def getOutState(self, index):
        return index

    def getInState(self, index):
        return index

    def getOutProb(self, i, c=0):
        return self.out_a[c][i]

    def getInProb(self, i, c=0):
        return self.in_a[c][i]

    def getMean(self, i):
        return self.e[i].mean

    def getStdDev(self, i):
        return self.e[i].variance

    def setMean(self, i, value):
        self.e[i].mean = value

    def setStdDev(self, i, value):
        self.e[i].variance = value

    def setWeight(self, i, value):
        if not self.c:
            self.c = [None] * self.M
        self.c[i] = value

    def getWeight(self, i):
        return self.c[i]


    #============================================================================
    # PDF(omega) in a given state
    def calc_b(self, omega):
        b = 0.0

        for m in range(self.M):
            b += self.c[m] * self.e[m].linear_pdf(omega)
        return b

    def calc_cmbm(self, m, omega):
        emission = self.e[m]
        # return self.c[m] * density_func[emission.type](emission, omega)
        return emission.linear_pdf(omega)

