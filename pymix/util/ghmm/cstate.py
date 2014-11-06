#*
#    State class for continuous emission HMMs.
#
from pymix.util.ghmm.randvar import density_func
from pymix.util.ghmm.wrapper import ARRAY_CALLOC, ighmm_cmatrix_alloc


class ghmm_cstate:
    def __init__(self, M, in_states, out_states, cos):
        #* Number of output densities per state
        self.M = M  # int
        #* initial prob.
        self.pi = None  # double

        #* IDs of successor states
        self.out_id = None  # int *
        #   matrix in case of mult. transition matrices (COS > 1)
        self.out_a = None  # double **
        #* number of  successor states
        self.out_states = out_states  # int
        if out_states > 0:
            self.out_id = ARRAY_CALLOC(out_states)
            self.out_a = ighmm_cmatrix_alloc(cos, out_states)

        #* IDs of predecessor states
        self.in_id = None  # int *
        #* transition probs to successor states. It is a
        #* transition probs from predecessor states. It is a
        #   matrix in case of mult. transition matrices (COS > 1)
        self.in_a = None  # double **
        #* number of  predecessor states
        self.in_states = in_states  # int
        if in_states > 0:
            self.in_id = ARRAY_CALLOC(in_states)
            self.in_a = ighmm_cmatrix_alloc(cos, in_states)

        #* weight vector for output function components
        self.c = ARRAY_CALLOC(M)  # double *

        #* flag for fixation of parameter. If fix = 1 do not change parameters of
        #      output functions, if fix = 0 do normal training. Default is 0.
        self.fix = 0  # int
        #* vector of ghmm_c_emission
        #      (type and parameters of output function components)
        self.e = ARRAY_CALLOC(M)  # ghmm_c_emission *

        #* contains a description of the state (null terminated utf-8)
        self.desc = None  # char *
        #* x coordinate position for graph representation plotting *
        self.xPosition = 0.0  # int
        #* y coordinate position for graph representation plotting *
        self.yPosition = 0.0  # int

    def setDensity(self, i, type):
        self.e[i].type = type


    def getEmission(self, i):
        return self.e[i]

    def getOutState(self, index):
            return self.out_id[index]

    def getInState(self, index):
        return self.in_id[index]

    def getOutProb(self, i, c=0):
        return self.out_a[c][i]

    def getInProb(self, i, c=0):
        return self.in_a[c][i]

    def calc_cmbm(self, m, omega):
        emission = self.e[m]
        return self.c[m] * density_func[emission.type](emission, omega)


    #============================================================================
    # PDF(omega) in a given state
    def calc_b(self, omega):
        b = 0.0

        for m in range(self.M):
            b += self.c[m] * density_func[self.e[m].type](self.e[m], omega)
        return b
