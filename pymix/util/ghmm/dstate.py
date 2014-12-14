from pymix.util.ghmm.wrapper import GHMM_EPS_PREC


class ghmm_dstate():
    def __init__(self):
        # Output probability
        self.b = [0.0] #double*

        # Initial probability
        self.pi = 0.0 #double

        # transition probabilities to successor states.
        self.out_a = [] #double*
        # Number of successor states

        # transition probabilities from predecessor states.
        self.in_a = [] #double*
        # Number of precursor states

        # if fix == 1 -. b stays fix during the training
        self.fix = 0 #int
        # contains a description of the state (null terminated utf-8)
        self.desc = None #char*


    def forward_step(self, alpha_t, b_symb):
        value = 0.0

        if b_symb < GHMM_EPS_PREC:
            return 0

        for i in range(len(self.in_a)):
            value += self.in_a[i] * alpha_t[i]

        value *= b_symb
        return value

    def getOutState(self, index):
        return index

def model_state_alloc(
    M,
    N
):
    s = ghmm_dstate()
    s.b = [0.0] * M
    s.out_a = [0.0] * N
    s.in_a = [0.0] * N

    return s


