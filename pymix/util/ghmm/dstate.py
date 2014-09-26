from util.ghmm.wrapper import GHMM_EPS_PREC


class ghmm_dstate():
    def __init__(self):
        # Initial probability */
        self.pi = 0.0 #double
        # Output probability */
        self.b = [0.0] #double*

        # IDs of the following states */
        self.out_id = [0] #int*
        # IDs of the previous states */
        self.in_id = [0] #int*

        # transition probabilities to successor states. */
        self.out_a = [0.0] #double*
        # transition probabilities from predecessor states. */
        self.in_a = [0.0] #double*

        # Number of successor states */
        self.out_states = 0 #int
        # Number of precursor states */
        self.in_states = 0 #int
        # if fix == 1 -. b stays fix during the training */
        self.fix = 0 #int
        # contains a description of the state (null terminated utf-8)*/
        self.desc = None #char*
        # x coordinate position for graph representation plotting **/
        self.xPosition = 0 #int
        # y coordinate position for graph representation plotting **/
        self.yPosition = 0 #int


    def forward_step(self, alpha_t, b_symb):
        value = 0.0

        if b_symb < GHMM_EPS_PREC:
            return 0

        for i in range(self.in_states):
            id = self.in_id[i]
            value += self.in_a[i] * alpha_t[id]

        value *= b_symb
        return value


def model_state_alloc(
    M,
    in_states,
    out_states
):
    s = ghmm_dstate()
    s.b = [0.0] * M
    s.out_id = [0] * out_states
    s.out_a = [0.0] * out_states

    s.in_id = [0] * in_states
    s.in_a = [0.0] * in_states

    return s


