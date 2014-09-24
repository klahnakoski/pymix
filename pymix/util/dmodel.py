from math import log
from util.ghmmwrapper import kHigherOrderEmissions, model_state_alloc, kSilentStates, kUntied, kTiedEmissions, kNoBackgroundDistribution, kBackgroundDistributions, kLabeledStates, ghmm_dseq, RNG, GHMM_RNG_SET, GHMM_MAX_SEQ_LEN, GHMM_RNG_UNIFORM, GHMM_EPS_PREC, ARRAY_REALLOC
from vendor.pyLibrary.env.logs import Log


class ghmm_dmodel():
    def __init__(self, N, M, model_type=0, inDegVec=None, outDegVec=None):
        # assert (modeltype & kDiscreteHMM)

        # Number of states */
        self.N = N  # int

        # Number of outputs */
        self.M = M  # int

        # Vector of the states */
        if inDegVec:
            self.s = [model_state_alloc(M, i, o) for i, o in zip(inDegVec, outDegVec)]
        else:
            self.s = None  # ghmm_dstate*

        # The a priori probability for the model.
        # A value of -1 indicates that no prior is defined.
        # Note: this is not to be confused with priors on emission
        # distributions*/
        self.prior = 0.0 #double

        # contains a arbitrary name for the model (None terminated utf-8) */
        self.name = "" #char*

        # Contains bit flags for varios model extensions such as
        # kSilentStates, kTiedEmissions (see ghmm.h for a complete list)
        # */
        self.model_type = model_type #int

        # Flag variables for each state indicating whether it is emitting
        # or not.
        # Note: silent != None iff (model_type & kSilentStates) == 1  */

        if self.model_type & kSilentStates:
            self.silent = [0] * N
        else:
            self.silent = None

        #AS*/
        # Int variable for the maximum level of higher order emissions */
        self.maxorder = 0 #int
        # saves the history of emissions as int,
        # the nth-last emission is (emission_history * |alphabet|^n+1) % |alphabet|
        # see ...*/
        self.emission_history = 0 #int

        # Flag variables for each state indicating whether the states emissions
        # are tied to another state. Groups of tied states are represented
        # by their tie group leader (the lowest numbered member of the group).
        #
        # tied_to[s] == kUntied  : s is not a tied state
        #
        # tied_to[s] == s        : s is a tie group leader
        #
        # tied_to[t] == s        : t is tied to state s (t>s)
        #
        # Note: tied_to != None iff (model_type & kTiedEmissions) != 0  */
        if self.model_type & kTiedEmissions:
            self.tied_to = [kUntied] * N
        else:
            self.tied_to = None

        # Note: State store order information of the emissions.
        # Classic HMMS have emission order 0, that is the emission probability
        # is conditioned only on the state emitting the symbol.
        #
        # For higher order emissions, the emission are conditioned on the state s
        # as well as the previous emission_order[s] observed symbols.
        #
        # The emissions are stored in the state's usual double* b. The order is
        # set order.
        #
        # Note: order != None iff (model_type & kHigherOrderEmissions) != 0  */
        if self.model_type & kHigherOrderEmissions:
            self.order = [0] * N
        else:
            self.order = None #int*

        # ghmm_dbackground is a pointer to a
        # ghmm_dbackground structure, which holds (essentially) an
        # array of background distributions (which are just vectors of floating
        # point numbers like state.b).
        #
        # For each state the array background_id indicates which of the background
        # distributions to use in parameter estimation. A value of kNoBackgroundDistribution
        # indicates that none should be used.
        #
        #
        # Note: background_id != None iff (model_type & kHasBackgroundDistributions) != 0  */
        if self.model_type & kBackgroundDistributions:
            self.background_id = [kNoBackgroundDistribution] * N
        else:
            self.background_id = None #int*

        self.bp = None #ghmm_dbackground*

        # (WR) added these variables for topological ordering of silent states
        # Condition: topo_order != None iff (model_type & kSilentStates) != 0  */
        self.topo_order = None #int*
        self.topo_order_length = 0 #int

        # Store for each state a class label. Limits the possibly state sequence
        #
        # Note: label != None iff (model_type & kLabeledStates) != 0  */
        if self.model_type & kLabeledStates:
            self.label = [0] * N
        else:
            self.label = None

        self.label_alphabet = None #ghmm_alphabet*

        self.alphabet = None #ghmm_alphabet*

    def getStateName(self, index):
        try:
            return self.s[index].desc
        except Exception:
            return None

    def setStateName(self, index, name):
        try:
            self.s[index].desc = name
        except Exception:
            return None

    def generate_sequences(
        self,
        seed,
        global_len,
        seq_number,
        Tmax
    ):
    #define CUR_PROC "generate_sequences"

        sq = ghmm_dseq()
        len = global_len
        n = 0

        # allocating additional fields for the state sequence in the ghmm_dseq struct */
        sq.seq_number = seq_number
        sq.seq = [None] * seq_number
        sq.seq_len = [None] * seq_number
        sq.states = [None] * seq_number
        sq.states_len = [0] * seq_number

        # A specific length of the sequences isn't given. As a model should have
        #    an end state, the konstant MAX_SEQ_LEN is used. */
        if len <= 0:
            len = GHMM_MAX_SEQ_LEN

        if seed > 0:
            GHMM_RNG_SET(RNG, seed)


        # initialize the emission history */
        self.emission_history = 0

        while n < seq_number:
            sq.seq[n] = [0] * ( len)

            # for silent models we have to allocate for the maximal possible number
            #    of lables and states */
            if self.model_type & kSilentStates:
                sq.states[n] = [0] * ( len * self.N)
            else:
                sq.states[n] = [0] * ( len)

            pos = label_pos = 0

            # Get a random initial state i */
            p = GHMM_RNG_UNIFORM(RNG)
            sum = 0.0
            for state in range(self.N):
                sum += self.s[state].pi
                if sum >= p:
                    break

            while pos < len:
                # save the state path and label */
                sq.states[n][label_pos] = state
                label_pos += 1

                # Get a random output m if the state is not a silent state */
                if not (self.model_type & kSilentStates) or not (self.silent[state]):
                    m = self.get_random_output(state, pos)
                    self.update_emission_history(m)
                    sq.seq[n][pos] = m
                    pos += 1


                # get next state */
                p = GHMM_RNG_UNIFORM(RNG)
                if pos < self.maxorder:
                    max_sum = 0.0
                    for j in range(self.s[state].out_states):
                        j_id = self.s[state].out_id[j]
                        if not (self.model_type & kHigherOrderEmissions) or self.order[j_id] <= pos:
                            max_sum += self.s[state].out_a[j]

                    if j and abs(max_sum) < GHMM_EPS_PREC:
                        Log.error("No possible transition from state %d "
                                  "since all successor states require more history "
                                  "than seen up to position: %d.",
                            state, pos)
                        break

                    if j:
                        p *= max_sum

                sum = 0.0
                for j in range(self.s[state].out_states):
                    j_id = self.s[state].out_id[j]
                    if not (self.model_type & kHigherOrderEmissions) or self.order[j_id] <= pos:
                        sum += self.s[state].out_a[j]
                        if sum >= p:
                            break

                if sum == 0.0:
                    Log.note("final state (%d) reached at position %d of sequence %d", state, pos, n)
                    break

                state = j_id
                # while (pos < len) */

            # realocate state path and label sequence to actual size */
            if self.model_type & kSilentStates:
                ARRAY_REALLOC(sq.states[n], label_pos)

            sq.seq_len[n] = pos
            sq.states_len[n] = label_pos
            n += 1
            # while( n < seq_number ) */

        return sq

    def get_random_output(self, i, position):
        #define CUR_PROC "get_random_output"
        sum = 0.0

        p = GHMM_RNG_UNIFORM(RNG)

        for m in range(self.M):
            # get the right index for higher order emission models */
            e_index = self.get_emission_index(i, m, position)

            # get the probability, exit, if the index is -1 */
            if -1 != e_index:
                sum += self.s[i].b[e_index]
                if sum >= p:
                    break
            else:
                Log.error("ERROR: State has order %d, but in the history are only %d emissions.\n", self.order[i], position)
                return -1
        else:
            Log.error("ERROR: no valid output choosen. Are the Probabilities correct? sum: %g, p: %g\n", sum, p)
            return -1

        return m

    def get_emission_index(self, S, O, T):
        if self.model_type & kHigherOrderEmissions:
            if self.order[S] > T:
                return -1
            else:
                return (self.emission_history * self.M) % pow(self.M, self.order[S] + 1) + O
        else:
            return O

    def update_emission_history(self, O):
        if self.model_type & kHigherOrderEmissions:
            self.emission_history = self.emission_history * self.M % pow(self.M, self.maxorder) + O


    def logp(self, O, len, log_p):
        scale = [0] * len
        alpha = [[0] * self.N] * len
        self.forward(O, len, alpha, scale, log_p)


    def forward_init(self, alpha_1, symb, scale):
        scale[0] = 0.0

        #printf(" *** foba_initforward\n")*/

        #iterate over non-silent states*/
        #printf(" *** iterate over non-silent states \n")*/
        for i in range(self.N):
            if not (self.model_type & kSilentStates) or not (self.silent[i]):
                #no starting in states with order > 0 not not not */
                if not (self.model_type & kHigherOrderEmissions) or self.order[i] == 0:
                    alpha_1[i] = self.s[i].pi * self.s[i].b[symb]
                    scale[0] += alpha_1[i]
                else:
                    alpha_1[i] = 0

        #iterate over silent states*/
        #printf(" *** iterate over silent states \n")*/
        if self.model_type & kSilentStates:
            for i in range(self.topo_order_length):
                id = self.topo_order[i]
                alpha_1[id] = self.s[id].pi

                #printf("\nsilent_start alpha1[%i]=%f\n",id,alpha_1[id])*/

                for j in range(self.s[id].in_states):
                    in_id = self.s[id].in_id[j]
                    alpha_1[id] += self.s[id].in_a[j] * alpha_1[in_id]

                    #printf("\n\tsilent_run alpha1[%i]=%f\n",id,alpha_1[id])*/

                scale[0] += alpha_1[id]

        if scale[0] >= GHMM_EPS_PREC:
            c_0 = 1 / scale[0]
            for i in range(self.N):
                alpha_1[i] *= c_0


    def forward_step(s, alpha_t, b_symb):
        value = 0.0

        if b_symb < GHMM_EPS_PREC:
            return 0.

        #printf(" *** foba_stepforward\n")*/

        for i in range(s.in_states):
            id = s.in_id[i]
            value += s.in_a[i] * alpha_t[id]
            #printf("    state %d, value %f, p_symb %f\n",id, value, b_symb) */

        value *= b_symb
        return (value)


    def forward(self, O, len, alpha, scale):
        log_scale_sum = 0.0
        non_silent_salpha_sum = 0.0

        if self.model_type & kSilentStates:
            self.order_topological()

        self.forward_init(alpha[0], O[0], scale)

        if scale[0] < GHMM_EPS_PREC:
            Log.error("first symbol can't be generated by hmm")

        log_p = -log(1 / scale[0])
        for t in range(1, len):

            scale[t] = 0.0
            self.update_emission_history(O[t - 1])

            # printf("\n\nStep t=%i mit len=%i, O[i]=%i\n",t,len,O[t])
            #    printf("iterate over non-silent state\n") */
            # iterate over non-silent states */
            for i in range(self.N):
                if not (self.model_type & kSilentStates) or not (self.silent[i]):
                    e_index = self.get_emission_index(i, O[t], t)
                    if e_index != -1:
                        alpha[t][i] = self.s[i].forward_step(alpha[t - 1], self.s[i].b[e_index])
                        scale[t] += alpha[t][i]

                    else:
                        alpha[t][i] = 0

            # iterate over silent states */
            # printf("iterate over silent state\n") */
            if self.model_type & kSilentStates:
                for i in range(self.topo_order_length):
                    #printf("\nget id\n")*/
                    id = self.topo_order[i]
                    #printf("  akt_ state %d\n",id)*/
                    #printf("\nin stepforward\n")*/
                    alpha[t][id] = self.s[id].forward_step(alpha[t], 1)
                    #printf("\nnach stepforward\n")*/
                    scale[t] += alpha[t][id]

            if scale[t] < GHMM_EPS_PREC:
                Log.error("scale smaller than epsilon (%g < %g) in position %d. Can't generate symbol %d\n", scale[t], GHMM_EPS_PREC, t, O[t])

            c_t = 1 / scale[t]
            for i in range(self.N):
                alpha[t][i] *= c_t

            if not (self.model_type & kSilentStates):
                # sum log(c[t]) scaling values to get  log( P(O|lambda) ) */

                #printf("log_p %f -= log(%f) = ",log_p,c_t)*/
                log_p -= log(c_t)
                #printf(" %f\n",log_p) */

        if self.model_type & kSilentStates:
            #printf("silent model\n")*/
            for i in range(len):
                log_scale_sum += log(scale[i])

            for i in range(self.N):
                if not (self.silent[i]):
                    non_silent_salpha_sum += alpha[len - 1][i]

            salpha_log = log(non_silent_salpha_sum)
            log_p = log_scale_sum + salpha_log

        return log_p


    def forward_descale(alpha, scale, t, n, newalpha):
        for i in range(t):
            for j in range(n):
                newalpha[i][j] = alpha[i][j]
                for k in range(i + 1):
                    newalpha[i][j] *= scale[k]


    def backward(self, O, len, beta, scale):
        # beta_tmp holds beta-variables for silent states */
        beta_tmp = None

        for t in range(len):
            if scale[t] != 0:
                Log.error()

        # topological ordering for models with silent states and allocating
        #    temporary array needed for silent states */
        if self.model_type & kSilentStates:
            beta_tmp = [0] * self.N
            self.order_topological()


        # initialize all states */
        for i in range(self.N):
            beta[len - 1][i] = 1.0

        if not (self.model_type & kHigherOrderEmissions):
            self.maxorder = 0

        # initialize emission history */
        for t in range(len - self.maxorder, len):
            self.update_emission_history(O[t])


        # Backward Step for t = T-1, ..., 0 */
        # loop over reverse topological ordering of silent states, non-silent states  */
        for t in range(0, len - 1, -1):  #for (t = len - 2 t >= 0 t--) :
            # printf(" ----------- *** t = %d ***  ---------- \n",t) */
            # printf("\n*** O(%d) = %d\n",t+1,O[t+1]) */

            # updating of emission_history with O[t] such that emission_history memorizes
            #    O[t - maxorder ... t] */
            if 0 <= t - self.maxorder + 1:
                self.update_emission_history_front(O[t - self.maxorder + 1])

            # iterating over the the silent states and filling beta_tmp */
            if self.model_type & kSilentStates:
                for k in range(0, self.topo_order_length, -1):#for (k = self.topo_order_length - 1 k >= 0 k--) :
                    id = self.topo_order[k]
                    # printf("  silent[%d] = %d\n",id,self.silent[id]) */
                    assert (self.silent[id] == 1)

                    sum = 0.0
                    for j in range(self.s[id].out_states):
                        j_id = self.s[id].out_id[j]

                        # out_state is not silent */
                        if not self.silent[j_id]:
                            e_index = self.get_emission_index(j_id, O[t + 1], t + 1)
                            if e_index != -1:
                                sum += self.s[id].out_a[j] * self.s[j_id].b[e_index] * beta[t + 1][j_id]


                        # out_state is silent, beta_tmp[j_id] is useful since we go through
                        #    the silent states in reversed topological order */
                        else:
                            sum += self.s[id].out_a[j] * beta_tmp[j_id]


                    # setting beta_tmp for the silent state
                    #    don't scale the betas for silent states now
                    #    wait until the betas for non-silent states are complete to avoid
                    #    multiple scaling with the same scalingfactor in one term */
                    beta_tmp[id] = sum



            # iterating over the the non-silent states */
            for i in range(self.N):
                if not (self.model_type & kSilentStates) or not (self.silent[i]):
                    sum = 0.0

                    for j in range(self.s[i].out_states):
                        j_id = self.s[i].out_id[j]

                        # out_state is not silent: get the emission probability
                        #    and use beta[t+1]
                    if not (self.model_type & kSilentStates) or not (self.silent[j_id]):
                        e_index = self.get_emission_index(j_id, O[t + 1], t + 1)
                        if e_index != -1:
                            emission = self.s[j_id].b[e_index]
                        else:
                            emission = 0
                        sum += self.s[i].out_a[j] * emission * beta[t + 1][j_id]

                        # out_state is silent: use beta_tmp */
                    else:
                        sum += self.s[i].out_a[j] * beta_tmp[j_id]


                    # updating beta[t] for non-silent state */
                    beta[t][i] = sum / scale[t + 1]


            # updating beta[t] for silent states, finally scale them
            #    and resetting beta_tmp */
            if self.model_type & kSilentStates:
                for i in range(self.N):
                    if self.silent[i]:
                        beta[t][i] = beta_tmp[i] / scale[t + 1]
                        beta_tmp[i] = 0.0

    def backward_termination(self, O, length, beta, scale):
    #define CUR_PROC "ghmm_dmodel_backward_termination"
        beta_tmp = None

        # topological ordering for models with silent states and precomputing
        #    the beta_tmp for silent states */
        if self.model_type & kSilentStates:
            self.order_topological()

            beta_tmp = [0] * self.N
            for k in range(0, self.topo_order_length, -1):#for (k = self.topo_order_length - 1 k >= 0 k--) :
                id = self.topo_order[k]
                assert (self.silent[id] == 1)
                sum = 0.0

                for j in range(self.s[id].out_states):
                    j_id = self.s[id].out_id[j]

                    # out_state is not silent */
                    if not self.silent[j_id]:
                        # no emission history for the first symbol */
                        if not (self.model_type & kHigherOrderEmissions) or self.order[id] == 0:
                            sum += self.s[id].out_a[j] * self.s[j_id].b[O[0]] * beta[0][j_id]


                    # out_state is silent, beta_tmp[j_id] is useful since we go through
                    #    the silent states in reversed topological order */
                    else:
                        sum += self.s[id].out_a[j] * beta_tmp[j_id]


                # setting beta_tmp for the silent state
                #    don't scale the betas for silent states now */
                beta_tmp[id] = sum

        sum = 0.0
        # iterating over all states with pi > 0.0 */
        for i in range(self.N):
            if self.s[i].pi > 0.0:
                # silent states */
                if (self.model_type & kSilentStates) and self.silent[i]:
                    sum += self.s[i].pi * beta_tmp[i]

                # non-silent states */
                else:
                    # no emission history for the first symbol */
                    if not (self.model_type & kHigherOrderEmissions) or self.order[i] == 0:
                        sum += self.s[i].pi * self.s[i].b[O[0]] * beta[0][i]

        log_p = log(sum / scale[0])

        log_scale_sum = 0.0
        for i in range(length):
            log_scale_sum += log(scale[i])

        log_p += log_scale_sum

        return log_p

    def logp(self, O, len):
        alpha = [[0] * self.N] * len
        scale = [0] * len

        # run ghmm_dmodel_forward */
        log_p = self.forward(O, len, alpha, scale)
        return log_p


    def logp_joint(self, O, len, S, slen):
    # define CUR_PROC "ghmm_dmodel_logp_joint"
        state_pos = 0
        pos = 0

        prevstate = state = S[0]
        log_p = log(self.s[state].pi)
        if not (self.model_type & kSilentStates) or not self.silent[state]:
            log_p += log(self.s[state].b[O[pos]])
            pos += 1

        for state_pos in range(1, slen):
            if pos >= len:
                break
            state = S[state_pos]
            for j in range(0, self.s[state].in_states):
                if prevstate == self.s[state].in_id[j]:
                    break

            if (j == self.s[state].in_states or abs(self.s[state].in_a[j]) < GHMM_EPS_PREC):
                Log.error("Sequence can't be built. There is no transition from state %d to %d.", prevstate, state)

            log_p += log(self.s[state].in_a[j])

            if not (self.model_type & kSilentStates) or not self.silent[state]:
                log_p += log(self.s[state].b[O[pos]])
                pos += 1

            prevstate = state

        if (pos < len):
            Log.note("state sequence too shortnot  processed only %d symbols", pos)
        if (state_pos < slen):
            Log.note("sequence too shortnot  visited only %d states", state_pos)

        return log_p


    def forward_lean(self, O, len):
        log_scale_sum = 0.0
        non_silent_salpha_sum = 0.0

        alpha_last_col = [0] * self.N
        alpha_curr_col = [0] * self.N
        scale = [0] * len

        if (self.model_type & kSilentStates):
            self.order_topological()

        self.forward_init(alpha_last_col, O[0], scale)
        if scale[0] < GHMM_EPS_PREC:
            Log.error("first symbol can't be generated by hmm")

        log_p = -log(1 / scale[0])

        for t in range(1, len):
            scale[t] = 0.0
            self.update_emission_history(O[t - 1])

            # iterate over non-silent states */
            for i in range(self.N):
                if not (self.model_type & kSilentStates) or not (self.silent[i]):
                    e_index = self.get_emission_index(i, O[t], t)
                    if e_index != -1:
                        alpha_curr_col[i] = self.s[i].forward_step(alpha_last_col, self.s[i].b[e_index])
                        scale[t] += alpha_curr_col[i]

                    else:
                        alpha_curr_col[i] = 0


            # iterate over silent states  */
            if self.model_type & kSilentStates:
                for i in range(self.topo_order_length):
                    id = self.topo_order[i]
                    alpha_curr_col[id] = self.s[id].forward_step(alpha_curr_col, 1)
                    scale[t] += alpha_curr_col[id]

            if scale[t] < GHMM_EPS_PREC:
                Log.error("O-string  can't be generated by hmm")

            c_t = 1 / scale[t]
            for i in range(self.N):
                alpha_curr_col[i] *= c_t

            if not (self.model_type & kSilentStates):
                #sum log(c[t]) scaling values to get  log( P(O|lambda) ) */
                log_p -= log(c_t)


            # switching pointers of alpha_curr_col and alpha_last_col
            #    don't set alpha_curr_col[i] to zero since its overwritten */
            switching_tmp = alpha_last_col
            alpha_last_col = alpha_curr_col
            alpha_curr_col = switching_tmp


        # Termination step: compute log likelihood */
        if self.model_type & kSilentStates:
            #printf("silent model\n")*/
            for i in range(len):
                log_scale_sum += log(scale[i])

            for i in range(self.N):
                # use alpha_last_col since the columms are also in the last step
                #    switched */
                if (not (self.silent[i])):
                    non_silent_salpha_sum += alpha_last_col[i]

            salpha_log = log(non_silent_salpha_sum)
            log_p = log_scale_sum + salpha_log

        return log_p


    def foba_label_initforward(self, alpha_1, symb, label, scale):
    # define CUR_PROC "foba_label_initforward"
        scale[0] = 0.0

        # iterate over non-silent states */
        for i in range(self.N):
            if not (self.model_type & kSilentStates) or not (self.silent[i]):
                if self.label[i] == label:
                    alpha_1[i] = self.s[i].pi * self.s[i].b[symb]
                else:
                    alpha_1[i] = 0.0
            else:
                alpha_1[i] = 0.0

            scale[0] += alpha_1[i]

        if scale[0] >= GHMM_EPS_PREC:
            c_0 = 1 / scale[0]
            for i in range(self.N):
                alpha_1[i] *= c_0

    def label_forward(self, O, label, len, alpha, scale):
    # define CUR_PROC "ghmm_dl_forward"

        self.foba_label_initforward(alpha[0], O[0], label[0], scale)
        if scale[0] < GHMM_EPS_PREC:
            Log.error("means: first symbol can't be generated by hmm")
        else:
            log_p = -log(1 / scale[0])

            for t in range(1, len):

                self.update_emission_history(O[t - 1])
                scale[t] = 0.0

                # printf("\n\nStep t=%i mit len=%i, O[i]=%i\n",t,len,O[t]) */

                for i in range(self.N):
                    if not (self.model_type & kSilentStates) or not (self.silent[i]):
                        if self.label[i] == label[t]:
                        #printf("%d: akt_ state %d, label: %d \t current Label: %d\n",
                        # t, i, self.label[i], label[t])*/
                            e_index = self.get_emission_index(i, O[t], t)
                            if -1 != e_index:
                                alpha[t][i] = self.s[i].forward_step(alpha[t - 1], self.s[i].b[e_index])
                                #if alpha[t][i] < GHMM_EPS_PREC:
                                #   printf("alpha[%d][%d] = %g \t ", t, i, alpha[t][i])
                                #   printf("self.s[%d].b[%d] = %g\n", i, e_index, self.s[i].b[e_index])
                                #
                                # else printf("alpha[%d][%d] = %g\n", t, i, alpha[t][i])*/

                            else:
                                alpha[t][i] = 0


                        else:
                            alpha[t][i] = 0

                        scale[t] += alpha[t][i]

                    else:
                        Log.error("ERROR: Silent state in foba_label_forward.\n")

                if scale[t] < GHMM_EPS_PREC:
                    if t > 4:
                        Log.note("%g\t%g\t%g\t%g\t%g\n", scale[t - 5], scale[t - 4], scale[t - 3], scale[t - 2], scale[t - 1])
                        Log.error("scale = %g smaller than eps = EPS_PREC in the %d-th char.\ncannot generate emission: %d with label: %d in sequence of length %d\n", scale[t], t, O[t], label[t], len)

                c_t = 1 / scale[t]
                for i in range(self.N):
                    alpha[t][i] *= c_t

                if not (self.model_type & kSilentStates):
                    log_p -= log(c_t)

        return log_p

    def label_logp(self, O, label, len):
    # define CUR_PROC "ghmm_dl_logp"
        scale = [0] * len
        alpha = [[0] * self.N] * len

        # run ghmm_dmodel_forward */
        log_p = self.label_forward(O, label, len, alpha, scale)
        return log_p

    def label_backward(self, O, label, len, beta, scale):
        beta_tmp = [0] * self.N
        for t in range(len):
            if scale[t] != 0:
                Log.error()

        # check for silent states */
        if self.model_type & kSilentStates:
            Log.error("ERROR: No silent states allowed in labelled HMMnot \n");

        # initialize */
        for i in range(self.N):
            # start only in states with the correct label */
            if (label[len - 1] == self.label[i]):
                beta[len - 1][i] = 1.0
            else:
                beta[len - 1][i] = 0.0

            beta_tmp[i] = beta[len - 1][i] / scale[len - 1]


        # initialize emission history */
        if (not (self.model_type & kHigherOrderEmissions)):
            self.maxorder = 0
        for t in range(len - (self.maxorder), len):
            self.update_emission_history(O[t])


        # Backward Step for t = T-1, ..., 0
        #    beta_tmp: Vector for storage of scaled beta in one time step
        #    loop over reverse topological ordering of silent states, non-silent states */
        for t in range(0, len - 1, -1): #for (t = len - 2 t >= 0 t--) :

            # updating of emission_history with O[t] such that emission_history
            #    memorizes O[t - maxorder ... t] */
            if (0 <= t - self.maxorder + 1):
                self.update_emission_history_front(O[t - self.maxorder + 1])

            for i in range(self.N):
                sum = 0.0
                for j in range(self.s[i].out_states):
                    j_id = self.s[i].out_id[j]
                    # The state has only a emission with probability > 0, if the label matches */
                    if label[t] == self.label[i]:
                        e_index = self.get_emission_index(j_id, O[t + 1], t + 1)
                        if (e_index != -1):
                            emission = self.s[j_id].b[e_index]
                        else:
                            emission = 0.0

                    else:
                        emission = 0.0

                    sum += self.s[i].out_a[j] * emission * beta_tmp[j_id]

                beta[t][i] = sum
                # if ((beta[t][i] > 0) and ((beta[t][i] < .01) or (beta[t][i] > 100))) beta_out++ */

            for i in range(self.N):
                beta_tmp[i] = beta[t][i] / scale[t]


    def ghmm_dl_forward_lean(self, O, label, len):

        log_scale_sum = 0.0
        non_silent_salpha_sum = 0.0
        salpha_log = 0.0

        alpha_last_col = [0] * self.N
        alpha_curr_col = [0] * self.N
        switching_tmp = None
        scale = [0] * len

        if (self.model_type & kSilentStates):
            self.order_topological()

        self.foba_label_initforward(alpha_last_col, O[0], label[0], scale)
        if scale[0] < GHMM_EPS_PREC:
            Log.error("first symbol can't be generated by hmm")

        log_p = -log(1 / scale[0])

        for t in range(1, len):
            scale[t] = 0.0

            # iterate over non-silent states */
            for i in range(self.N):
                if not (self.model_type & kSilentStates) or not (self.silent[i]):

                    # printf("  akt_ state %d\n",i)*/
                    if self.label[i] == label[t]:
                        e_index = self.get_emission_index(i, O[t], t)
                        if e_index != -1:
                            alpha_curr_col[i] = self.s[i].forward_step(alpha_last_col, self.s[i].b[e_index])
                            scale[t] += alpha_curr_col[i]

                        else:
                            alpha_curr_col[i] = 0

                    else:
                        alpha_curr_col[i] = 0


            # iterate over silent states  */
            if self.model_type & kSilentStates:
                for i in range(self.topo_order_length):
                    id = self.topo_order[i]
                    alpha_curr_col[id] = self.s[id].forward_step(alpha_last_col, 1)
                    scale[t] += alpha_curr_col[id]

            if scale[t] < GHMM_EPS_PREC:
                Log.error("scale smaller than epsilon\n")

            c_t = 1 / scale[t]
            for i in range(self.N):
                alpha_curr_col[i] *= c_t

            if not (self.model_type & kSilentStates):
                #sum log(c[t]) scaling values to get  log( P(O|lambda) ) */
                log_p -= log(c_t)


            # switching pointers of alpha_curr_col and alpha_last_col
            #    don't set alpha_curr_col[i] to zero since its overwritten */
            switching_tmp = alpha_last_col
            alpha_last_col = alpha_curr_col
            alpha_curr_col = switching_tmp


        # Termination step: compute log likelihood */
        if self.model_type & kSilentStates:
            #printf("silent model\n")*/
            for i in range(len):
                log_scale_sum += log(scale[i])

            for i in range(self.N):
                if (not (self.silent[i])):
                    non_silent_salpha_sum += alpha_curr_col[i]

            salpha_log = log(non_silent_salpha_sum)
            log_p = log_scale_sum + salpha_log

        return log_p
