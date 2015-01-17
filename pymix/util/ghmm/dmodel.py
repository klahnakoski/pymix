from pyLibrary.maths import Math
from pymix.util.ghmm import random_mt
from pymix.util.ghmm.sequences import sequence
from pymix.util.ghmm.dstate import model_state_alloc
from pymix.util.ghmm.local_store_t import reestimate_alloc
from pymix.util.ghmm.reestimate import ighmm_reestimate_alloc_matvek, nologSum
from pymix.util.ghmm.topological_sort import topological_sort
from pymix.util.ghmm.types import kHigherOrderEmissions, kSilentStates, kUntied, kTiedEmissions, kNoBackgroundDistribution, kBackgroundDistributions, kLabeledStates
from pymix.util.ghmm.wrapper import GHMM_MAX_SEQ_LEN, GHMM_EPS_PREC, ARRAY_REALLOC, double_matrix_alloc, double_array_alloc, ARRAY_CALLOC, ARRAY_MALLOC, MAX_ITER_BW, EPS_ITER_BW, ighmm_cvector_normalize
from pymix.util.logs import Log


class ghmm_dmodel():
    def __init__(self, N, M, model_type=0, inDegVec=None, outDegVec=None):
        # assert (modeltype & kDiscreteHMM)

        # Number of states
        self.N = N  # int

        # Number of outputs
        self.M = M  # int

        # Vector of the states
        if inDegVec:
            self.s = [model_state_alloc(M, N) for _ in range(N)]
        else:
            self.s = None  # ghmm_dstate*

        # The a priori probability for the model.
        # A value of -1 indicates that no prior is defined.
        # Note: this is not to be confused with priors on emission
        # distributions
        self.prior = 0.0 #double

        # contains a arbitrary name for the model (None terminated utf-8)
        self.name = "" #char*

        # Contains bit flags for varios model extensions such as
        # kSilentStates, kTiedEmissions (see ghmm.h for a complete list)
        #
        self.model_type = model_type #int

        # Flag variables for each state indicating whether it is emitting
        # or not.
        # Note: silent != None iff (model_type & kSilentStates) == 1

        if self.model_type & kSilentStates:
            self.silent = double_array_alloc(N)
        else:
            self.silent = None

        #AS
        # Int variable for the maximum level of higher order emissions
        self.maxorder = 0 #int
        # saves the history of emissions as int,
        # the nth-last emission is (emission_history * |alphabet|^n+1) % |alphabet|
        # see ...
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
        # Note: tied_to != None iff (model_type & kTiedEmissions) != 0
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
        # Note: order != None iff (model_type & kHigherOrderEmissions) != 0
        if self.model_type & kHigherOrderEmissions:
            self.order = double_array_alloc(N)
        else:
            self.order = None  # int*

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
        # Note: background_id != None iff (model_type & kHasBackgroundDistributions) != 0
        if self.model_type & kBackgroundDistributions:
            self.background_id = [kNoBackgroundDistribution] * N
        else:
            self.background_id = None #int*

        self.bp = None #ghmm_dbackground*

        # (WR) added these variables for topological ordering of silent states
        # Condition: topo_order != None iff (model_type & kSilentStates) != 0
        self.topo_order = None #int*
        self.topo_order_length = 0 #int

        # Store for each state a class label. Limits the possibly state sequence
        #
        # Note: label != None iff (model_type & kLabeledStates) != 0
        if self.model_type & kLabeledStates:
            self.label = double_array_alloc(N)
        else:
            self.label = None

        self.label_alphabet = None

        self.alphabet = None


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
        Tmax,
        native=False
    ):


        sq = sequence([[]] * seq_number)
        n = 0

        # A specific length of the sequences isn't given. As a model should have
        # an end state, the konstant MAX_SEQ_LEN is used.
        if global_len <= 0:
            global_len = GHMM_MAX_SEQ_LEN

        if seed > 0:
            random_mt.set_seed( seed)


        # initialize the emission history
        self.emission_history = 0

        while n < seq_number:
            sq.seq[n] = double_array_alloc(( global_len))

            # for silent models we have to allocate for the maximal possible number
            # of lables and states
            if self.model_type & kSilentStates:
                sq.states[n] = double_array_alloc(global_len * self.N)
            else:
                sq.states[n] = double_array_alloc(global_len)

            pos = label_pos = 0

            # Get a random initial state i
            p = random_mt.float23()
            sum_ = 0.0
            for state in range(self.N):
                sum_ += self.s[state].pi
                if sum_ >= p:
                    break
            else:
                state = self.N

            while pos < global_len:
                # save the state path and label
                sq.states[n][label_pos] = state
                label_pos += 1

                # Get a random output m if the state is not a silent state
                if not (self.model_type & kSilentStates) or not (self.silent[state]):
                    m = self.get_random_output(state, pos)
                    self.update_emission_history(m)
                    sq.seq[n][pos] = m
                    pos += 1


                # get next state
                p = random_mt.float23()
                if pos < self.maxorder:
                    max_sum = 0.0
                    for j in range(self.N):
                        if not (self.model_type & kHigherOrderEmissions) or self.order[j] <= pos:
                            max_sum += self.s[state].out_a[j]
                    else:
                        j = self.N

                    if j and abs(max_sum) < GHMM_EPS_PREC:
                        Log.error("No possible transition from state %d "
                                  "since all successor states require more history "
                                  "than seen up to position: %d.",
                            state, pos)

                    if j:
                        p *= max_sum

                sum_ = 0.0
                for j in range(self.N):
                    if not (self.model_type & kHigherOrderEmissions) or self.order[j] <= pos:
                        sum_ += self.s[state].out_a[j]
                        if sum_ >= p:
                            break
                else:
                    j = self.N
                if sum_ == 0.0:
                    Log.note("final state (%d) reached at position %d of sequence %d", state, pos, n)
                    break

                state = j
                # while (pos < len)

            # realocate state path and label sequence to actual size
            if self.model_type & kSilentStates:
                sq.states[n] = ARRAY_REALLOC(sq.states[n], label_pos)

            sq.seq_len[n] = pos
            sq.states_len[n] = label_pos
            n += 1
            # while( n < seq_number )

        return sq

    def get_random_output(self, i, position):
        #define CUR_PROC "get_random_output"
        sum_ = 0.0

        p = random_mt.float23()

        for m in range(self.M):
            # get the right index for higher order emission models
            e_index = self.get_emission_index(i, m, position)

            # get the probability, exit, if the index is -1
            if -1 != e_index:
                sum_ += self.s[i].b[e_index]
                if sum_ >= p:
                    break
            else:
                Log.error("ERROR: State has order %d, but in the history are only %d emissions.\n", self.order[i], position)
        else:
            Log.error("ERROR: no valid output choosen. Are the Probabilities correct? sum: %g, p: %g\n", sum_, p)

        return m

    def get_emission_index(self, S, O, T):
        if self.model_type & kHigherOrderEmissions:
            if self.order[S] > T:
                return -1
            else:
                # THIS MAKING A REF INTO MULTIDIMENSIONAL CUBE
                output = (self.emission_history * self.M) % pow(self.M, self.order[S] + 1) + O
                return output
        else:
            return O

    def update_emission_history(self, O):
        if self.model_type & kHigherOrderEmissions:
            self.emission_history = (self.emission_history * self.M) % pow(self.M, self.maxorder) + O


    def logp(self, O, len):
        scale = double_array_alloc(len)
        alpha = double_matrix_alloc(len, self.N)

        # run ghmm_dmodel_forward
        log_p = self.forward(O, len, alpha, scale)
        return log_p


    def forward_init(self, alpha_1, symb, scale):
        scale[0] = 0.0

        #printf(" *** foba_initforward\n")

        #iterate over non-silent states
        #printf(" *** iterate over non-silent states \n")
        for i in range(self.N):
            if not (self.model_type & kSilentStates) or not (self.silent[i]):
                #no starting in states with order > 0 not not not
                if not (self.model_type & kHigherOrderEmissions) or self.order[i] == 0:
                    alpha_1[i] = self.s[i].pi * self.s[i].b[symb]
                    scale[0] += alpha_1[i]
                else:
                    alpha_1[i] = 0


        #iterate over silent states
        #printf(" *** iterate over silent states \n")
        if self.model_type & kSilentStates:
            for i in range(self.topo_order_length):
                id = self.topo_order[i]
                alpha_1[id] = self.s[id].pi

                #printf("\nsilent_start alpha1[%i]=%f\n",id,alpha_1[id])

                for j in range(self.N):
                    alpha_1[id] += self.s[id].in_a[j] * alpha_1[j]

                    #printf("\n\tsilent_run alpha1[%i]=%f\n",id,alpha_1[id])
                scale[0] += alpha_1[id]

        if scale[0] >= GHMM_EPS_PREC:
            c_0 = 1 / scale[0]
            for i in range(self.N):
                alpha_1[i] *= c_0

    def forward(self, O, len, alpha, scale):
        log_scale_sum = 0.0
        non_silent_salpha_sum = 0.0

        if self.model_type & kSilentStates:
            self.order_topological()

        self.forward_init(alpha[0], O[0], scale)

        if scale[0] < GHMM_EPS_PREC:
            Log.error("first symbol can't be generated by hmm")

        log_p = Math.log(scale[0])
        for t in range(1, len):

            scale[t] = 0.0
            self.update_emission_history(O[t - 1])

            # printf("\n\nStep t=%i mit len=%i, O[i]=%i\n",t,len,O[t])
            # printf("iterate over non-silent state\n")
            # iterate over non-silent states
            for i in range(self.N):
                if not (self.model_type & kSilentStates) or not (self.silent[i]):
                    e_index = self.get_emission_index(i, O[t], t)
                    if e_index != -1:
                        alpha[t][i] = self.s[i].forward_step(alpha[t - 1], self.s[i].b[e_index])
                        scale[t] += alpha[t][i]

                    else:
                        alpha[t][i] = 0

            # iterate over silent states
            # printf("iterate over silent state\n")
            if self.model_type & kSilentStates:
                for i in range(self.topo_order_length):
                    #printf("\nget id\n")
                    id = self.topo_order[i]
                    #printf("  akt_ state %d\n",id)
                    #printf("\nin stepforward\n")
                    alpha[t][id] = self.s[id].forward_step(alpha[t], 1)
                    #printf("\nnach stepforward\n")
                    scale[t] += alpha[t][id]

            if scale[t] < GHMM_EPS_PREC:
                Log.error("scale smaller than epsilon (%g < %g) in position %d. Can't generate symbol %d\n", scale[t], GHMM_EPS_PREC, t, O[t])

            c_t = 1 / scale[t]
            for i in range(self.N):
                alpha[t][i] *= c_t

            if not (self.model_type & kSilentStates):
                # sum Math.log(c[t]) scaling values to get  Math.log( P(O|lambda) )

                #printf("log_p %f -= Math.log(%f) = ",log_p,c_t)
                log_p -= Math.log(c_t)
                #printf(" %f\n",log_p)

        if self.model_type & kSilentStates:
            #printf("silent model\n")
            for i in range(len):
                log_scale_sum += Math.log(scale[i])

            for i in range(self.N):
                if not (self.silent[i]):
                    non_silent_salpha_sum += alpha[len - 1][i]

            salpha_log = Math.log(non_silent_salpha_sum)
            log_p = log_scale_sum + salpha_log

        return log_p


    def forward_descale(alpha, scale, t, n, newalpha):
        for i in range(t):
            for j in range(n):
                newalpha[i][j] = alpha[i][j]
                for k in range(i + 1):
                    newalpha[i][j] *= scale[k]

    def backward(self, O, len, beta, scale):
        # beta_tmp holds beta-variables for silent states
        beta_tmp = None

        for t in range(len):
            if scale[t] == 0:
                Log.error("Expecting non-zero scale")

        # topological ordering for models with silent states and allocating
        # temporary array needed for silent states
        if self.model_type & kSilentStates:
            beta_tmp = double_array_alloc(self.N)
            self.order_topological()


        # initialize all states
        for i in range(self.N):
            beta[len - 1][i] = 1.0

        if not (self.model_type & kHigherOrderEmissions):
            self.maxorder = 0

        # initialize emission history
        for t in range(len - self.maxorder, len):
            self.update_emission_history(O[t])

        # Backward Step for t = T-1, ..., 0
        # loop over reverse topological ordering of silent states, non-silent states
        for t in reversed(range(len - 1)):  #for (t = len - 2 t >= 0 t--) :
            # printf(" ----------- *** t = %d ***  ---------- \n",t)
            # printf("\n*** O(%d) = %d\n",t+1,O[t+1])

            # updating of emission_history with O[t] such that emission_history memorizes
            # O[t - maxorder ... t]
            if 0 <= t - self.maxorder + 1:
                self.update_emission_history_front(O[t - self.maxorder + 1])

            # iterating over the the silent states and filling beta_tmp
            if self.model_type & kSilentStates:
                for k in reversed(range(self.topo_order_length)):#for (k = self.topo_order_length - 1 k >= 0 k--) :
                    id = self.topo_order[k]
                    # printf("  silent[%d] = %d\n",id,self.silent[id])
                    assert (self.silent[id] == 1)

                    sum_ = 0.0
                    for j in range(self.N):
                        # out_state is not silent
                        if not self.silent[j]:
                            e_index = self.get_emission_index(j, O[t + 1], t + 1)
                            if e_index != -1:
                                sum_ += self.s[id].out_a[j] * self.s[j].b[e_index] * beta[t + 1][j]


                        # out_state is silent, beta_tmp[j_id] is useful since we go through
                        # the silent states in reversed topological order
                        else:
                            sum_ += self.s[id].out_a[j] * beta_tmp[j]

                    # setting beta_tmp for the silent state
                    # don't scale the betas for silent states now
                    # wait until the betas for non-silent states are complete to avoid
                    # multiple scaling with the same scalingfactor in one term
                    beta_tmp[id] = sum_



            # iterating over the the non-silent states
            for i in range(self.N):
                if not (self.model_type & kSilentStates) or not (self.silent[i]):
                    sum_ = 0.0

                    for j in range(self.N):
                        # out_state is not silent: get the emission probability
                        # and use beta[t+1]
                        if not (self.model_type & kSilentStates) or not (self.silent[j]):
                            e_index = self.get_emission_index(j, O[t + 1], t + 1)
                            if e_index != -1:
                                emission = self.s[j].b[e_index]
                            else:
                                emission = 0
                            sum_ += self.s[i].out_a[j] * emission * beta[t + 1][j]

                            # out_state is silent: use beta_tmp
                        else:
                            sum_ += self.s[i].out_a[j] * beta_tmp[j]

                    # updating beta[t] for non-silent state
                    beta[t][i] = sum_ / scale[t + 1]

            # updating beta[t] for silent states, finally scale them
            # and resetting beta_tmp
            if self.model_type & kSilentStates:
                for i in range(self.N):
                    if self.silent[i]:
                        beta[t][i] = beta_tmp[i] / scale[t + 1]
                        beta_tmp[i] = 0.0

    def backward_termination(self, O, length, beta, scale):
        beta_tmp = None

        # topological ordering for models with silent states and precomputing
        # the beta_tmp for silent states
        if self.model_type & kSilentStates:
            self.order_topological()

            beta_tmp = double_array_alloc(self.N)
            for k in reversed(range(self.topo_order_length)):#for (k = self.topo_order_length - 1 k >= 0 k--) :
                id = self.topo_order[k]
                assert (self.silent[id] == 1)
                sum_ = 0.0

                for j in range(self.N):
                    # out_state is not silent
                    if not self.silent[j]:
                        # no emission history for the first symbol
                        if not (self.model_type & kHigherOrderEmissions) or self.order[id] == 0:
                            sum_ += self.s[id].out_a[j] * self.s[j].b[O[0]] * beta[0][j]


                    # out_state is silent, beta_tmp[j_id] is useful since we go through
                    # the silent states in reversed topological order
                    else:
                        sum_ += self.s[id].out_a[j] * beta_tmp[j]

                # setting beta_tmp for the silent state
                # don't scale the betas for silent states now
                beta_tmp[id] = sum_

        sum_ = 0.0
        # iterating over all states with pi > 0.0
        for i in range(self.N):
            if self.s[i].pi > 0.0:
                # silent states
                if (self.model_type & kSilentStates) and self.silent[i]:
                    sum_ += self.s[i].pi * beta_tmp[i]

                # non-silent states
                else:
                    # no emission history for the first symbol
                    if not (self.model_type & kHigherOrderEmissions) or self.order[i] == 0:
                        sum_ += self.s[i].pi * self.s[i].b[O[0]] * beta[0][i]

        log_p = Math.log(sum_) - Math.log(scale[0])

        log_scale_sum = 0.0
        for i in range(length):
            log_scale_sum += Math.log(scale[i])

        log_p += log_scale_sum

        return log_p

    def logp_joint(self, O, len, S, slen):
        state_pos = 0
        pos = 0

        prevstate = state = S[0]
        log_p = Math.log(self.s[state].pi)
        if not (self.model_type & kSilentStates) or not self.silent[state]:
            log_p += Math.log(self.s[state].b[O[pos]])
            pos += 1

        for state_pos in range(1, slen):
            if pos >= len:
                break

            state = S[state_pos]
            log_p += Math.log(self.s[state].in_a[prevstate])

            if not (self.model_type & kSilentStates) or not self.silent[state]:
                log_p += Math.log(self.s[state].b[O[pos]])
                pos += 1

            prevstate = state
        else:
            state_pos = slen

        if pos < len:
            Log.error("state sequence too short!  processed only %d symbols", pos)
        if state_pos < slen:
            Log.error("sequence too short!  visited only %d states", state_pos)

        return log_p


    def forward_lean(self, O, len):
        log_scale_sum = 0.0
        non_silent_salpha_sum = 0.0

        alpha_last_col = double_array_alloc(self.N)
        alpha_curr_col = double_array_alloc(self.N)
        scale = double_array_alloc(len)

        if (self.model_type & kSilentStates):
            self.order_topological()

        self.forward_init(alpha_last_col, O[0], scale)
        if scale[0] < GHMM_EPS_PREC:
            Log.error("first symbol can't be generated by hmm")

        log_p = -Math.log(1 / scale[0])

        for t in range(1, len):
            scale[t] = 0.0
            self.update_emission_history(O[t - 1])

            # iterate over non-silent states
            for i in range(self.N):
                if not (self.model_type & kSilentStates) or not (self.silent[i]):
                    e_index = self.get_emission_index(i, O[t], t)
                    if e_index != -1:
                        alpha_curr_col[i] = self.s[i].forward_step(alpha_last_col, self.s[i].b[e_index])
                        scale[t] += alpha_curr_col[i]

                    else:
                        alpha_curr_col[i] = 0

            # iterate over silent states
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
                #sum Math.log(c[t]) scaling values to get  Math.log( P(O|lambda) )
                log_p -= Math.log(c_t)


            # switching pointers of alpha_curr_col and alpha_last_col
            # don't set alpha_curr_col[i] to zero since its overwritten
            switching_tmp = alpha_last_col
            alpha_last_col = alpha_curr_col
            alpha_curr_col = switching_tmp

        # Termination step: compute log likelihood
        if self.model_type & kSilentStates:
            #printf("silent model\n")
            for i in range(len):
                log_scale_sum += Math.log(scale[i])

            for i in range(self.N):
                # use alpha_last_col since the columms are also in the last step
                # switched
                if not self.silent[i]:
                    non_silent_salpha_sum += alpha_last_col[i]

            salpha_log = Math.log(non_silent_salpha_sum)
            log_p = log_scale_sum + salpha_log

        return log_p


    def foba_label_initforward(self, alpha_1, symb, label, scale):
        scale[0] = 0.0

        # iterate over non-silent states
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

        log_p = Math.log(scale[0])

        for t in range(1, len):
            self.update_emission_history(O[t - 1])
            scale[t] = 0.0

            for i in range(self.N):
                if not (self.model_type & kSilentStates) or not (self.silent[i]):
                    if self.label[i] == label[t]:
                    #printf("%d: akt_ state %d, label: %d \t current Label: %d\n",
                    # t, i, self.label[i], label[t])
                        e_index = self.get_emission_index(i, O[t], t)
                        if -1 != e_index:
                            try:
                                alpha[t][i] = self.s[i].forward_step(alpha[t - 1], self.s[i].b[e_index])
                            except Exception, e:
                                pass
                            #if alpha[t][i] < GHMM_EPS_PREC:
                            # printf("alpha[%d][%d] = %g \t ", t, i, alpha[t][i])
                            # printf("self.s[%d].b[%d] = %g\n", i, e_index, self.s[i].b[e_index])
                            #
                            # else printf("alpha[%d][%d] = %g\n", t, i, alpha[t][i])

                        else:
                            alpha[t][i] = 0


                    else:
                        alpha[t][i] = 0

                    scale[t] += alpha[t][i]

                else:
                    Log.error("ERROR: Silent state in foba_label_forward.\n")

            if scale[t] < GHMM_EPS_PREC:
                Log.note("%g\t%g\t%g\t%g\t%g\n", scale[t - 5], scale[t - 4], scale[t - 3], scale[t - 2], scale[t - 1])
                Log.error("scale = %g smaller than eps = EPS_PREC in the %d-th char.\ncannot generate emission: %d with label: %d in sequence of length %d\n", scale[t], t, O[t], label[t], len)

            c_t = 1 / scale[t]
            for i in range(self.N):
                alpha[t][i] *= c_t

            if not (self.model_type & kSilentStates):
                log_p -= Math.log(c_t)

        return log_p

    def label_logp(self, O, label, len):

        scale = double_array_alloc(len)
        alpha = double_matrix_alloc(len, self.N)

        # run ghmm_dmodel_forward
        log_p = self.label_forward(O, label, len, alpha, scale)
        return log_p

    def label_backward(self, O, label, len, beta, scale):
        beta_tmp = double_array_alloc(self.N)
        for t in range(len):
            if scale[t] == 0:
                Log.error("")

        # check for silent states
        if self.model_type & kSilentStates:
            Log.error("ERROR: No silent states allowed in labelled HMMnot \n")

        # initialize
        for i in range(self.N):
            # start only in states with the correct label
            if (label[len - 1] == self.label[i]):
                beta[len - 1][i] = 1.0
            else:
                beta[len - 1][i] = 0.0

            beta_tmp[i] = beta[len - 1][i] / scale[len - 1]


        # initialize emission history
        if not (self.model_type & kHigherOrderEmissions):
            self.maxorder = 0
        for t in range(len - (self.maxorder), len):
            self.update_emission_history(O[t])


        # Backward Step for t = T-1, ..., 0
        # beta_tmp: Vector for storage of scaled beta in one time step
        # loop over reverse topological ordering of silent states, non-silent states
        for t in reversed(range(len - 1)): #for (t = len - 2 t >= 0 t--) :

            # updating of emission_history with O[t] such that emission_history
            # memorizes O[t - maxorder ... t]
            if 0 <= t - self.maxorder + 1:
                self.update_emission_history_front(O[t - self.maxorder + 1])

            for i in range(self.N):
                sum_ = 0.0
                for j in range(self.N):
                    # The state has only a emission with probability > 0, if the label matches
                    if label[t] == self.label[i]:
                        e_index = self.get_emission_index(j, O[t + 1], t + 1)
                        if e_index != -1:
                            emission = self.s[j].b[e_index]
                        else:
                            emission = 0.0

                    else:
                        emission = 0.0

                    sum_ += emission * self.s[i].out_a[j] * beta_tmp[j]

                beta[t][i] = sum_
                # if ((beta[t][i] > 0) and ((beta[t][i] < .01) or (beta[t][i] > 100))) beta_out++

            for i in range(self.N):
                beta_tmp[i] = beta[t][i] / scale[t]


    def ghmm_dl_forward_lean(self, O, label, len):
        log_scale_sum = 0.0
        non_silent_salpha_sum = 0.0

        alpha_last_col = double_array_alloc(self.N)
        alpha_curr_col = double_array_alloc(self.N)
        scale = double_array_alloc(len)

        if self.model_type & kSilentStates:
            self.order_topological()

        self.foba_label_initforward(alpha_last_col, O[0], label[0], scale)
        if scale[0] < GHMM_EPS_PREC:
            Log.error("first symbol can't be generated by hmm")

        log_p = -Math.log(1 / scale[0])

        for t in range(1, len):
            scale[t] = 0.0

            # iterate over non-silent states
            for i in range(self.N):
                if not (self.model_type & kSilentStates) or not (self.silent[i]):

                    # printf("  akt_ state %d\n",i)
                    if self.label[i] == label[t]:
                        e_index = self.get_emission_index(i, O[t], t)
                        if e_index != -1:
                            alpha_curr_col[i] = self.s[i].forward_step(alpha_last_col, self.s[i].b[e_index])
                            scale[t] += alpha_curr_col[i]

                        else:
                            alpha_curr_col[i] = 0

                    else:
                        alpha_curr_col[i] = 0

            # iterate over silent states
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
                #sum_ Math.log(c[t]) scaling values to get  Math.log( P(O|lambda) )
                log_p -= Math.log(c_t)


            # switching pointers of alpha_curr_col and alpha_last_col
            # don't set alpha_curr_col[i] to zero since its overwritten
            switching_tmp = alpha_last_col
            alpha_last_col = alpha_curr_col
            alpha_curr_col = switching_tmp

        # Termination step: compute log likelihood
        if self.model_type & kSilentStates:
            #printf("silent model\n")
            for i in range(len):
                log_scale_sum += Math.log(scale[i])

            for i in range(self.N):
                if not self.silent[i]:
                    non_silent_salpha_sum += alpha_curr_col[i]

            salpha_log = Math.log(non_silent_salpha_sum)
            log_p = log_scale_sum + salpha_log

        return log_p

    def update_tie_groups(self):
    #define CUR_PROC "ghmm_dmodel_update_tied_groups"
        nr = 0

        # do nothing if there are no tied emissions
        if not (self.model_type & kTiedEmissions):
            Log.error("No tied emissions. Exiting.")

        if self.model_type & kHigherOrderEmissions:
            new_emissions = ARRAY_MALLOC(pow(self.M, self.maxorder + 1))

        else:
            new_emissions = ARRAY_MALLOC(self.M)

        for i in range(self.N):
            # find tie group leaders
            if self.tied_to[i] == i:


                if self.model_type & kHigherOrderEmissions:
                    bi_len = pow(self.M, self.order[i] + 1)
                else:
                    bi_len = self.M

                if self.model_type & kSilentStates and self.silent[i]:
                    Log.warning("Tie group leader %d is silent.", i)
                    nr = 0
                    # initializing with zeros
                    for k in range(bi_len):
                        new_emissions[k] = 0.0

                else:
                    nr = 1
                    # initializing with tie group leader emissions
                    for k in range(bi_len):
                        new_emissions[k] = self.s[i].b[k]

                # finding tie group members
                for j in range(i + 1, self.N):
                    if self.tied_to[j] == i and (not (self.model_type & kHigherOrderEmissions) or self.order[i] == self.order[j]):
                        # silent states have no contribution to the pooled emissions within a group
                        if not (self.model_type & kSilentStates) or (self.silent[j] == 0):
                            nr += 1
                            # printf("  tie group member %d . leader %d.\n",j,i)
                            # summing up emissions in the tie group
                            for k in range(bi_len):
                                new_emissions[k] += self.s[j].b[k]
                        else:
                            Log.warning("Tie group member %d is silent.", j)

                # updating emissions
                if nr > 1:
                    for j in range(i, self.N):
                        # states within one tie group are required to have the same order
                        if self.tied_to[j] == i and (not (self.model_type & kHigherOrderEmissions) or self.order[i] == self.order[j]) and (
                                not (self.model_type & kSilentStates) or (self.silent[j] == 0)):
                            for k in range(bi_len):
                                self.s[j].b[k] = new_emissions[k] / nr
                                # printf("s(%d)[%d] . %f / %f = %f\n", j, k, new_emissions[k], nr,mo.s[j].b[k])
                else:
                    Log.note("The tie group with leader {{id}} has only one non-silent state. Kind of pointless!", {"id": i})

    def reestimate_setlambda(self, r):
        for i in range(self.N):
            reachable = 1
            positive = 0

            # Pi
            self.s[i].pi = r.pi_num[i] / r.pi_denom

            # A
            # note: denom. might be 0 never reached state?
            p_i = 0.0
            if r.a_denom[i] < GHMM_EPS_PREC:
                for h in range(self.N):
                    p_i += self.s[i].in_a[h]

                if p_i == 0.0:
                    Log.note("State %d can't be reached (prob = 0.0)", i)
                    reachable = 0

                factor = 0.0

            else:
                factor = (1 / r.a_denom[i])

            for j in range(self.N):
                # TEST: denom. < numerator
                if (r.a_denom[i] - r.a_num[i][j]) <= -GHMM_EPS_PREC:
                    Log.error("numerator > denominator")

                self.s[i].out_a[j] = r.a_num[i][j] * factor
                if r.a_num[i][j] >= GHMM_EPS_PREC:
                    positive = 1
                    # important: also update in_a
                l = 0
                self.s[j].in_a[i] = self.s[i].out_a[j]

            # if not positive:
            # str = ighmm_mprintf(None, 0,
            # "All numerator a[%d][j] == 0 (denom=%.4f, P(in)=%.4f)not \n",
            # i, r.a_denom[i], p_i)
            # Log.error(str)
            # m_free(str)
            #

            # if fix, continue to next state
            if self.s[i].fix:
                continue

            # B
            if self.model_type & kHigherOrderEmissions:
                size = pow(self.M, self.order[i])
            else:
                size = 1
                # If all in_a's are zero, the state can't be reached.
            # Set all b's to -1.0
            if not reachable:
                for hist in range(size):
                    col = hist * self.M
                    for m in range(col, col + self.M):
                        self.s[i].b[m] = -1.0
                else:
                    hist = size
            else:
                for hist in range(size):
                    # If the denominator is very small, we have not seen many emissions
                    # in this state with this history.
                    # We are conservative and just skip them.
                    if r.b_denom[i][hist] < GHMM_EPS_PREC:
                        continue
                    else:
                        factor = (1.0 / r.b_denom[i][hist])

                    positive = 0
                    # TEST: denom. < numerator
                    col = hist * self.M
                    for m in range(col, col + self.M):
                        if (r.b_denom[i][hist] - r.b_num[i][m]) <= -GHMM_EPS_PREC:
                            Log.note("numerator b (%.4f) > denom (%.4f)!\n", r.b_num[i][m], r.b_denom[i][hist])

                        self.s[i].b[m] = r.b_num[i][m] * factor
                        if self.s[i].b[m] >= GHMM_EPS_PREC:
                            positive = 1

                    if not positive:
                        Log.note("All numerator b[%d][%d-%d] == 0 (denom = %g)!\n", i, col, col + self.M, r.b_denom[i][hist])


    def reestimate_one_step(self, r, seq_number, seq_length, O, seq_w):
        alpha = None
        beta = None
        scale = None
        T_k = 0
        # first set maxorder to zero if model_type & kHigherOrderEmissions is FALSE
        #
        # TODO XXX use model.maxorder only
        # if model_type & kHigherOrderEmissions is TRUE

        if not (self.model_type & kHigherOrderEmissions):
            self.maxorder = 0

        log_p = 0.0
        # loop over all sequences
        for k in range(seq_number):
            self.emission_history = 0
            T_k = seq_length[k]        # current seq. length

            # initialization of  matrices and vector depends on T_k
            alpha, beta, scale = ighmm_reestimate_alloc_matvek(T_k, self.N)

            log_p_k = self.forward(O[k], T_k, alpha, scale)
            log_p += log_p_k

            self.backward(O[k], T_k, beta, scale)

            # loop over all states
            for i in range(self.N):
                # Pi
                r.pi_num[i] += seq_w[k] * alpha[0][i] * beta[0][i]
                r.pi_denom += seq_w[k] * alpha[0][i] * beta[0][i]

                for t in range(T_k - 1):
                    # B
                    if not self.s[i].fix:
                        e_index = self.get_emission_index(i, O[k][t], t)
                        if e_index != -1:
                            gamma = seq_w[k] * alpha[t][i] * beta[t][i]
                            r.b_num[i][e_index] += gamma
                            r.b_denom[i][e_index / (self.M)] += gamma

                    self.update_emission_history(O[k][t])

                    # A
                    r.a_denom[i] += (seq_w[k] * alpha[t][i] * beta[t][i])
                    for j in range(self.N):
                        e_index = self.get_emission_index(j, O[k][t + 1], t + 1)
                        if e_index != -1:
                            r.a_num[i][j] += (seq_w[k] * alpha[t][i] * self.s[i].out_a[j] * self.s[j].b[e_index] * beta[t + 1][j] * (1.0 / scale[t + 1]))       # c[t] = 1/scale[t]
                else:
                    t = T_k - 1
                    # B: last iteration for t==T_k-1 :
                if not self.s[i].fix:
                    e_index = self.get_emission_index(i, O[k][t], t)
                    if e_index != -1:
                        gamma = seq_w[k] * alpha[t][i] * beta[t][i]
                        r.b_num[i][e_index] += gamma
                        r.b_denom[i][e_index / self.M] += gamma



        # new parameter lambda: set directly in model
        self.reestimate_setlambda(r)
        self.check()

        return log_p


    def reestimate_one_step_lean(self, r, seq_number, seq_length, seqs, seq_w):
        # allocating memory for two columns of alpha matrix
        scale = [0.0]
        alpha_last_col = ARRAY_CALLOC(self.N)
        alpha_curr_col = ARRAY_CALLOC(self.N)

        # allocating 2*N local_store_t
        last_est = ARRAY_CALLOC(self.N)
        for i in range(self.N):
            last_est[i] = reestimate_alloc(self)
        curr_est = ARRAY_CALLOC(self.N)
        for i in range(self.N):
            curr_est[i] = reestimate_alloc(self)


        # temporary array to hold logarithmized summands
        # for sums over probabilities :
        summands = ARRAY_CALLOC(max(self.N, pow(self.M, self.maxorder + 1)) + 1)

        for k in range(seq_number):
            len = seq_length[k]
            O = seqs[k]

            self.forward_init(alpha_last_col, O[0], scale)
            if scale[0] < GHMM_EPS_PREC:
                Log.error("first symbol can't be generated by hmm")

            log_p = Math.log(scale[0])

            for t in range(1, len):
                old_scale = scale[0]
                scale[0] = 0.0
                self.update_emission_history(O[t - 1])

                # iterate over non-silent states
                for i in range(self.N):
                    # printf("  akt_ state %d\n",i)

                    e_index = self.get_emission_index(i, O[t], t)
                    if e_index != -1:
                        alpha_curr_col[i] = self.s[i].forward_step(alpha_last_col, self.s[i].b[e_index])
                        scale[0] += alpha_curr_col[i]

                    else:
                        alpha_curr_col[i] = 0

                if scale[0] < GHMM_EPS_PREC:
                    Log.error("scale smaller than eps")

                c_t = 1 / scale[0]
                for i in range(self.N):
                    alpha_curr_col[i] *= c_t

                # sum Math.log(c[t]) scaling values to get  Math.log( P(O|lambda) )
                log_p -= Math.log(c_t)

                scalingf = 1 / old_scale
                for m in range(self.N):
                    for i in range(self.N):
                        # computes estimates for the numerator of transition probabilities :
                        for j in range(self.N):
                            for g in range(self.N):
                                e_index = self.get_emission_index(g, O[t], t)
                                # scales all summands with the current
                                summands[g] = last_est[m].a_num[i][j] * self.s[j].in_a[g] * self.s[g].b[e_index] * scalingf
                            else:
                                g = self.N

                            if j == m:
                                e_index = self.get_emission_index(j, O[t], t)
                                # alpha is scaled. no other scaling necessary
                                summands[g] = alpha_last_col[i] * self.s[i].out_a[j] * self.s[j].b[e_index]
                                g += 1

                            curr_est[m].a_num[i][j] = nologSum(summands, g)

                        # computes denominator of transition probabilities
                        for g in range(self.N):
                            e_index = self.get_emission_index(m, O[t], t)
                            # scales all summands with the current factor
                            summands[g] = last_est[m].a_denom[i] * self.s[m].in_a[g] * self.s[m].b[e_index] * scalingf

                        if i == m:
                            e_index = self.get_emission_index(i, O[t], t)
                            # alpha is scaled. no other scaling necessary
                            g += 1
                            summands[g] = alpha_last_col[i] * self.s[i].out_a[i] * self.s[i].b[e_index]

                        curr_est[m].a_denom[i] = nologSum(summands, g)

                        # computes estimates for the numerator of emmission probabilities:
                        if self.model_type & kHigherOrderEmissions:
                            size = pow(self.M, self.order[i])
                        else:
                            size = 1
                        for h in range(size):
                            for s in range(h * self.M, h * self.M + self.M):
                                for g in range(self.N):
                                    e_index = self.get_emission_index(g, O[t], t)
                                    # scales all summands with the last scaling factor
                                    # of alpha
                                    summands[g] = last_est[m].b_num[i][s] * self.s[m].in_a[g] * self.s[g].b[e_index] * scalingf
                                else:
                                    g = len(self.N)

                                curr_est[m].b_num[i][s] = nologSum(summands, g)

                        e_index = self.get_emission_index(m, O[t], t)
                        if i == m:
                            # alpha is scaled. no other scaling necessary
                            curr_est[i].b_num[i][e_index] += alpha_last_col[i] * self.s[i].out_a[i] * self.s[i].b[e_index]
                            break

                # switching pointers of alpha_curr_col and alpha_last_col
                switching_tmp = alpha_last_col
                alpha_last_col = alpha_curr_col
                alpha_curr_col = switching_tmp

                switch_lst = last_est
                last_est = curr_est
                curr_est = switch_lst

            # filling the usual reestimate arrays by summing all states
            for m in range(self.N):
                for i in range(self.N):
                    # PI
                    # XXX calculate the estimates for pi numerator :
                    curr_est[m].pi_num[i] = self.s[i].pi
                    curr_est[m].pi_denom += self.s[i].pi

                    r.pi_num[i] += seq_w[k] * curr_est[m].pi_num[i]
                    r.pi_denom += seq_w[k] * curr_est[m].pi_num[i]

                    # A
                    curr_est[m].a_denom[i] = 0
                    for j in range(self.N):
                        r.a_num[i][j] += seq_w[k] * curr_est[m].a_num[i][j]
                        curr_est[m].a_denom[i] += curr_est[m].a_num[i][j]

                    r.a_denom[i] += seq_w[k] * curr_est[m].a_denom[i]

                    # B
                    for h in range(size):
                        curr_est[m].b_denom[i][h] = 0
                        for s in range(h * self.M, h * self.M + self.M):
                            r.b_num[i][s] += seq_w[k] * curr_est[m].b_num[i][s]
                            curr_est[m].b_denom[i][h] += curr_est[m].b_num[i][s]

                        r.b_denom[i][h] += seq_w[k] * curr_est[m].b_denom[i][h]
                        # PI


        return log_p


    def baum_welch(self, sq):
        return self.baum_welch_nstep(sq, MAX_ITER_BW, EPS_ITER_BW)


    def baum_welch_nstep(self, sq, max_step, likelihood_delta):
        # local store for all iterations :
        r = reestimate_alloc(self).reestimate_init(self)

        log_p_old = -1e300
        n = 1

        # main loop Baum-Welch-Alg.
        while n <= max_step:

            if 1:
                log_p = self.reestimate_one_step(r, sq.seq_number, sq.seq_len, sq.seq, sq.seq_w)
            else:
                log_p = self.reestimate_one_step_lean(r, sq.seq_number, sq.seq_len, sq.seq, sq.seq_w)

            diff = log_p - log_p_old
            # error in convergence ?
            if diff < -GHMM_EPS_PREC:
                Log.error("No convergence: log P < log P-oldnot  (n=%d)\n", n)

            elif log_p > GHMM_EPS_PREC:
                Log.error("No convergence: log P > 0not  (n=%d)\n", n)

            # stop iterations?
            if diff < abs(likelihood_delta * log_p):
                Log.note("Convergence after %d steps", n)
                break

            else:
                # for next iteration :
                log_p_old = log_p
                r.reestimate_init(self)  # sets all fields to zero
                n += 1

                # while (n <= MAX_ITER)

        # log_p of reestimated model
        log_p = 0.0
        for k in range(sq.seq_number):
            log_p_k = self.logp(sq.seq[k], sq.seq_len[k])

        log_p += log_p_k

        return log_p


    def reestimate_one_step_label(self, r, seq_number, seq_length, O, label, seq_w):
        valid = 0

        # first set maxorder to zero if model_type & kHigherOrderEmissions is FALSE
        #
        # TODO XXX use model.maxorder only
        # if model_type & kHigherOrderEmissions is TRUE

        if not (self.model_type & kHigherOrderEmissions):
            self.maxorder = 0

        log_p = 0.0

        # loop over all sequences
        for k in range(seq_number):
            self.emission_history = 0
            T_k = seq_length[k]        # current seq. length

            # initialization of  matrices and vector depends on T_k
            alpha, beta, scale = ighmm_reestimate_alloc_matvek(T_k, self.N)
            log_p_k = self.label_forward(O[k], label[k], T_k, alpha, scale)

            if 1:
                log_p += log_p_k

                log_p_k = self.label_backward(O[k], label[k], T_k, beta, scale)

                # loop over all states
                for i in range(self.N):
                    # Pi
                    r.pi_num[i] += seq_w[k] * alpha[0][i] * beta[0][i]
                    r.pi_denom += seq_w[k] * alpha[0][i] * beta[0][i]

                    for t in range(T_k-1):
                        # B
                        if not (self.s[i].fix) and (self.label[i] == label[k][t]):
                            e_index = self.get_emission_index(i, O[k][t], t)
                            if e_index != -1:
                                gamma = seq_w[k] * alpha[t][i] * beta[t][i]
                                r.b_num[i][e_index] += gamma
                                r.b_denom[i][e_index / (self.M)] += gamma

                        self.update_emission_history(O[k][t])

                        # A
                        r.a_denom[i] += seq_w[k] * alpha[t][i] * beta[t][i]
                        for j in range(self.N):
                            if label[k][t + 1] != self.label[j]:
                                continue
                            e_index = self.get_emission_index(j, O[k][t + 1], t + 1)
                            if e_index != -1:
                                r.a_num[i][j] += (seq_w[k] * alpha[t][i] * self.s[i].out_a[j] * self.s[j].b[e_index] * beta[t + 1][j] * (1.0 / scale[t + 1]))

                    # B: last iteration for t==T_k-1 :
                    t = T_k - 1
                    if not self.s[i].fix and self.label[i] == label[k][t]:
                        e_index = self.get_emission_index(i, O[k][t], t)
                        if e_index != -1:
                            gamma = seq_w[k] * alpha[t][i] * beta[t][i]
                            r.b_num[i][e_index] += gamma
                            r.b_denom[i][e_index / self.M] += gamma
        if valid:
            # new parameter lambda: set directly in model
            self.reestimate_setlambda(r)
            errors = self.check()
            if errors:
                Log.error("Reestimated model is invalid, model_check found %d errors", -errors)

        return log_p

    def label_baum_welch(self, sq):
        return self.label_baum_welch_nstep(sq, MAX_ITER_BW, EPS_ITER_BW)


    def label_baum_welch_nstep(self, sq, max_step, likelihood_delta):
        # local store for all iterations :
        r = reestimate_alloc(self)

        log_p_old = -1e300
        n = 1

        # main loop Baum-Welch-Alg.
        while n <= max_step:

            log_p = self.reestimate_one_step_label(r, sq.seq_number, sq.seq_len, sq.seq, sq.state_labels, sq.seq_w)

            if n == 1:
                Log.note("{{log_p|round(places=5)}} (-log_p input model)\n", {"log_p": -log_p})
            else:
                Log.note("{{log_p|round(places=5)}} (-log_p input model)\n", {"log_p": -log_p})

            diff = log_p - log_p_old
            # error in convergence ?
            if diff < -GHMM_EPS_PREC:
                Log.error("No convergence: log P < log P-oldnot  (n = %d)\n", n)

            elif log_p > GHMM_EPS_PREC:
                Log.error("No convergence: log P > 0not  (n = %d)\n", n)

            # stop iterations?
            if diff < abs(likelihood_delta * log_p):
                Log.note("Convergence after %d steps\n", n)
                break

            else:
                # for next iteration :
                log_p_old = log_p
                r.reestimate_init(self)  # sets all fields to zero
                n += 1

                # while (n <= MAX_ITER)

        # log_p of reestimated model
        log_p = 0.0
        for k in range(sq.seq_number):
            log_p_k = self.label_logp(sq.seq[k], sq.state_labels[k], sq.seq_len[k])
            log_p += log_p_k

        Log.note("%8.5f (-log_p optimized model)", -log_p)
        return log_p

    def check(self):
        imag = 0

        # The sum of the Pi[i]'s is 1
        sum_ = 0.0
        for i in range(self.N):
            sum_ += self.s[i].pi

        if abs(sum_ - 1.0) >= GHMM_EPS_PREC:
            Log.error("sum Pi[i] != 1.0")


        # check each state
        for i in range(self.N):
            sum_ = sum(self.s[i].out_a)

            if sum_ == 0.0:
                Log.warning("sum of s[%d].out_a[*] = 0.0 (assumed final state but %d transitions)", i, self.N)
            elif abs(sum_ - 1.0) >= GHMM_EPS_PREC:
                Log.error("sum of s[%d].out_a[*] = %f != 1.0", i, sum_)

            # Sum the a[i][j]'s : normalized in transitions
            sum_ = self.s[i].pi
            sum_ += sum(self.s[i].in_a)

            if abs(sum_) <= GHMM_EPS_PREC:
                imag = 1
                Log.note("state %d can't be reached", i)


            # Sum the b[j]'s: normalized emission probs
            sum_ = sum(self.s[i].b)

            if imag:
                # not reachable states
                if (abs(sum_ + self.M) >= GHMM_EPS_PREC):
                    Log.error("state %d can't be reached but is not set as non-reachale state", i)
            elif (self.model_type & kSilentStates) and self.silent[i]:
                # silent states
                if sum_ != 0.0:
                    Log.error("state %d is silent but has a non-zero emission probability", i)
            else:
                # normal states
                if abs(sum_ - 1.0) >= GHMM_EPS_PREC:
                    Log.error("sum s[%d].b[*] = %f != 1.0", i, sum_)

    def update_emission_history_front(self, O):
        if self.model_type & kHigherOrderEmissions:
            self.emission_history = pow(self.M, self.maxorder - 1) * O + self.emission_history / self.M

    def get_transition(self, i, j):
        if self.s and self.s[i].out_a and self.s[j].in_a:
            return self.s[i].out_a[j]
        return 0.0


    def set_transition(mo, i, j, prob):
        if mo.s and mo.s[i].out_a and mo.s[j].in_a:
            mo.s[i].out_a[j] = prob
            mo.s[j].in_a[i] = prob

    def setSilent(self, index, value):
        self.silent[index] = value

    def getSilent(self, index):
        return self.silent[index]

    def order_topological(self):
        self.topo_order = topological_sort(self)
        self.topo_order_length = len(self.topo_order)


    def normalize(self):
    # Scales the output and transitions probs of all states in a given model
        pi_sum = 0.0
        i_id = 0
        res = 0
        size = 1

        for i in range(self.N):
            if self.s[i].pi >= 0.0:
                pi_sum += self.s[i].pi
            else:
                self.s[i].pi = 0.0

            # check model_type before using state order
            if self.model_type & kHigherOrderEmissions:
                size = pow(self.M, self.order[i])

            # normalize transition probabilities
            ighmm_cvector_normalize(self.s[i].out_a, 0, self.N)

            # for every outgoing probability update the corrosponding incoming probability
            for j in range(self.N):
                self.s[j].in_a[i] = self.s[i].out_a[j]

            # normalize emission probabilities, but not for silent states
            if not ((self.model_type & kSilentStates) and self.silent[i]):
                if size == 1:
                    ighmm_cvector_normalize(self.s[i].b, 0, self.M)
                else:
                    for m in range(size):
                        #NORMALIZE THIS SUB-ARRAY
                        ighmm_cvector_normalize(self.s[i].b, m * self.M, self.M)

        for i in range(self.N):
            self.s[i].pi /= pi_sum


    def background_apply(self, background_weight):
        if not (self.model_type & kBackgroundDistributions):
            Log.error("Error: No background distributions")

        for i in range(self.N):
            if self.background_id[i] != kNoBackgroundDistribution:
                if self.model_type & kHigherOrderEmissions:
                    if self.order[i] != self.bp.order[self.background_id[i]]:
                        Log.error("State (%d) and background order (%d) do not match in state %d. Background_id = %d",
                            self.order[i],
                            self.bp.order[self.background_id[i]],
                            i,
                            self.background_id[i]
                        )

                    # XXX Cache in ghmm_dbackground
                    size = pow(self.M, self.order[i] + 1)
                    for j in range(size):
                        self.s[i].b[j] = (1.0 - background_weight[i]) * self.s[i].b[j] + background_weight[i] * self.bp.b[self.background_id[i]][j]
                else:
                    if self.bp.order[self.background_id[i]] != 0:
                        Log.error("Error: State and background order do not match\n")

                    for j in range(self.M):
                        self.s[i].b[j] = (1.0 - background_weight[i]) * self.s[i].b[j] + background_weight[i] * self.bp.b[self.background_id[i]][j]


    def label_generate_sequences(self, seed, len, seq_number, Tmax, native=False):
        n = 0

        sq = sequence(seq_number)

        # allocating additional fields for the state sequence in the sequence class
        sq.states = ARRAY_CALLOC(seq_number)
        sq.states_len = ARRAY_CALLOC(seq_number)

        # allocating additional fields for the labels in the sequence class
        sq.state_labels = ARRAY_CALLOC(seq_number)
        sq.state_labels_len = ARRAY_CALLOC(seq_number)

        # A specific length of the sequences isn't given. As a model should have
        #     an end state, the konstant MAX_SEQ_LEN is used.
        if len <= 0:
            len = GHMM_MAX_SEQ_LEN

        if seed > 0:
            random_mt.set_seed( seed)


        # initialize the emission history
        self.emission_history = 0

        while n < seq_number:
            sq.seq[n] = ARRAY_CALLOC(len)

            # for silent models we have to allocate for the maximal possible number
            #       of lables and states
            if self.model_type & kSilentStates:
                sq.states[n] = ARRAY_CALLOC(len * self.N)
                sq.state_labels[n] = ARRAY_CALLOC(len * self.N)

            else:
                sq.states[n] = ARRAY_CALLOC(len)
                sq.state_labels[n] = ARRAY_CALLOC(len)

            pos = label_pos = 0

            # Get a random initial state i
            p = random_mt.float23()
            sum_ = 0.0
            for state in range(self.N):
                sum_ += self.s[state].pi
                if sum_ >= p:
                    break

            while pos < len:
                # save the state path and label
                sq.states[n][label_pos] = state
                sq.state_labels[n][label_pos] = self.label[state]
                label_pos += 1

                # Get a random output m if the state is not a silent state
                if not (self.model_type & kSilentStates) or not (self.silent[state]):
                    m = self.get_random_output(state, pos)
                    self.update_emission_history(m)
                    sq.seq[n][pos] = m
                    pos += 1


                # get next state
                p = random_mt.float23()
                if pos < self.maxorder:
                    max_sum = 0.0
                    for j in range(0, self.N):
                        if not (self.model_type & kHigherOrderEmissions) or self.order[j] < pos:
                            max_sum += self.s[state].out_a[j]

                    if j and abs(max_sum) < GHMM_EPS_PREC:
                        Log.error("No possible transition from state %d since all successor states require more history than seen up to position: %d.", state, pos)
                        break

                    if j:
                        p *= max_sum

                sum_ = 0.0
                for j in range(0, self.N):
                    if not (self.model_type & kHigherOrderEmissions) or self.order[j] < pos:
                        sum_ += self.s[state].out_a[j]
                        if sum_ >= p:
                            break

                if sum_ == 0.0:
                    Log.note("final state (%d) reached at position %d of sequence %d", state, pos, n)
                    break

                state = j
                # while pos < len:
            # realocate state path and label sequence to actual size
            if self.model_type & kSilentStates:
                sq.states[n] = ARRAY_REALLOC(sq.states[n], label_pos)
                sq.state_labels[n] = ARRAY_REALLOC(sq.state_labels[n], label_pos)

            sq.seq_len[n] = pos
            sq.states_len[n] = label_pos
            sq.state_labels_len[n] = label_pos
            n += 1
            # while  n < seq_number :
        return sq
