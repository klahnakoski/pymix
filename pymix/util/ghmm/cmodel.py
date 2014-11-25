import random
from pyLibrary.maths import Math
from pymix.util.ghmm import random_mt
from pymix.util.ghmm.matrixop import cholesky
from pymix.util.ghmm.sequences import sequence
from pymix.util.ghmm.cstate import ghmm_cstate
from pymix.util.ghmm.sfoba import sfoba_initforward, LOWER_SCALE_BOUND, sfoba_stepforward
from pymix.util.ghmm.types import kContinuousHMM, kSilentStates
from pymix.util.ghmm.wrapper import ARRAY_REALLOC, GHMM_MAX_SEQ_LEN, ARRAY_CALLOC, matrix_alloc, GHMM_EPS_PREC, DBL_MIN, ighmm_cmatrix_stat_alloc, ighmm_cvector_normalize
from pymix.util.logs import Log


class ghmm_cmodel:
    def __init__(self, N, modeltype):
        assert (modeltype & kContinuousHMM)
        # Number of states
        self.N = N  # int
        # Maximun number of components in the states
        self.M = 1  # int
        # Number of dimensions of the emission components.
        # All emissions must have the same number of dimensions
        self.dim = 1  # int
        # ghmm_cmodel includes continuous model with one transition matrix
        # (cos  is set to 1) and an extension for models with several matrices
        # (cos is set to a positive integer value > 1).
        self.cos = 1  # int
        # prior for a priori prob. of the model. -1 means no prior specified (all
        # models have equal prob. a priori.
        self.prior = None  # double

        # contains a arbitrary name for the model (null terminated utf-8)

        self.name = None  # char *

        # Contains bit flags for varios model extensions such as
        # kSilentStates (see ghmm.h for a complete list)
        self.model_type = modeltype

        # All states of the model. Transition probs are part of the states.
        self.s = [ghmm_cstate(1, N, self.cos) for i in range(N)]

        # pointer to a ghmm_cmodel_class_change_context struct necessary for multiple transition
        # classes * /
        self.class_change = None  # ghmm_cmodel_class_change_context *

    def ghmm_cmodel_get_random_var(self, state, m, native=False):
        # PARAMETER x IS THE RETURN VALUES
        # define CUR_PROC "ghmm_cmodel_get_random_var"
        emission = self.s[state].e[m]
        return emission.sample(native=native)

    def generate_sequences(self, seed, global_len, seq_number, Tmax, native=False):
        # An end state is characterized by not having an output probabiliy.

        len = global_len
        up = 0
        stillbadseq = 0
        reject_os_tmp = 0

        sq = sequence([[] for i in range(seq_number)])

        # set dimension of the sequence to match dimension of the model (multivariate)
        sq.dim = self.dim

        # A specific length of the sequences isn't given. As a model should have
        # an end state, the konstant MAX_SEQ_LEN is used.
        if len <= 0:
            len = GHMM_MAX_SEQ_LEN

        # Maximum length of a sequence not given
        if Tmax <= 0:
            Tmax = GHMM_MAX_SEQ_LEN

        # seed == -1: Initialization, has already been done outside the function
        if seed >= 0:
            if seed > 0:
                if native:
                    random.seed(seed)
                else:
                    random_mt.set_seed(seed)
            else:                        # Random initialization!
                pass

        n = 0
        reject_os = reject_tmax = 0

        # calculate cholesky decomposition of covariance matrix (multivariate case),
        # this needs to be called before ghmm_c_get_random_var
        if self.dim > 1:
            for i in range(self.N):
                for m in range(self.s[i].M):
                    self.s[i].e[m].sigmacd = cholesky(self.dim, self.s[i].e[m].variance)

        while n < seq_number:
            # Test: A new seed for each sequence
            stillbadseq = badseq = 0
            sq.seq[n] = matrix_alloc(len, self.dim)

            # Get a random initial state i
            if native:
                p = random.random()
            else:
                p = random_mt.float23()
            sum_ = 0.0
            for i in range(self.N):
                sum_ += self.s[i].pi
                if sum_ >= p:
                    break
            else:
                i = self.N
            if i == self.N:          # Can happen by a rounding error in the input
                i -= 1
                while i > 0 and self.s[i].pi == 0.0:
                    i -= 1


            # Get a random initial output
            # . get a random m and then respectively a pdf omega.
            if native:
                p = random.random()
            else:
                p = random_mt.float23()
            sum_ = 0.0
            for m in range(self.s[i].M):
                sum_ += self.s[i].c[m]
                if sum_ >= p:
                    break
            else:
                m = self.s[i].M
            if m == self.s[i].M:
                m -= 1

            # Get random numbers according to the density function
            sq.seq[n][0] = self.ghmm_cmodel_get_random_var(i, m, native=native)
            pos = 1

            # The first symbol chooses the start class
            if self.cos == 1:
                clazz = 0
            else:
                #Log.error("1: cos = %d, k = %d, t = %d\n",self.cos,n,state)

                if not self.class_change.get_class:
                    Log.error("ERROR: get_class not initialized\n")
                    return None

                clazz = self.class_change.get_class(self, sq.seq[n], n, 0)
                if clazz >= self.cos:
                    Log.error("ERROR: get_class returned index %d but model has only %d classes !\n", clazz, self.cos)
                    return None

            while pos < len:
                # Get a new state
                if native:
                    p=random.random()
                else:
                    p = random_mt.float23()
                sum_ = 0.0
                for j in range(self.N):
                    sum_ += self.s[i].out_a[clazz][j]
                    if sum_ >= p:
                        break
                else:
                    j = self.N

                if j == self.N:  # Can happen by a rounding error
                    j -= 1
                    while j > 0 and self.s[i].out_a[clazz][j] == 0.0:
                        j -= 1

                if sum_ == 0.0:
                    if self.N > 0:
                        # Repudiate the sequence, if all self.s[i].out_a[clazz][.] == 0,
                        # that is, clazz "clazz" isn't used in the original data:
                        # go out of the while-loop, n should not be counted.
                        # Log.error("Zustand %d, clazz %d, len %d out_states %d \n", i, clazz,
                        # state, self.N)
                        badseq = 1
                        # break

                        # Try: If the clazz is "empty", try the neighbour clazz
                        # first, sweep down to zero if still no success, sweep up to
                        # COS - 1. If still no success -. Repudiate the sequence.
                        if clazz > 0 and up == 0:
                            clazz -= 1
                            continue
                        elif clazz < self.cos - 1:
                            clazz += 1
                            up = 1
                            continue
                        else:
                            stillbadseq = 1
                            break

                    else:
                        # Final state reached, out of while-loop
                        break

                i = j
                # fprintf(stderr, "%d\n", i)
                # fprintf(stderr, "%d\n", i)

                # Get output from state i
                if native:
                    p=random.random()
                else:
                    p = random_mt.float23()
                sum_ = 0.0
                for m in range(self.s[i].M):
                    sum_ += self.s[i].c[m]
                    if sum_ >= p:
                        break
                else:
                    m = self.s[i].M

                if m == self.s[i].M:
                    m -= 1
                    while m > 0 and self.s[i].c[m] == 0.0:
                        m -= 1

                # Get a random number from the corresponding density function
                sq.seq[n][pos] = self.ghmm_cmodel_get_random_var(i, m, native=native)
                # Decide the clazz for the next step
                if self.cos == 1:
                    clazz = 0
                else:
                    #Log.error("2: cos = %d, k = %d, t = %d\n",self.cos,n,state)
                    if not self.class_change.get_class:
                        Log.error("ERROR: get_class not initialized\n")
                        return None

                    clazz = self.class_change.get_class(self, sq.seq[n], n, pos)
                    Log.error("class = %d\n", clazz)
                    if clazz >= self.cos:
                        Log.error("ERROR: get_class returned index %d but model has only %d classes !\n", clazz, self.cos)
                        return None

                up = 0
                pos += 1
                # while (state < len)
            if badseq:
                reject_os_tmp += 1

            if stillbadseq:
                reject_os += 1
            elif pos > Tmax:
                reject_tmax += 1
            else:
                if pos < len:
                    sq.seq[n] = ARRAY_REALLOC(sq.seq[n], pos)
                sq.seq_len[n] = pos
                # ighmm_cvector_print(stdout, sq.seq[n], sq.seq_len[n]," "," ","")
                n += 1

            # Log.error("reject_os %d, reject_tmax %d\n", reject_os, reject_tmax)

            if reject_os > 10000:
                Log.note("Reached max. no. of rejections\n")
                break

            if not n % 1000:
                Log.error("%d Seqs. generated\n", n)
                # n-loop

        if reject_os > 0:
            Log.error("%d sequences rejected (os)!\n", reject_os)
        if reject_os_tmp > 0:
            Log.error("%d sequences changed class\n", reject_os_tmp - reject_os)
        if reject_tmax > 0:
            Log.error("%d sequences rejected (Tmax, %d)!\n", reject_tmax, Tmax)
            # Log.error ("End of selfdel_generate_sequences.\n")

        return sq


    def getStateName(self, index):
        try:
            return self.s[index].desc
        except Exception:
            return None

    def getEmission(self, index):
        return self.s[index].e

    def check(self):
        sum_ = 0.0

        for i in range(self.N):
            sum_ += self.s[i].pi

        if abs(sum_ - 1.0) >= GHMM_EPS_PREC:
            Log.error("sum Pi[i] != 1.0\n")

        # only 0/1 in fix?
        for i in range(self.N):
            if self.s[i].fix != 0 and self.s[i].fix != 1:
                Log.error("in vector fix_state only 0/1 values!\n")

        for i in range(self.N):
            # sum  a[i][k][j]
            for k in range(self.cos):
                sum_ = sum(self.s[i].out_a[k])

                if abs(sum_ - 1.0) >= GHMM_EPS_PREC:
                    Log.error("sum out_a[j] = %.4f != 1.0 (state %d, class %d)\n", sum_, i, k)


            # sum_ c[j]
            sum_ = sum(self.s[i].c)
            if abs(sum_ - 1.0) >= GHMM_EPS_PREC:
                Log.error("sum c[j] = %.2f != 1.0 (state %d)\n", sum_, i)

        # do all emissions have the same dimension as specified in smo.dim
        for i in range(self.N):
            for k in range(self.s[i].M):
                if self.dim != self.s[i].e[k].dimension:
                    Log.error("dim s[%d].e[%d] != dimension of model\n", i, k)

    def get_transition(self, i, j, c=None):
        if self.s and self.s[i].out_a and self.s[j].in_a:
            for out in range(self.N):
                if c is None:
                    return self.s[i].out_a[j]
                else:
                    return self.s[i].out_a[c][j]

        return 0.0

    def set_transition(self, i, j, c, value):
        if self.s and self.s[i].out_a and self.s[j].in_a:
            for out in range(self.N):
                self.s[i].out_a[c][j] = value

        return 0.0

    def check_transition(smo, i, j, c=None):
        try:
           return smo.s[i].out_a[c][j] > 0.0
        except Exception, e:
            raise e

    def logp_joint(self, O, len, S, slen):
        state_pos = 0
        pos = 0
        osc = 0
        dim = self.dim

        prevstate = state = S[0]
        log_p = Math.log(self.s[state].pi)
        if not (self.model_type & kSilentStates) or 1: # XXX not mo.silent[state]  :
            log_p += Math.log(self.s[state].calc_b(O[pos]))
            pos += 1

        for state_pos in range(1, len):
            state = S[state_pos]
            if self.cos > 1:
                if not self.class_change.get_class:
                    Log.error("get_class not initialized")

                osc = self.class_change.get_class(self, O, self.class_change.k, pos)
                if osc >= self.cos:
                    Log.error("get_class returned index %d but model has only %d classes!", osc, self.cos)

            if abs(self.s[state].in_a[osc][prevstate]) < GHMM_EPS_PREC:
                Log.error("Sequence can't be built. There is no transition from state %d to %d.", prevstate, state)

            log_p += Math.log(self.s[state].in_a[osc][prevstate])

            if not (self.model_type & kSilentStates) or 1: # XXX !mo.silent[state]
                log_p += Math.log(self.s[state].calc_b(O[pos]))
                pos += 1

            prevstate = state

        if pos < len:
            Log.note("state sequence too shortnot  processed only %d symbols", pos)
        if state_pos < slen:
            Log.note("sequence too shortnot  visited only %d states", state_pos)

        return log_p


    def forward(self, O, T, b, alpha, scale):
        t = 0
        osc = 0

        # T is length of sequence divide by dimension to represent the number of time points
        T /= self.dim
        # calculate alpha and scale for t = 0
        if b == None:
            sfoba_initforward(self, alpha[0], O[0], scale, None)
        else:
            sfoba_initforward(self, alpha[0], O[0], scale, b[0])

        if scale[0] <= DBL_MIN:
            Log.error(" means f(O[0], mue, u) << 0, first symbol very unlikely")

        log_p = Math.log(scale[0])

        if self.cos == 1:
            osc = 0
        else:
            if not self.class_change.get_class:
                Log.error("get_class not initialized\n")

            # printf("1: cos = %d, k = %d, t = %d\n",smo.cos,smo.class_change.k,t)
            osc = self.class_change.get_class(self, O, self.class_change.k, t)
            if osc >= self.cos:
                Log.error("get_class returned index %d but model has only %d classes! \n", osc, self.cos)

        for t in range(1, T):
            scale[t] = 0.0
            # b not calculated yet
            if b == None:
                for i in range(self.N):
                    alpha[t][i] = sfoba_stepforward(self.s[i], alpha[t - 1], osc, self.s[i].calc_b(O[t]))
                    scale[t] += alpha[t][i]

            # b precalculated
            else:
                for i in range(self.N):
                    alpha[t][i] = sfoba_stepforward(self.s[i], alpha[t - 1], osc, b[t][i][self.M])
                    scale[t] += alpha[t][i]

            if scale[t] <= DBL_MIN:        #
                Log.error(" seq. can't be build")

            c_t = 1 / scale[t]
            # scale alpha
            for i in range(self.N):
                alpha[t][i] *= c_t
                # summation of Math.log(c[t]) for calculation of Math.log( P(O|lambda) )
            log_p -= Math.log(c_t)

            if self.cos == 1:
                osc = 0

            else:
                if not self.class_change.get_class:
                    Log.error("get_class not initialized\n")

                # printf("1: cos = %d, k = %d, t = %d\n",smo.cos,smo.class_change.k,t)
                osc = self.class_change.get_class(self, O, self.class_change.k, t)
                if osc >= self.cos:
                    Log.error("get_class returned index %d but model has only %d classes! \n", osc, self.cos)
        return log_p


    def backward(self, O, T, b, beta, scale):
        # T is length of sequence divide by dimension to represent the number of time points
        T /= self.dim

        beta_tmp = ARRAY_CALLOC(self.N)

        for t in range(T):
            # try differenent bounds here in case of problems
            #       like beta[t] = NaN
            if scale[t] < LOWER_SCALE_BOUND:
                Log.error("error")

        # initialize
        c_t = 1 / scale[T - 1]
        for i in range(self.N):
            beta[T - 1][i] = 1
            beta_tmp[i] = c_t

        # Backward Step for t = T-2, ..., 0
        # beta_tmp: Vector for storage of scaled beta in one time step

        if self.cos == 1:
            osc = 0

        else:
            if not self.class_change.get_class:
                Log.error("get_class not initialized\n")

            osc = self.class_change.get_class(self, O, self.class_change.k, T - 2)
            # printf("osc(%d) = %d\n",T-2,osc)
            if osc >= self.cos:
                Log.error("get_class returned index %d but model has only %d classes not \n", osc, self.cos)

        for t in reversed(range(T - 1)):
            if b == None:
                for i in range(self.N):
                    sum_ = 0.0
                    for j in range(self.N):
                        sum_ += self.s[i].out_a[osc][j] * self.s[j].calc_b(O[t+1]) * beta_tmp[j]

                    beta[t][i] = sum_

            else:
                for i in range(self.N):
                    sum_ = 0.0
                    for j in range(self.N):
                        sum_ += self.s[i].out_a[osc][j] * b[t + 1][j][self.M] * beta_tmp[j]

                        #printf("  smo.s[%d].out_a[%d][%d] * b[%d] * beta_tmp[%d] = %f * %f *
                        #            %f\n",i,osc,j,t+1,j_id,smo.s[i].out_a[osc][j], b[t + 1][j_id][smo.M], beta_tmp[j_id])

                    beta[t][i] = sum_
                    # printf(" .   beta[%d][%d] = %f\n",t,i,beta[t][i])

            c_t = 1 / scale[t]
            for i in range(self.N):
                beta_tmp[i] = beta[t][i] * c_t

            if self.cos == 1:
                osc = 0

            else:
                if not self.class_change.get_class:
                    Log.error("get_class not initialized\n")

                # if t=1 the next iteration will be the last
                if t >= 1:
                    osc = self.class_change.get_class(self, O, self.class_change.k, t - 1)
                    # printf("osc(%d) = %d\n",t-1,osc)
                    if osc >= self.cos:
                        Log.error("get_class returned index %d but model has only %d classes not \n", osc, self.cos)


    def normalize(self):
    # Scales the output and transitions probs of all states in a given model
        pi_sum = 0.0

        for i in range(self.N):
            if self.s[i].pi >= 0.0:
                pi_sum += self.s[i].pi
            else:
                self.s[i].pi = 0.0

            # normalize transition probabilities
            for c in range(self.cos):
                ighmm_cvector_normalize(self.s[i].out_a[c], 0, self.N)

                # for every outgoing probability update the corrosponding incoming probability
                for j in range(self.N):
                    self.s[j].in_a[c][i] = self.s[i].out_a[c][j]

        for i in range(self.N):
            self.s[i].pi /= pi_sum

    def logp(self, O, T):
        alpha = ighmm_cmatrix_stat_alloc(T, self.N)
        scale = ARRAY_CALLOC(T)
        # run forward alg.
        return self.forward(O, T, None, alpha, scale)


