from util.ghmm.wrapper import ARRAY_REALLOC, GHMM_RNG_UNIFORM, RNG, GHMM_MAX_SEQ_LEN, ghmm_rng_init, multinormal, binormal, normal, normal_approx, normal_right, normal_left, uniform, ighmm_cholesky_decomposition
from vendor.pyLibrary.env.logs import Log
from vendor.pyLibrary.maths.randoms import Random


class ghmm_cmodel:
    def __init__(self):
        # Number of states */
        self.N = None  # int
        # Maximun number of components in the states */
        self.M = None  # int
        # Number of dimensions of the emission components.
        # All emissions must have the same number of dimensions */
        self.dim = None  # int
        # ghmm_cmodel includes continuous model with one transition matrix
        # (cos  is set to 1) and an extension for models with several matrices
        # (cos is set to a positive integer value > 1).*/
        self.cos = None  # int
        # prior for a priori prob. of the model. -1 means no prior specified (all
        # models have equal prob. a priori. */
        self.prior = None  # double

        # contains a arbitrary name for the model (null terminated utf-8) */

        self.name = None  # char *

        # Contains bit flags for varios model extensions such as
        # kSilentStates (see ghmm.h for a complete list)
        self.model_type = None  # int

        # All states of the model. Transition probs are part of the states. */
        self.s = None  # ghmm_cstate *

        # pointer to a ghmm_cmodel_class_change_context struct necessary for multiple transition
        # classes * /
        self.class_change = None  # ghmm_cmodel_class_change_context *


    def ghmm_cmodel_get_random_var(self, state, m):
        # PARAMETER x IS THE RETURN VALUES
        # define CUR_PROC "ghmm_cmodel_get_random_var"
        emission = self.s[state].e[m]
        if emission.type in (normal_approx, normal):
            return ighmm_rand_normal(emission.mean.val, emission.variance.val, 0)
        elif emission.type == binormal:
            #return ighmm_rand_binormal(emission.mean.vec, emission.variance.mat, 0)*/
            pass
        elif emission.type == multinormal:
            return ighmm_rand_multivariate_normal(emission.dimension, x,
                emission.mean.vec,
                emission.sigmacd, 0)
        elif emission.type == normal_right:
            return ighmm_rand_normal_right(emission.min, emission.mean.val, emission.variance.val, 0)
        elif emission.type == normal_left:
            return -ighmm_rand_normal_right(-emission.max, -emission.mean.val, emission.variance.val, 0)
        elif emission.type == uniform:
            return ighmm_rand_uniform_cont(0, emission.max, emission.min)
        else:
            Log.error("unknown density function specified!")
            return -1

    def generate_sequences(self, seed, global_len, seq_number, Tmax):
        # An end state is characterized by not having an output probabiliy. */

        len = global_len
        up = 0
        stillbadseq = 0
        reject_os_tmp = 0

        sq = ghmm_cseq()

        # set dimension of the sequence to match dimension of the model (multivariate) */
        sq.dim = self.dim

        # A specific length of the sequences isn't given. As a model should have
        # an end state, the konstant MAX_SEQ_LEN is used. */
        if len <= 0:
            len = GHMM_MAX_SEQ_LEN

        # Maximum length of a sequence not given */
        if Tmax <= 0:
            Tmax = GHMM_MAX_SEQ_LEN


        # rng is also used by ighmm_rand_std_normal
        # seed == -1: Initialization, has already been done outside the function */
        if seed >= 0:
            ghmm_rng_init()
            if seed > 0:
                Random.set_seed(seed)
            else:                        # Random initialization! */
                pass

        n = 0
        reject_os = reject_tmax = 0

        # calculate cholesky decomposition of covariance matrix (multivariate case),
        # this needs to be called before ghmm_c_get_random_var */
        if self.dim > 1:
            for i in range(self.N):
                for m in range(self.s[i].M):
                    self.s[i].e[m].sigmacd = ighmm_cholesky_decomposition(self.dim, self.s[i].e[m].variance.mat)
                m = self.s[i].M
            i=self.N
        while n < seq_number:
            # Test: A new seed for each sequence */
            # ghmm_rng_timeseed(RNG) */
            stillbadseq = badseq = 0
            sq.seq[n] = [0] * ( len * (self.dim))

            # Get a random initial state i */
            p = GHMM_RNG_UNIFORM(RNG)
            sum = 0.0
            for i in range(self.N):
                sum += self.s[i].pi
                if sum >= p:
                    break
            else:
                i=self.N
            if i == self.N:          # Can happen by a rounding error in the input */
                i -= 1
                while i > 0 and self.s[i].pi == 0.0:
                    i -= 1


            # Get a random initial output
            # . get a random m and then respectively a pdf omega. */
            p = GHMM_RNG_UNIFORM(RNG)
            sum = 0.0
            for m in range(self.s[i].M):
                sum += self.s[i].c[m]
                if sum >= p:
                    break
            else:
                m=self.s[i].M
            if m == self.s[i].M:
                m -= 1

            # Get random numbers according to the density function */
            self.ghmm_cmodel_get_random_var(i, m, sq.seq[n])
            pos = 1

            # The first symbol chooses the start class */
            if self.cos == 1:
                clazz = 0
            else:
                #Log.error("1: cos = %d, k = %d, t = %d\n",self.cos,n,state)*/

                if not self.class_change.get_class:
                    Log.error("ERROR: get_class not initialized\n")
                    return None

                clazz = self.class_change.get_class(self, sq.seq[n], n, 0)
                if clazz >= self.cos:
                    Log.error("ERROR: get_class returned index %d but model has only %d classes !\n", clazz, self.cos)
                    return None

            while pos < len:
                # Get a new state */
                p = GHMM_RNG_UNIFORM(RNG)
                sum = 0.0
                for j in range(self.s[i].out_states):
                    sum += self.s[i].out_a[clazz][j]
                    if sum >= p:
                        break
                else:
                    j=self.s[i].out_states

                if j == self.s[i].out_states:  # Can happen by a rounding error */
                    j -= 1
                    while j > 0 and self.s[i].out_a[clazz][j] == 0.0:
                        j -= 1

                if sum == 0.0:
                    if self.s[i].out_states > 0:
                        # Repudiate the sequence, if all self.s[i].out_a[clazz][.] == 0,
                        # that is, clazz "clazz" isn't used in the original data:
                        # go out of the while-loop, n should not be counted. */
                        # Log.error("Zustand %d, clazz %d, len %d out_states %d \n", i, clazz,
                        # state, self.s[i].out_states) */
                        badseq = 1
                        # break */

                        # Try: If the clazz is "empty", try the neighbour clazz
                        # first, sweep down to zero if still no success, sweep up to
                        # COS - 1. If still no success -. Repudiate the sequence. */
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
                        # Final state reached, out of while-loop */
                        break

                i = self.s[i].out_id[j]
                # fprintf(stderr, "%d\n", i) */
                # fprintf(stderr, "%d\n", i) */

                # Get output from state i */
                p = GHMM_RNG_UNIFORM(RNG)
                sum = 0.0
                for m in range(self.s[i].M):
                    sum += self.s[i].c[m]
                    if sum >= p:
                        break
                else:
                    m=self.s[i].M

                if m == self.s[i].M:
                    m -= 1
                    while m > 0 and self.s[i].c[m] == 0.0:
                        m -= 1

                # Get a random number from the corresponding density function */
                self.ghmm_cmodel_get_random_var(i, m, sq.seq[n] + (pos * self.dim))
                # Decide the clazz for the next step */
                if self.cos == 1:
                    clazz = 0
                else:
                    #Log.error("2: cos = %d, k = %d, t = %d\n",self.cos,n,state)*/
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
                # while (state < len) */
            if badseq:
                reject_os_tmp += 1

            if stillbadseq:
                reject_os += 1
            elif pos > Tmax:
                reject_tmax += 1
            else:
                if pos < len:
                    sq.seq[n] = ARRAY_REALLOC(sq.seq[n], pos)
                sq.seq_len[n] = pos * (self.dim)
                # ighmm_cvector_print(stdout, sq.seq[n], sq.seq_len[n]," "," ","") */
                n += 1

            # Log.error("reject_os %d, reject_tmax %d\n", reject_os, reject_tmax) */

            if reject_os > 10000:
                Log.note("Reached max. no. of rejections\n")
                break

            if not n % 1000:
                Log.error("%d Seqs. generated\n", n)
                # n-loop */

        if reject_os > 0:
            Log.error("%d sequences rejected (os)!\n", reject_os)
        if reject_os_tmp > 0:
            Log.error("%d sequences changed class\n", reject_os_tmp - reject_os)
        if reject_tmax > 0:
            Log.error("%d sequences rejected (Tmax, %d)!\n", reject_tmax, Tmax)
            # Log.error ("End of selfdel_generate_sequences.\n") */

        return sq
