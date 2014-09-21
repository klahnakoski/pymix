## encoding: utf-8
################################################################################
#
# This file is part of the Modified Python Mixture Package, original
# source code is from https://svn.code.sf.net/p/pymix/code.  Also see
# http://www.pymix.org/pymix/index.php?n=PyMix.Download
#
# Changes made by: Kyle Lahnakoski (kyle@lahnakoski.com)
#
################################################################################
from math import sqrt
from vendor.pyLibrary.env.logs import Log
from vendor.pyLibrary.maths.randoms import Random


kNotSpecified = 0
kLeftRight = 1
kSilentStates = pow(2, 2)
kTiedEmissions = pow(2, 3)
kUntied = -1
kHigherOrderEmissions = pow(2, 4)
kBackgroundDistributions = pow(2, 5)
kNoBackgroundDistribution = -1
kLabeledStates = pow(2, 6)
kTransitionClasses = pow(2, 7)
kDiscreteHMM = pow(2, 8)
kContinuousHMM = pow(2, 9)
kPairHMM = pow(2, 10)
kMultivariate = pow(2, 11)

EPS_ITER_BW = 0.0001
MAX_ITER_BW = 500
SMO_FILE_SUPPORT = 0
ASCI_SEQ_FILE = 0
SEQ_LABEL_FIELD = 0
GHMM_MAX_SEQ_LEN = 1000000
DBL_MIN = 0.0000001


normal=0,        #< gaussian */
normal_right=1  #< right tail */
normal_approx=2 #< approximated gaussian */
normal_left=3   #< left tail */
uniform=4
binormal=5      #< two dimensional gaussian */
multinormal=6   #< multivariate gaussian */
density_number=7 #< number of density types, has to stay last */

def fake_random_number_generator():
    pass
RNG = fake_random_number_generator


def int_array_getitem(array, index):
    return array[index]

def ARRAY_REALLOC(pointer, bytes):
    pass

def ARRAY_CALLOC(pointer, bytes):
    pass

def set_pylogging(logwrapper):
    pass


def ghmm_rng_init():
    pass


def time_seed():
    pass


def double_matrix_alloc(rows, cols):
    return [[0.0] * cols] * rows


def int_matrix_alloc_row(rows):
    return [0] * rows


def double_array_alloc(length):
    return [0.0] * length


def double_array2list(array, length):
    return array


def double_matrix2list(array, length, N):
    if length != N:
        Log.error("not expected, look at C source")
    return array


def list2double_array(array, length=None):
    return array


def double_matrix2list(array, a, b):
    return array


def double_matrix_get_col(cmatrix, i):
    return cmatrix[i]


def int_array2list(array, length):
    return array


def int_array_alloc(length):
    return [0] * length


def int_array_setitem(array, index, val):
    array[index] = val


def int_matrix_get_col(array, index):
    return array[index]


def double_array_getitem(array, index):
    return array[index]


def list2int_array(array):
    return array


def dseq_ptr_array_getitem(array, index):
    return array[index]


def dpstate_array_getitem(states, i):
    return states[i]


def cseq_ptr_array_getitem(array, index):
    return array[index]


def double_matrix_alloc_row(length):
    return [0.0] * length


def int_matrix_set_col(array, index, value):
    array[index] = value


def double_matrix_set_col(array, index, value):
    array[index] = value


def double_matrix_setitem(matrix, row, col, value):
    matrix[row][col] = value


def set_2d_arrayd(matrix, row, col, value):
    matrix[row][col] = value


def double_matrix_getitem(matrix, row, col):
    return matrix[row][col]


def double_array_setitem(array, index, value):
    array[index] = value


def free(cscale):
    pass


def double_matrix_free(calpha, t):
    pass


# if the sum of the ka values is less than the threshold return 1
def lt_sum(
    mo, #ghmm_dpmodel *
    X, #ghmm_dpseq *
    Y, #ghmm_dpseq *
    index_x,
    index_y,
    user_data
):
    # cast the user data
    td = user_data
    if ghmm_dpseq_get_double(X, td.seq_index, index_x + td.offset_x) + ghmm_dpseq_get_double(Y, td.seq_index, index_y + td.offset_y) < td.threshold:
        return 1
    else:
        return 0

# if the sum of the ka values is less than the threshold return 1
def lt_sum(
    mo, #ghmm_dpmodel *
    X, #ghmm_dpseq *
    Y, #ghmm_dpseq *
    index_x,
    index_y,
    user_data
):
    # cast the user data
    td = user_data
    if ghmm_dpseq_get_double(X, td.seq_index, index_x + td.offset_x) + ghmm_dpseq_get_double(Y, td.seq_index, index_y + td.offset_y) < td.threshold:
        return 1
    else:
        return 0

# if the sum of the ka values is greater than the threshold return 1
def gt_sum(
    mo, #ghmm_dpmodel *
    X, #ghmm_dpseq *
    Y, #ghmm_dpseq *
    index_x,
    index_y,
    user_data
):
    # cast the user data
    td = user_data
    if ghmm_dpseq_get_double(X, td.seq_index, index_x + td.offset_x) + ghmm_dpseq_get_double(Y, td.seq_index, index_y + td.offset_y) > td.threshold:
        return 1
    else:
        return 0


def boolean_or(
    mo, #ghmm_dpmodel *
    X, #ghmm_dpseq *
    Y, #ghmm_dpseq *
    index_x,
    index_y,
    user_data
):
    # cast the user data
    td = user_data
    if ghmm_dpseq_get_double(X, td.seq_index, index_x + td.offset_x) or ghmm_dpseq_get_double(Y, td.seq_index, index_y + td.offset_y):
        return 1
    else:
        return 0


def boolean_and(
    mo, #ghmm_dpmodel *
    X, #ghmm_dpseq *
    Y, #ghmm_dpseq *
    index_x,
    index_y,
    user_data
):
    # cast the user data
    td = user_data
    if ghmm_dpseq_get_double(X, td.seq_index, index_x + td.offset_x) and ghmm_dpseq_get_double(Y, td.seq_index, index_y + td.offset_y):
        return 1
    else:
        return 0


def ghmm_dpseq_get_double(seq_pointer, seq_index, index):
    return seq_pointer.d_value[seq_index][index]


def ghmm_dpmodel_default_transition_class(mo, X, Y, index_x, index_y, user_data):
    return 0


def ghmm_dp_set_to_default_transition_class(pccc):
    pccc.get_class = ghmm_dpmodel_default_transition_class
    pccc.user_data = None


def set_to_lt_sum(pccc, seq_index, threshold, offset_x, offset_y):
    td = threshold_user_data()
    td.seq_index = seq_index
    td.threshold = threshold
    td.offset_x = offset_x
    td.offset_y = offset_y
    pccc.user_data = td
    pccc.get_class = lt_sum


def set_to_gt_sum(pccc, seq_index, threshold, offset_x, offset_y):
    td = threshold_user_data()
    td.seq_index = seq_index
    td.threshold = threshold
    td.offset_x = offset_x
    td.offset_y = offset_y
    pccc.user_data = td
    pccc.get_class = gt_sum


def set_to_boolean_and(pccc, seq_index, offset_x, offset_y):
    td = threshold_user_data()
    td.seq_index = seq_index
    td.offset_x = offset_x
    td.offset_y = offset_y
    pccc.user_data = td
    pccc.get_class = boolean_and


def set_to_boolean_or(pccc, seq_index, offset_x, offset_y):
    td = threshold_user_data()
    td.seq_index = seq_index
    td.offset_x = offset_x
    td.offset_y = offset_y
    pccc.user_data = td
    pccc.get_class = boolean_or


class ghmm_alphabet:
    def __init__(self, length, description):
        self.id = 0
        self.description = description
        self.size = length
        self.symbols = [None] * length


    def setSymbol(self, index, value):
        self.symbols[index] = str(value)

# class ghmm_dpmodel_class_change_context:
#
# def __init(self):
# # Names of class change module/function (for python callback)
# python_module=""
# python_function=""
#
# # pointer to class function called with seq X, Y and resp indices
# # in the void you can pass the user data
# int (*get_class)(struct ghmm_dpmodel*, ghmm_dpseq*, ghmm_dpseq*, int, int,void*)
#
# # space for any data necessary for class switch, USER is RESPONSIBLE
# void* user_data
#

class threshold_user_data():
    def __init__(self):
        # which double value in myseq
        self.seq_index = 0
        # cut off value
        self.threshold = 0.0
        # add this to the index in sequence X
        self.offset_x = 0
        # add this to the index in sequence Y
        self.offset_y = 0


class ghmm_dpseq():
    def __init__(self):
        ## for each alphabet in model.number_of_alphabets there is one int seq *
        self.seq = None #int**
        ## number of alphabets (same as in model) *
        self.number_of_alphabets = 0
        ## for each sequence position there are also double values (e.g) Ka *
        self.d_value = None #double**
        ## number of continous sequences *
        self.number_of_d_seqs = 0
        ## length of the sequence *
        self.length = 0


class ghmm_dseq():
    def __init__(self):
        # sequence array. sequence[i] [j] = j-th symbol of i-th seq.
        self.seq = None #int **

        # matrix of state ids, can be used to save the viterbi path during sequence generation.
        # ATTENTION: is NOT allocated by ghmm_dseq_calloc
        self.states = None  #int **

        # array of sequence length
        self.seq_len = 0
        # array of state path lengths
        self.states_len = []

        ## array of sequence IDs
        self.seq_id = 0.0 #double *
        ## positiv! sequence weights.  default is 1 = no weight
        self.seq_w = 0.0 #double *
        ## total number of sequences
        self.seq_number = 0
        ## reserved space for sequences is always >= seq_number
        self.capacity = 0
        ## sum of sequence weights
        self.total_w = 0

        ## matrix of state labels corresponding to seq
        self.state_labels = None #int **
        ## number of labels for each sequence
        self.state_labels_len = None # int*

        ## flags internal
        self.flags = 0


    def init_labels(self, labels, length):
        self.state_labels = labels
        self.state_labels_len = length


class ghmm_cseq():
    """
    Sequence structure for double sequences.

    Contains an array of sequences and corresponding
    data like sequnce label, sequence weight, etc. Sequences may have
    different length. Multi-dimension sequences are linearized.
    """

    def __init__(self):
        # sequence array. sequence[i][j] = j-th symbol of i-th seq.
        # sequence[i][D * j] = first dimension of j-th observation of i-th sequence
        self.seq = None #double**
        # array of sequence length
        self.seq_len = 0 #int*
        # array of sequence IDs
        self.seq_id = 0 #double*
        # positive! sequence weights.  default is 1 = no weight
        self.seq_w = 0.0 #double*
        # total number of sequences
        self.seq_number = 0 #long
        # reserved space for sequences is always >= seq_number
        self.capacity = 0 #long
        # sum of sequence weights
        self.total_w = 0.0 #double
        # total number of dimensions
        self.dim = 0 #int

        # flags (internal)
        self.flags = 0 #int

    def init_labels(self, labels, length):
        self.state_labels = labels
        self.state_labels_len = length


class ghmm_dstate():
    def __init__(self):
        # Initial probability */
        self.pi = 0.0 #double
        # Output probability */
        self.b = 0.0 #double*

        # IDs of the following states */
        self.out_id = 0 #int*
        # IDs of the previous states */
        self.in_id = 0 #int*

        # transition probabilities to successor states. */
        self.out_a = 0.0 #double*
        # transition probabilities from predecessor states. */
        self.in_a = 0.0 #double*

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

        # contains a arbitrary name for the model (null terminated utf-8) */
        self.name = "" #char*

        # Contains bit flags for varios model extensions such as
        # kSilentStates, kTiedEmissions (see ghmm.h for a complete list)
        # */
        self.model_type = model_type #int

        # Flag variables for each state indicating whether it is emitting
        # or not.
        # Note: silent != NULL iff (model_type & kSilentStates) == 1  */

        if self.model_type & kSilentStates:
            self.silent = [0]*N
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
        # Note: tied_to != NULL iff (model_type & kTiedEmissions) != 0  */
        if (self.model_type & kTiedEmissions) :
            self.tied_to=[kUntied]*N
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
        # Note: order != NULL iff (model_type & kHigherOrderEmissions) != 0  */
        if self.model_type & kHigherOrderEmissions:
            self.order=[0]*N
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
        # Note: background_id != NULL iff (model_type & kHasBackgroundDistributions) != 0  */
        if self.model_type & kBackgroundDistributions:
            self.background_id = [kNoBackgroundDistribution]*N
        else:
            self.background_id = None #int*

        self.bp = None #ghmm_dbackground*

        # (WR) added these variables for topological ordering of silent states
        # Condition: topo_order != NULL iff (model_type & kSilentStates) != 0  */
        self.topo_order = None #int*
        self.topo_order_length = 0 #int

        # pow_lookup is a array of precomputed powers
        # It contains in the i-th entry M (alphabet size) to the power of i
        # The last entry is maxorder+1 */
        self.pow_lookup = None #int*

        # Store for each state a class label. Limits the possibly state sequence
        #
        # Note: label != NULL iff (model_type & kLabeledStates) != 0  */
        if self.model_type & kLabeledStates:
            self.label=[0]*N
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
                GHMM_RNG_SET(RNG, seed)
            else:                        # Random initialization! */
                ghmm_rng_timeseed(RNG)

        n = 0
        reject_os = reject_tmax = 0

        # calculate cholesky decomposition of covariance matrix (multivariate case),
        # this needs to be called before ghmm_c_get_random_var */
        if self.dim > 1:
            for i in range(self.N):
                for m in range(self.s[i].M):
                    self.s[i].e[m].sigmacd = ighmm_cholesky_decomposition(self.dim, self.s[i].e[m].variance.mat)

        while n < seq_number:
            # Test: A new seed for each sequence */
            # ghmm_rng_timeseed(RNG) */
            stillbadseq = badseq = 0
            ARRAY_CALLOC(sq.seq[n], len * (self.dim))

            # Get a random initial state i */
            p = GHMM_RNG_UNIFORM(RNG)
            sum = 0.0
            for i in range(self.N):
                sum += self.s[i].pi
                if sum >= p:
                    break
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

            if m == self.s[i].M:
                m -= 1

            # Get random numbers according to the density function */
            ghmm_cmodel_get_random_var(self, i, m, sq.seq[n] + 0)
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

                if m == self.s[i].M:
                    m -= 1
                    while m > 0 and self.s[i].c[m] == 0.0:
                        m -= 1

                # Get a random number from the corresponding density function */
                ghmm_cmodel_get_random_var(self, i, m, sq.seq[n] + (pos * self.dim))
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
                    ARRAY_REALLOC(sq.seq[n], pos)
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


def model_state_alloc(
    M,
    in_states,
    out_states
):
    assert isinstance(s, ghmm_dstate)

    s = ghmm_dstate()
    s.b = [0.0] * M
    s.out_id = [0] * out_states
    s.out_a = [0.0] * out_states

    s.in_id = [0] * in_states
    s.in_a = [0.0] * in_states

    return s


def ighmm_cholesky_decomposition (dim, cov):
  sigmacd = [row.copy() for row in cov]

  for row in range(dim):
    # First compute U[row][row] */
    total = cov[row][row]
    for j in range(row-1):
        total -= sigmacd[j][row]*sigmacd[j][row]
    if total > DBL_MIN:
      sigmacd[row][row] = sqrt(total)
      # Now find elements sigmacd[row*dim+k], k > row. */
      for k in range(row+1, dim):
        total = cov[row][k]
        for j in range(0, row-1):
          total -= sigmacd[j][row]*sigmacd[j][k]
        sigmacd[row][k] = total/sigmacd[row][row]

    else:  # blast off the entire row. */
      for k in range(row, dim):
          sigmacd[row][k] = 0.0
  return sigmacd


def GHMM_RNG_UNIFORM(rng):
    return Random.float()


class ghmm_dsmodel():

    def __init__(self):
        # Number of states */
        self.N = 0  # int
        # Number of outputs */
        self.M = 0  # int
        # ghmm_dsmodel includes continuous model with one transition matrix
        # (cos  is set to 1) and an extension for models with several matrices
        # (cos is set to a positive integer value > 1).*/
        self.cos = 0  # int
        # Vector of the states */
        self.s = None  # ghmm_dsstate *
        # Prior for the a priori probability for the model.
        # A value of -1 indicates that no prior is defined. */
        self.prior = 0.0  # double

        # contains a arbitrary name for the model (null terminated utf-8) */
        self.name = None  # char *

        # pointer to class function   */
        self.get_class = None  # int (*get_class) (int *, int)

        # Contains bit flags for various model extensions such as
        # kSilentStates, kTiedEmissions (see ghmm.h for a complete list)
        self.model_type = 0  # int

        # Flag variables for each state indicating whether it is emitting
        # or not.
        # Note: silent != NULL iff (model_type & kSilentStates) == 1  */
        self.silent = None  # int *

        # Int variable for the maximum level of higher order emissions */
        self.maxorder = 0  # int
        # saves the history of emissions as int,
        # the nth-last emission is (emission_history * |alphabet|^n+1) % |alphabet|
        # see ...*/
        self.emission_history = 0  # int

        # Flag variables for each state indicating whether the states emissions
        # are tied to another state. Groups of tied states are represented
        # by their tie group leader (the lowest numbered member of the group).
        # tied_to[s] == kUntied  : s is not a tied state
        # tied_to[s] == s        : s is a tie group leader
        # tied_to[t] == s        : t is tied to state s (t>s)
        # Note: tied_to != NULL iff (model_type & kTiedEmissions) != 0  */
        self.tied_to = None  # int *

        # Note: State store order information of the emissions.
        # Classic HMMS have emission order 0, that is the emission probability
        # is conditioned only on the state emitting the symbol.

        # For higher order emissions, the emission are conditioned on the state s
        # as well as the previous emission_order[s] observed symbols.

        # The emissions are stored in the state's usual double* b. The order is
        # set order.

        # Note: order != NULL iff (model_type & kHigherOrderEmissions) != 0  */
        self.order = None  # int *

        # ghmm_dbackground is a pointer to a
        # ghmm_dbackground structure, which holds (essentially) an
        # array of background distributions (which are just vectors of floating
        # point numbers like state.b).
        #
        # For each state the array background_id indicates which of the background
        # distributions to use in parameter estimation. A value of kNoBackgroundDistribution
        # indicates that none should be used.
        #
        # Note: background_id != NULL iff (model_type & kHasBackgroundDistributions) != 0  */
        self.background_id = None  # int *
        self.bp = None  # ghmm_dbackground *

        # (WR) added these variables for topological ordering of silent states
        # Condition: topo_order != NULL iff (model_type & kSilentStates) != 0  */
        self.topo_order = None  # int *
        self.topo_order_length = 0  # int

        # pow_lookup is a array of precomputed powers
        # It contains in the i-th entry M (alphabet size) to the power of i
        # The last entry is maxorder+1 */
        self.pow_lookup = None  # int *

       # Store for each state a class label. Limits the possibly state sequence

          # Note: label != NULL iff (model_type & kLabeledStates) != 0  */
        self.label = None  # int*
        self.label_alphabet = None  # ghmm_alphabet*

        self.alphabet = None  # ghmm_alphabet*

class ghmm_cmodel:

    def __init__(self):

        pass

    def ghmm_cmodel_get_random_var(self, state, m, x):
    # define CUR_PROC "ghmm_cmodel_get_random_var"
      ghmm_c_emission *emission = self.s[state].e + m
      if emission.type in (normal_approx, normal):
        *x = ighmm_rand_normal(emission.mean.val, emission.variance.val, 0)
        return 0
      elif emission.type==binormal:
        #return ighmm_rand_binormal(emission.mean.vec, emission.variance.mat, 0)*/
      elif emission.type==multinormal:
        return ighmm_rand_multivariate_normal(emission.dimension, x,
                                              emission.mean.vec,
                                              emission.sigmacd, 0)
      elif emission.type==normal_right:
        *x = ighmm_rand_normal_right(emission.min, emission.mean.val,
                                     emission.variance.val, 0)
        return 0
      elif emission.type==normal_left:
        *x = -ighmm_rand_normal_right(-emission.max, -emission.mean.val,
                                      emission.variance.val, 0)
        return 0
      elif emission.type==uniform:
        *x = ighmm_rand_uniform_cont(0, emission.max, emission.min)
        return 0
      else:
        Log.error("unknown density function specified!")
        return -1
      }
    # undef CUR_PROC
    }                               # ghmm_cmodel_get_random_var */

