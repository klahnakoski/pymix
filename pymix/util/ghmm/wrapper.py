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


EPS_ITER_BW = 0.0001
MAX_ITER_BW = 500
SMO_FILE_SUPPORT = 0
ASCI_SEQ_FILE = 0
SEQ_LABEL_FIELD = 0
GHMM_MAX_SEQ_LEN = 1000000
DBL_MIN = 0.0000001
GHMM_EPS_PREC = 1e-8

normal = 0,        #< gaussian */
normal_right = 1  #< right tail */
normal_approx = 2 #< approximated gaussian */
normal_left = 3   #< left tail */
uniform = 4
binormal = 5      #< two dimensional gaussian */
multinormal = 6   #< multivariate gaussian */
density_number = 7 #< number of density types, has to stay last */


def fake_random_number_generator():
    pass


RNG = fake_random_number_generator


def int_array_getitem(array, index):
    return array[index]


def ARRAY_REALLOC(n):
    return [0]*n


def ARRAY_CALLOC(n):
    return [0]*n

def ARRAY_MALLOC(n):
    return [0]*n


def ghmm_dseq_read():
    pass

def set_pylogging(logwrapper):
    pass


def GHMM_RNG_SET(rng, seed):
    Random.set_seed(seed)


def ghmm_rng_init():
    pass


def time_seed():
    pass


def ghmm_cseq_read(filename):
    pass

def ghmm_xmlfile_validate(filename):
    pass

def double_matrix_alloc(rows, cols):
    try:
        return [[0.0] * cols for r in range(rows)]
    except Exception, e:
        Log.unexpected(e)


def ighmm_cmatrix_stat_alloc(n, m):
    return [[0.0] * m for i in range(n)]


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

def ghmm_xmlfile_parse(fileName):
    pass

def long_array_getitem(a, i):
    return a[i]


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
    try:
        array[index] = value
    except Exception, e:
        raise e


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

    def getSymbol(self, index):
        return self.symbols[index]
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


class ghmm_cseq():
    """
    Sequence structure for double sequences.

    Contains an array of sequences and corresponding
    data like sequnce label, sequence weight, etc. Sequences may have
    different length. Multi-dimension sequences are linearized.
    """

    def __init__(self, seq):

        # sequence array. sequence[i][j] = j-th symbol of i-th seq.
        # sequence[i][D * j] = first dimension of j-th observation of i-th sequence
        self.seq = seq # int **

        # matrix of state ids, can be used to save the viterbi path during sequence generation.
        # ATTENTION: is NOT allocated by ghmm_dseq_calloc
        self.states = double_array_alloc(len(seq))  # int **

        # array of sequence length
        self.seq_len = [len(s) for s in seq]  # int*

        # array of state path lengths
        self.states_len = double_array_alloc(len(seq))

        ## array of sequence IDs
        self.seq_id = double_array_alloc(len(seq)) # double *
        # positive! sequence weights.  default is 1 = no weight
        self.seq_w = [1.0]*len(seq) # double*
        ## total number of sequences
        self.seq_number = len(seq)
        ## reserved space for sequences is always >= seq_number
        self.capacity = 0
        ## sum of sequence weights
        self.total_w = 0

        ## matrix of state labels corresponding to seq
        self.state_labels = None # int **
        ## number of labels for each sequence
        self.state_labels_len = None # int*

        # flags (internal)
        self.flags = 0 # int




def ighmm_cholesky_decomposition(dim, cov):
    sigmacd = [row.copy() for row in cov]

    for row in range(dim):
        # First compute U[row][row] */
        total = cov[row][row]
        for j in range(row - 1):
            total -= sigmacd[j][row] * sigmacd[j][row]
        if total > DBL_MIN:
            sigmacd[row][row] = sqrt(total)
            # Now find elements sigmacd[row*dim+k], k > row. */
            for k in range(row + 1, dim):
                total = cov[row][k]
                for j in range(0, row - 1):
                    total -= sigmacd[j][row] * sigmacd[j][k]
                sigmacd[row][k] = total / sigmacd[row][row]

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

        # Store for each state a class label. Limits the possibly state sequence

        # Note: label != NULL iff (model_type & kLabeledStates) != 0  */
        self.label = None  # int*
        self.label_alphabet = None  # ghmm_alphabet*

        self.alphabet = None  # ghmm_alphabet*

