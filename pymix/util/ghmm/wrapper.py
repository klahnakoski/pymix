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
from math import exp
from pyLibrary.maths import Math
from pymix.util.logs import Log


DBL_MAX = 1e300
EPS_ITER_BW = 0.0001
MAX_ITER_BW = 500
SMO_FILE_SUPPORT = 0
ASCI_SEQ_FILE = 0
SEQ_LABEL_FIELD = 0
GHMM_MAX_SEQ_LEN = 1000000
DBL_MIN = 1e-15
GHMM_EPS_PREC = 1e-8
GHMM_PENALTY_LOGP = -500.0



binormal = 5      #< two dimensional gaussian
multinormal = 6   #< multivariate gaussian
density_number = 7 #< number of density types, has to stay last





def fake_random_number_generator():
    pass


GHMM_MAX_SEQ_NUMBER = 1500000

def int_array_getitem(array, index):
    return array[index]


def ARRAY_REALLOC(array, n):
    return array[0:n]


def ARRAY_CALLOC(n):
    return [0] * n


def ARRAY_MALLOC(n):
    return [0] * n





def ghmm_xmlfile_validate(filename):
    pass


def double_matrix_alloc(rows, cols):
    try:
        return [[0.0] * cols for r in range(rows)]
    except Exception, e:
        Log.error("not expected", e)


ighmm_dmatrix_stat_alloc = double_matrix_alloc
ighmm_cmatrix_alloc = double_matrix_alloc


def ighmm_cmatrix_stat_alloc(n, m):
    return [[0.0] * m for i in range(n)]


def ighmm_dmatrix_alloc(n, m):
    return [[0.0] * m for i in range(n)]


def matrix_alloc(n, m):
    return [[0.0] * m for i in range(n)]


def int_matrix_alloc_row(rows):
    return [0] * rows


def double_array_alloc(length):
    return [0.0] * length


def double_array2list(array, index):
    return array


def double_matrix2list(array, length, N):
    if length != N:
        Log.error("not expected, look at C source")
    return array


def list2double_array(array, length=None):
    return array


def list2double_matrix(array, cols):
    rows = len(array) / cols
    output = matrix_alloc(rows, cols)
    for r in range(rows):
        for c in range(cols):
            output[r][c] = array[r * cols + c]
    return output


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


def dpstate_array_getitem(states, i):
    return states[i]


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
    def __init__(self, length, number_of_alphabets, number_of_d_seqs):
        ## for each alphabet in model.number_of_alphabets there is one int seq *
        self.seq = ighmm_dmatrix_alloc(number_of_alphabets, length) #int**
        ## number of alphabets (same as in model) *
        self.number_of_alphabets = number_of_alphabets
        ## for each sequence position there are also double values (e.g) Ka *
        self.d_value = self.d_value = ighmm_cmatrix_alloc(number_of_d_seqs, length) #double**
        ## number of continous sequences *
        self.number_of_d_seqs = number_of_d_seqs
        ## length of the sequence *
        self.length = length

    def get_discrete(self, index):
        return self.seq[index]


    def get_continuous(self, index):
        return self.d_value[index]



class ghmm_dsmodel():
    def __init__(self):
        # Number of states
        self.N = 0  # int
        # Number of outputs
        self.M = 0  # int
        # ghmm_dsmodel includes continuous model with one transition matrix
        # (cos  is set to 1) and an extension for models with several matrices
        # (cos is set to a positive integer value > 1).
        self.cos = 0  # int
        # Vector of the states
        self.s = None  # ghmm_dsstate *
        # Prior for the a priori probability for the model.
        # A value of -1 indicates that no prior is defined.
        self.prior = 0.0  # double

        # contains a arbitrary name for the model (null terminated utf-8)
        self.name = None  # char *

        # pointer to class function
        self.get_class = None  # int (*get_class) (int *, int)

        # Contains bit flags for various model extensions such as
        # kSilentStates, kTiedEmissions (see ghmm.h for a complete list)
        self.model_type = 0  # int

        # Flag variables for each state indicating whether it is emitting
        # or not.
        # Note: silent != NULL iff (model_type & kSilentStates) == 1
        self.silent = None  # int *

        # Int variable for the maximum level of higher order emissions
        self.maxorder = 0  # int
        # saves the history of emissions as int,
        # the nth-last emission is (emission_history * |alphabet|^n+1) % |alphabet|
        # see ...
        self.emission_history = 0  # int

        # Flag variables for each state indicating whether the states emissions
        # are tied to another state. Groups of tied states are represented
        # by their tie group leader (the lowest numbered member of the group).
        # tied_to[s] == kUntied  : s is not a tied state
        # tied_to[s] == s        : s is a tie group leader
        # tied_to[t] == s        : t is tied to state s (t>s)
        # Note: tied_to != NULL iff (model_type & kTiedEmissions) != 0
        self.tied_to = None  # int *

        # Note: State store order information of the emissions.
        # Classic HMMS have emission order 0, that is the emission probability
        # is conditioned only on the state emitting the symbol.

        # For higher order emissions, the emission are conditioned on the state s
        # as well as the previous emission_order[s] observed symbols.

        # The emissions are stored in the state's usual double* b. The order is
        # set order.

        # Note: order != NULL iff (model_type & kHigherOrderEmissions) != 0
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
        # Note: background_id != NULL iff (model_type & kHasBackgroundDistributions) != 0
        self.background_id = None  # int *
        self.bp = None  # ghmm_dbackground *

        # (WR) added these variables for topological ordering of silent states
        # Condition: topo_order != NULL iff (model_type & kSilentStates) != 0
        self.topo_order = None  # int *
        self.topo_order_length = 0  # int

        # Store for each state a class label. Limits the possibly state sequence

        # Note: label != NULL iff (model_type & kLabeledStates) != 0
        self.label = None  # int*
        self.label_alphabet = None  # ghmm_alphabet*

        self.alphabet = None  # ghmm_alphabet*


def ighmm_cmatrix_normalize(matrix, rows, cols):
    # Scales the row vectors of a matrix to have the sum 1
    for i in range(rows):
        ighmm_cvector_normalize(matrix[i], 0, cols)


def ighmm_cvector_normalize(v, start, len):
    """
    Scales the elements of a vector to have the sum 1
    PROBLEM: Entries can get very small and be rounded to 0
    """
    sum = 0.0

    for i in range(start, start+len):
        sum += v[i]
    if len > 0 and sum < GHMM_EPS_PREC:
        Log.error("Can't normalize vector. Sum smaller than %g\n", GHMM_EPS_PREC)
    for i in range(start, start+len):
        v[i] /= sum


class ghmm_cmodel_baum_welch_context():
#  Baum-Welch-Algorithm for parameter reestimation (training) in
#    a continuous (continuous output functions) HMM. Scaled version
#    for multiple sequences. Sequences may carry different weights
#    For reference see:
#    Rabiner, L.R.: "`A Tutorial on Hidden :Markov Models and Selected
#                Applications in Speech Recognition"', Proceedings of the IEEE,
#        77, no 2, 1989, pp 257--285
#

#  structure that combines a continuous model sequence class.
#
#    Is used by ghmm_cmodel_baum_welch() for parameter reestimation.
#
    def __init__(self, model, seq):
        #  pointer of continuous model
        self.smo = model
        #  sequence pointer
        self.sqd = seq
        #  calculated log likelihood
        self.logp = 0
        #  leave reestimation loop if diff. between successive logp values
        #      is smaller than eps
        self.eps = 0
        #  max. no of iterations
        self.max_iter = 0


def ighmm_cvector_log_sum(a, length):
    max = 1.0
    argmax = 0

    # find maximum value in a:
    for i in range(0, length):
        if max == 1.0 or (a[i] > max and a[i] != 1.0):
            max = a[i]
            argmax = i


    # calculate max+Math.log(1+sum[i!=argmax exp(a[i]-max)])
    result = 1.0
    for i in range(0, length):
        if a[i] != 1.0 and i != argmax:
            result += exp(a[i] - max)

    result = Math.log(result)
    result += max
    return result


def sequence_max_symbol(sq):
    max_symb = -1
    for i in range(sq.seq_number):
        for j in range(sq.seq_len[i]):
            if sq.seq[i][j] > max_symb:
                max_symb = sq.seq[i][j]

    return max_symb


