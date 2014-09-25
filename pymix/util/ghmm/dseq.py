from util.ghmm.wrapper import ARRAY_CALLOC, double_array_alloc


class ghmm_dseq():
    def __init__(self, seq_number):
        # sequence array. sequence[i] [j] = j-th symbol of i-th seq.
        self.seq = double_array_alloc(seq_number) #int **

        # matrix of state ids, can be used to save the viterbi path during sequence generation.
        # ATTENTION: is NOT allocated by ghmm_dseq_calloc
        self.states = double_array_alloc(seq_number)  #int **

        # array of sequence length
        self.seq_len = double_array_alloc(seq_number)
        # array of state path lengths
        self.states_len = double_array_alloc(seq_number)

        ## array of sequence IDs
        self.seq_id = double_array_alloc(seq_number) #double *
        ## positiv! sequence weights.  default is 1 = no weight
        self.seq_w = double_array_alloc(seq_number) #double *
        ## total number of sequences
        self.seq_number = seq_number
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

    def getSequence(self, index):
        return self.seq[index]

    def getLength(self, index):
        return self.seq_len[index]

    def add(self, source):
        old_seq = self.seq
        old_seq_len = self.seq_len
        old_seq_id = self.seq_id
        old_seq_w = self.seq_w
        old_seq_number = self.seq_number

        self.seq_number = old_seq_number + source.seq_number
        self.total_w += source.total_w

        self.seq = ARRAY_CALLOC(self.seq_number)
        #ARRAY_CALLOC (self.states, self.seq_number)*/
        self.seq_len = ARRAY_CALLOC(self.seq_number)
        self.seq_id = ARRAY_CALLOC(self.seq_number)
        self.seq_w = ARRAY_CALLOC(self.seq_number)

        for i in range(old_seq_number):
            self.seq[i] = old_seq[i]
            #self.states[i] = old_seq_st[i]*/
            self.seq_len[i] = old_seq_len[i]
            self.seq_id[i] = old_seq_id[i]
            self.seq_w[i] = old_seq_w[i]

        for i in range(source.seq_number):
            self.seq[i + old_seq_number] = ARRAY_CALLOC(source.seq_len[i])
            self.seq[i + old_seq_number] = list(source.seq[i])
            self.seq_len[i + old_seq_number] = source.seq_len[i]
            self.seq_id[i + old_seq_number] = source.seq_id[i]
            self.seq_w[i + old_seq_number] = source.seq_w[i]


