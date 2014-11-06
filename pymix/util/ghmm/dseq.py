from util.ghmm.wrapper import ARRAY_CALLOC, double_array_alloc


class ghmm_dseq():
    def __init__(self, seq):
        if isinstance(seq, int):
            seq = [[]] * seq

        # sequence array. sequence[i] [j] = j-th symbol of i-th seq.
        self.seq = seq  # int **

        # matrix of state ids, can be used to save the viterbi path during sequence generation.
        # ATTENTION: is NOT allocated by ghmm_dseq_calloc
        self.states = [None]*len(seq)  # int **

        # array of sequence length
        self.seq_len = [len(s) for s in seq]
        # array of state path lengths
        self.states_len = double_array_alloc(len(seq))

        ## array of sequence IDs
        self.seq_id = double_array_alloc(len(seq))  # double *
        ## positiv! sequence weights.  default is 1 = no weight
        self.seq_w = [1.0]*len(seq)  # double *
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
        # self.states=ARRAY_CALLOC( self.seq_number)*/
        self.seq_len = ARRAY_CALLOC(self.seq_number)
        self.seq_id = ARRAY_CALLOC(self.seq_number)
        self.seq_w = ARRAY_CALLOC(self.seq_number)

        for i in range(old_seq_number):
            self.seq[i] = old_seq[i]
            # self.states[i] = old_seq_st[i]*/
            self.seq_len[i] = old_seq_len[i]
            self.seq_id[i] = old_seq_id[i]
            self.seq_w[i] = old_seq_w[i]

        for i in range(source.seq_number):
            self.seq[i + old_seq_number] = ARRAY_CALLOC(source.seq_len[i])
            self.seq[i + old_seq_number] = list(source.seq[i])
            self.seq_len[i + old_seq_number] = source.seq_len[i]
            self.seq_id[i + old_seq_number] = source.seq_id[i]
            self.seq_w[i + old_seq_number] = source.seq_w[i]

    def getSymbol(self, seq_num, index):
        return self.seq[seq_num][index]

    def setSymbol(self, seq_num, index, value):
        self.seq[seq_num][index]=value

    def getLabels(self, index):
        return self.state_labels[index]

    def getLabelsLength(self, index):
        return len(self.state_labels[index])

    def getWeight(self, index):
        return self.seq_w[index]

    def setWeight(self, index, value):
        self.seq_w[index]=value


    def get_singlesequence(self, index):
        res = ghmm_dseq([self.seq[index]])

        res.seq_id[0] = self.seq_id[index]
        res.seq_w[0] = self.seq_w[index]
        res.total_w = self.seq_w[index]

        if self.state_labels:
            res.state_labels = [self.state_labels[index]]
            res.state_labels_len = [self.state_labels_len[index]]
        return res

    def write(self, filename):
        pass


    def copyStateLabel(self, index, target, no):
        target.state_labels_len[no] = self.state_labels_len[index]
        target.state_labels[no]=list(self.state_labels[index])



#============================================================================
def ghmm_dseq_add(target, source):
#define CUR_PROC "ghmm_dseq_add"

  old_seq = target.seq
  old_seq_len = target.seq_len
  old_seq_id = target.seq_id
  old_seq_w = target.seq_w
  old_seq_number = target.seq_number

  target.seq_number = old_seq_number + source.seq_number
  target.total_w += source.total_w

  target.seq=ARRAY_CALLOC( target.seq_number)
  #target.states=ARRAY_CALLOC( target.seq_number)
  target.seq_len=ARRAY_CALLOC( target.seq_number)
#ifdef GHMM_OBSOLETE
  target.seq_label=ARRAY_CALLOC( target.seq_number)
#endif # GHMM_OBSOLETE
  target.seq_id=ARRAY_CALLOC( target.seq_number)
  target.seq_w=ARRAY_CALLOC( target.seq_number)

  for i in range( 0,  old_seq_number):
    target.seq[i] = old_seq[i]
    #target.states[i] = old_seq_st[i]
    target.seq_len[i] = old_seq_len[i]
#ifdef GHMM_OBSOLETE
    target.seq_label[i] = old_seq_label[i]
#endif # GHMM_OBSOLETE
    target.seq_id[i] = old_seq_id[i]
    target.seq_w[i] = old_seq_w[i]


  for i in range( 0,  (target.seq_number - old_seq_number)):
    target.seq[i + old_seq_number]=ARRAY_CALLOC( source.seq_len[i])

    ghmm_dseq_copy (target.seq[i + old_seq_number], source.seq[i],
                   source.seq_len[i])

    #target.states[i+old_seq_number]=ARRAY_CALLOC( source.seq_len[i])

    # ghmm_dseq_copy(target.states[i+old_seq_number], source.states[i],
    #       source.seq_len[i])

    target.seq_len[i + old_seq_number] = source.seq_len[i]
#ifdef GHMM_OBSOLETE
    target.seq_label[i + old_seq_number] = source.seq_label[i]
#endif # GHMM_OBSOLETE
    target.seq_id[i + old_seq_number] = source.seq_id[i]
    target.seq_w[i + old_seq_number] = source.seq_w[i]



  m_free (old_seq)
  #m_free(old_seq_st)
  m_free (old_seq_len)
#ifdef GHMM_OBSOLETE
  m_free (old_seq_label)
#endif # GHMM_OBSOLETE
  m_free (old_seq_id)
  m_free (old_seq_w)
  res = 0
STOP:     # Label STOP from ARRAY_[CM]ALLOC
  return res
#undef CUR_PROC





def ghmm_dseq_read():
    pass


