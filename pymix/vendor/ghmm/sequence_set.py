import os
from string import join
from pymix.util.ghmm import wrapper
from pymix.util.ghmm.sequences import sequence
from pymix.util.ghmm.wrapper import double_matrix_getitem
from pymix.vendor.ghmm import ghmmhelper
from pymix.vendor.ghmm.emission_domain import LabelDomain
from pymix.util.logs import Log


class SequenceSet(object):
    """ A SequenceSet contains the *internal* representation of a number of
    sequences of emissions.

    It also contains a reference to the domain where the emissions orginated from.
    """

    def __init__(self, emissionDomain, sequenceSetInput, labelDomain=None, labelInput=None):
        """
        @p sequenceSetInput is a set of sequences from @p emissionDomain.

        There are several valid types for @p sequenceSetInput:
        - if @p sequenceSetInput is a string, it is interpreted as the filename
          of a sequence file to be read. File format should be fasta.
        - if @p sequenceSetInput is a list, it is considered as a list of lists
          containing the input sequences
        - @p sequenceSetInput can also be a pointer to a C sequence struct but
          this is only meant for internal use

        """
        self.emissionDomain = emissionDomain
        self.cseq = None

        if self.emissionDomain.CDataType == "int":
            # necessary C functions for accessing the sequence struct
            self.sequenceAllocationFunction = sequence
            #obsolete
            self.sequence_cmatrix = ghmmhelper.list2int_matrix
        elif self.emissionDomain.CDataType == "double":
            # necessary C functions for accessing the sequence struct
            self.sequenceAllocationFunction = sequence
            #obsolete
            self.sequence_cmatrix = ghmmhelper.list2double_matrix
        else:
            Log.error("C data type " + str(self.emissionDomain.CDataType) + " invalid.")


        # reads in the first sequence struct in the input file
        if isinstance(sequenceSetInput, str) or isinstance(sequenceSetInput, unicode):
            if sequenceSetInput[-3:] == ".fa" or sequenceSetInput[-6:] == ".fasta":
                # assuming FastA file:
                alfa = emissionDomain
                cseq = sequence(sequenceSetInput, alfa)
                if cseq is None:
                    Log.error("invalid FastA file: " + sequenceSetInput)
                self.cseq = cseq
            # check if ghmm is build with asci sequence file support
            elif not wrapper.ASCI_SEQ_FILE:
                Log.error("asci sequence files are deprecated. \
                Please convert your files to the new xml-format or rebuild the GHMM \
                with the conditional \"GHMM_OBSOLETE\".")
            else:
                if not os.path.exists(sequenceSetInput):
                    raise IOError, 'File ' + str(sequenceSetInput) + ' not found.'
                else:
                    tmp = self.seq_read(sequenceSetInput)
                    if len(tmp) > 0:
                        self.cseq = sequence(tmp[0])
                    else:
                        Log.error('File ' + str(sequenceSetInput) + ' not valid.')

        elif isinstance(sequenceSetInput, list):
            internalInput = [self.emissionDomain.internalSequence(seq) for seq in sequenceSetInput]
            (seq, lengths) = self.sequence_cmatrix(internalInput)
            # lens = wrapper.list2int_array(lengths)

            self.cseq = self.sequenceAllocationFunction(seq)

            if isinstance(labelInput, list) and isinstance(labelDomain, LabelDomain):
                assert len(sequenceSetInput) == len(labelInput), "no. of sequences and labels do not match."

                self.labelDomain = labelDomain
                internalLabels = [self.labelDomain.internalSequence(oneLabel) for oneLabel in labelInput]
                (label, labellen) = ghmmhelper.list2int_matrix(internalLabels)
                lens = wrapper.list2int_array(labellen)
                self.cseq.init_labels(label, lens)

        #internal use
        elif isinstance(sequenceSetInput, sequence) or isinstance(sequenceSetInput, sequence):
            Log.note("SequenceSet.__init__()" + str(sequenceSetInput))
            self.cseq = sequenceSetInput
            if labelDomain is not None:
                self.labelDomain = labelDomain

        else:
            Log.error("inputType " + str(type(sequenceSetInput)) + " not recognized.")


    def __del__(self):
        "Deallocation of C sequence struct."
        Log.note("__del__ SequenceSet " + str(self.cseq))


    def __str__(self):
        "Defines string representation."
        seq = self.cseq
        strout = ["SequenceSet (N=" + str(seq.seq_number) + ")"]

        if seq.seq_number <= 6:
            iter_list = range(seq.seq_number)
        else:
            iter_list = [0, 1, 'X', seq.seq_number - 2, seq.seq_number - 1]

        for i in iter_list:
            if i == 'X':
                strout.append('\n\n   ...\n')
            else:
                strout.append("\n  seq " + str(i) + "(len=" + str(seq.getLength(i)) + ")\n")
                strout.append('    ' + str(self[i]))

        return join(strout, '')


    def verboseStr(self):
        "Defines string representation."
        seq = self.cseq
        strout = ["\nNumber of sequences: " + str(seq.seq_number)]

        for i in range(seq.seq_number):
            strout.append("\nSeq " + str(i) + ", length " + str(seq.getLength(i)))
            strout.append(", weight " + str(seq.getWeight(i)) + ":\n")
            for j in range(seq.getLength(i)):
                if self.emissionDomain.CDataType == "int":
                    strout.append(str(self.emissionDomain.external((self.cseq.seq[i][j]))))
                elif self.emissionDomain.CDataType == "double":
                    strout.append(str(self.emissionDomain.external((double_matrix_getitem(self.cseq.seq, i, j) ))) + " ")

            # checking for labels
            if self.emissionDomain.CDataType == "int" and self.cseq.state_labels != None:
                strout.append("\nState labels:\n")
                for j in range(seq.getLabelsLength(i)):
                    strout.append(str(self.labelDomain.external(seq.state_labels[i][j])) + ", ")

        return join(strout, '')


    def __len__(self):
        """
        @returns the number of sequences in the SequenceSet.
        """
        return self.cseq.seq_number

    def sequenceLength(self, i):
        """
        @returns the lenght of sequence 'i' in the SequenceSet
        """
        return self.cseq.getLength(i)

    def getWeight(self, i):
        """
        @returns the weight of sequence i. @note Weights are used in Baum-Welch
        """
        return self.cseq.getWeight(i)

    def setWeight(self, i, w):
        """
        Set the weight of sequence i. @note Weights are used in Baum-Welch
        """
        wrapper.double_array_setitem(self.cseq.seq_w, i, w)

    def __getitem__(self, index):
        """
        @returns an EmissionSequence object initialized with a reference to
        sequence 'index'.
        """
        # check the index for correct range
        if index >= self.cseq.seq_number:
            raise IndexError

        seq = self.cseq.get_singlesequence(index)
        return EmissionSequence(self.emissionDomain, seq, ParentSequenceSet=self)


    def getSeqLabel(self, index):
        if not wrapper.SEQ_LABEL_FIELD:
            Log.error("the seq_label field is obsolete. If you need it rebuild the GHMM with the conditional \"GHMM_OBSOLETE\".")
        return wrapper.long_array_getitem(self.cseq.seq_label, index)

    def setSeqLabel(self, index, value):
        if not wrapper.SEQ_LABEL_FIELD:
            Log.error("the seq_label field is obsolete. If you need it rebuild the GHMM with the conditional \"GHMM_OBSOLETE\".")
        wrapper.long_array_setitem(self.cseq.seq_label, index, value)

    def getGeneratingStates(self):
        """
        @returns the state paths from which the sequences were generated as a
        Python list of lists.
        """
        states_len = wrapper.int_array2list(self.cseq.states_len, len(self))
        l_state = []
        for i, length in enumerate(states_len):
            col = wrapper.int_matrix_get_col(self.cseq.states, i)
            l_state.append(wrapper.int_array2list(col, length))

        return l_state


    def getSequence(self, index):
        """
        @returns the index-th sequence in internal representation
        """
        seq = []
        if self.cseq.seq_number > index:
            for j in range(self.cseq.getLength(index)):
                seq.append(self.cseq.getSymbol(index, j))
            return seq
        else:
            Log.error(str(index) + " is out of bounds, only " + str(self.cseq.seq_number) + "sequences")

    def getStateLabel(self, index):
        """
        @returns the labeling of the index-th sequence in internal representation
        """
        label = []
        if self.cseq.seq_number > index and self.cseq.state_labels != None:
            for j in range(self.cseq.getLabelsLength(index)):
                label.append(self.labelDomain.external(self.cseq.state_labels[index][j]))
            return label
        else:
            Log.error(str(0) + " is out of bounds, only " + str(self.cseq.seq_number) + "labels")

    def hasStateLabels(self):
        """
        @returns whether the sequence is labeled or not
        """
        return self.cseq.state_labels != None


    def merge(self, emissionSequences): # Only allow EmissionSequence?
        """
        Merges 'emissionSequences' into 'self'.
        @param emissionSequences can either be an EmissionSequence or SequenceSet
        object.
        """

        if not isinstance(emissionSequences, EmissionSequence) and not isinstance(emissionSequences, SequenceSet):
            Log.error("EmissionSequence or SequenceSet required, got " + str(emissionSequences.__class__.__name__))

        self.cseq.add(emissionSequences.cseq)
        del (emissionSequences) # removing merged sequences

    def getSubset(self, seqIndixes):
        """
        @returns a SequenceSet containing (references to) the sequences with the
        indices in 'seqIndixes'.
        """
        seqNumber = len(seqIndixes)
        seq = self.sequenceAllocationFunction([[]]*seqNumber)

        # checking for state labels in the source C sequence struct
        if self.emissionDomain.CDataType == "int" and self.cseq.state_labels is not None:

            Log.note("SequenceSet: found labels !")
            seq.calloc_state_labels()

        for i, seq_nr in enumerate(seqIndixes):
            # len_i = self.cseq.getLength(seq_nr)
            seq.seq[i] = self.cseq.getSequence(seq_nr)
            seq.seq_len[i] = self.cseq.getLength(seq_nr)
            seq.setWeight(i, self.cseq.getWeight(i))

            # setting labels if appropriate
            if self.emissionDomain.CDataType == "int" and self.cseq.state_labels is not None:
                self.cseq.copyStateLabel(seqIndixes[i], seq, seqIndixes[i])

        seq.seq_number = seqNumber

        return SequenceSetSubset(self.emissionDomain, seq, self)

    def write(self, fileName):
        "Writes (appends) the SequenceSet into file 'fileName'."
        self.cseq.write(fileName)

    def asSequenceSet(self):
        """convenience function, returns only self"""
        return self


class SequenceSetSubset(SequenceSet):
    """
    SequenceSetSubset contains a subset of the sequences from a SequenceSet
    object.

    @note On the C side only the references are used.
    """

    def __init__(self, emissionDomain, sequenceSetInput, ParentSequenceSet, labelDomain=None, labelInput=None):
        # reference on the parent SequenceSet object
        Log.note("SequenceSetSubset.__init__ -- begin -" + str(ParentSequenceSet))
        self.ParentSequenceSet = ParentSequenceSet
        SequenceSet.__init__(self, emissionDomain, sequenceSetInput, labelDomain, labelInput)



#-------------------------------------------------------------------------------
#Sequence, SequenceSet and derived  ------------------------------------------

class EmissionSequence(object):
    """ An EmissionSequence contains the *internal* representation of
    a sequence of emissions.

    It also contains a reference to the domain where the emissions originated from.
    """

    def __init__(self, emissionDomain, sequenceInput, labelDomain=None, labelInput=None, ParentSequenceSet=None):

        self.emissionDomain = emissionDomain

        if ParentSequenceSet is not None:
            # optional reference to a parent SequenceSet. Is needed for reference counting
            if not isinstance(ParentSequenceSet, SequenceSet):
                Log.error("Invalid reference. Only SequenceSet is valid.")
        self.ParentSequenceSet = ParentSequenceSet

        if self.emissionDomain.CDataType == "int":
            # necessary C functions for accessing the sequence struct
            self.sequenceAllocationFunction = sequence
            self.sequence_carray = wrapper.list2int_array
        elif self.emissionDomain.CDataType == "double":
            # necessary C functions for accessing the sequence struct
            self.sequenceAllocationFunction = sequence
            self.sequence_carray = wrapper.list2double_array
        else:
            Log.error("C data type " + str(self.emissionDomain.CDataType) + " invalid.")


        # check if ghmm is build with asci sequence file support
        if isinstance(sequenceInput, str) or isinstance(sequenceInput, unicode):
            if wrapper.ASCI_SEQ_FILE:
                if not os.path.exists(sequenceInput):
                    Log.error('File ' + str(sequenceInput) + ' not found.')
                else:
                    tmp = self.seq_read(sequenceInput)
                    if len(tmp) > 0:
                        self.cseq = tmp[0]
                    else:
                        Log.error('File ' + str(sequenceInput) + ' not valid.')

            else:
                Log.error("asci sequence files are deprecated. Please convert your files"
                                         + " to the new xml-format or rebuild the GHMM with"
                                         + " the conditional \"GHMM_OBSOLETE\".")

        #create a sequence with state_labels, if the appropiate parameters are set
        elif isinstance(sequenceInput, list):
            internalInput = self.emissionDomain.internalSequence(sequenceInput)
            seq = [internalInput]
            self.cseq = self.sequenceAllocationFunction(seq)

            if labelInput is not None and labelDomain is not None:
                assert len(sequenceInput) == len(labelInput), "Length of the sequence and labels don't match."
                assert isinstance(labelInput, list), "expected a list of labels."
                assert isinstance(labelDomain, LabelDomain), "labelDomain is not a LabelDomain class."

                self.labelDomain = labelDomain

                #translate the external labels in internal
                internalLabel = self.labelDomain.internalSequence(labelInput)
                self.cseq.init_labels([internalLabel], [len(internalInput)])

        # internal use
        elif isinstance(sequenceInput, (sequence, sequence)):
            if sequenceInput.seq_number > 1:
                Log.error("Use SequenceSet for multiple sequences.")
            self.cseq = sequenceInput
            if labelDomain != None:
                self.labelDomain = labelDomain

        else:
            Log.error("inputType " + str(type(sequenceInput)) + " not recognized.")

    def __len__(self):
        "Returns the length of the sequence."
        return self.cseq.getLength(0)

    def __setitem__(self, index, value):
        internalValue = self.emissionDomain.internal(value)
        self.cseq.setSymbol(0, index, internalValue)

    def __getitem__(self, index):
        """
        @returns the symbol at position 'index'.
        """
        if index < len(self):
            return self.cseq.getSymbol(0, index)
        else:
            raise IndexError

    def getSeqLabel(self):
        if not wrapper.SEQ_LABEL_FIELD:
            Log.error("the seq_label field is obsolete. If you need it rebuild the GHMM with the conditional \"GHMM_OBSOLETE\".")
        return wrapper.long_array_getitem(self.cseq.seq_label, 0)

    def setSeqLabel(self, value):
        if not wrapper.SEQ_LABEL_FIELD:
            Log.error("the seq_label field is obsolete. If you need it rebuild the GHMM with the conditional \"GHMM_OBSOLETE\".")
        wrapper.long_array_setitem(self.cseq.seq_label, 0, value)

    def getStateLabel(self):
        """
        @returns the labeling of the sequence in external representation
        """
        if self.cseq.state_labels != None:
            iLabel = wrapper.int_array2list(self.cseq.getLabels(0), self.cseq.getLabelsLength(0))
            return self.labelDomain.externalSequence(iLabel)
        else:
            Log.error(str(0) + " is out of bounds, only " + str(self.cseq.seq_number) + "labels")

    def hasStateLabels(self):
        """
        @returns whether the sequence is labeled or not
        """
        return self.cseq.state_labels != None

    def getGeneratingStates(self):
        """
        @returns the state path from which the sequence was generated as
        a Python list.
        """
        l_state = []
        for j in range(wrapper.int_array_getitem(self.cseq.states_len, 0)):
            l_state.append(self.cseq.states[0][j])

        return l_state

    def __str__(self):
        """Defines string representation."""
        seq = self.cseq
        strout = []

        l = seq.getLength(0)
        if l <= 80:

            for j in range(l):
                strout.append(str(self.emissionDomain.external(self[j])))
                if self.emissionDomain.CDataType == "double":
                    strout.append(" ")
        else:

            for j in range(5):
                strout.append(str(self.emissionDomain.external(self[j])))
                if self.emissionDomain.CDataType == "double":
                    strout.append(" ")
            strout.append('...')
            for j in range(l - 5, l):
                strout.append(str(self.emissionDomain.external(self[j])))
                if self.emissionDomain.CDataType == "double":
                    strout.append(" ")

        return join(strout, '')

    def verboseStr(self):
        "Defines string representation."
        seq = self.cseq
        strout = []
        strout.append("\nEmissionSequence Instance:\nlength " + str(seq.getLength(0)))
        strout.append(", weight " + str(seq.getWeight(0)) + ":\n")
        for j in range(seq.getLength(0)):
            strout.append(str(self.emissionDomain.external(self[j])))
            if self.emissionDomain.CDataType == "double":
                strout.append(" ")

        # checking for labels
        if self.emissionDomain.CDataType == "int" and self.cseq.state_labels != None:
            strout.append("\nState labels:\n")
            for j in range(seq.getLabelsLength(0)):
                strout.append(str(self.labelDomain.external(seq.state_labels[0][j])) + ", ")

        return join(strout, '')


    def sequenceSet(self):
        """
        @return a one-element SequenceSet with this sequence.
        """

        # in order to copy the sequence in 'self', we first create an empty SequenceSet and then
        # add 'self'
        seqSet = SequenceSet(self.emissionDomain, [])
        seqSet.cseq.add(self.cseq)
        return seqSet

    def write(self, fileName):
        "Writes the EmissionSequence into file 'fileName'."
        self.cseq.write(fileName)

    def setWeight(self, value):
        self.cseq.setWeight(0, value)
        self.cseq.total_w = value

    def getWeight(self):
        return self.cseq.getWeight(0)

    def asSequenceSet(self):
        """
        @returns this EmissionSequence as a one element SequenceSet
        """
        Log.note("EmissionSequence.asSequenceSet() -- begin " + repr(self.cseq))
        seq = self.sequenceAllocationFunction(1)

        # checking for state labels in the source C sequence struct
        if self.emissionDomain.CDataType == "int" and self.cseq.state_labels is not None:
            Log.note("EmissionSequence.asSequenceSet() -- found labels !")
            seq.state_labels_len = list(self.cseq.state_labels_len)
            seq.state_labels = [None]*len(seq.state_labels_len)
            self.cseq.copyStateLabel(0, seq, 0)

        seq.seq_len[0] = self.cseq.getLength(0)
        seq.seq[0] = self.cseq.getSequence(0)
        seq.seq_w[0] = self.cseq.getWeight(0)

        Log.note("EmissionSequence.asSequenceSet() -- end " + repr(seq))
        return SequenceSetSubset(self.emissionDomain, seq, self)


# XXX Change to MultivariateEmissionSequence
class ComplexEmissionSequence(object):
    """
    A MultivariateEmissionSequence is a sequence of multiple emissions per
    time-point. Emissions can be from distinct EmissionDomains. In particular,
    integer and floating point emissions are allowed. Access to emissions is
    given by the index, seperately for discrete and continuous EmissionDomains.

    Example: XXX

    MultivariateEmissionSequence also links to the underlying C-structure.

    Note: ComplexEmissionSequence has to be considered imutable for the moment.
    There are no means to manipulate the sequence positions yet.
    """

    def __init__(self, emissionDomains, sequenceInputs, labelDomain=None, labelInput=None):
        """
        @param emissionDomains a list of EmissionDomain objects corresponding
        to the list of sequenceInputs
        @param sequenceInputs a list of sequences of the same length (e.g.
        nucleotides and double values) that will be encoded
        by the corresponding EmissionDomain
        @bug @param labelDomain unused
        @bug @param labelInput unused
        """
        assert len(emissionDomains) == len(sequenceInputs)
        assert len(sequenceInputs) > 0
        self.length = len(sequenceInputs[0])
        for sequenceInput in sequenceInputs:
            assert self.length == len(sequenceInput)

        self.discreteDomains = []
        self.discreteInputs = []
        self.continuousDomains = []
        self.continuousInputs = []
        for i in range(len(emissionDomains)):
            if emissionDomains[i].CDataType == "int":
                self.discreteDomains.append(emissionDomains[i])
                self.discreteInputs.append(sequenceInputs[i])
            if emissionDomains[i].CDataType == "double":
                self.continuousDomains.append(emissionDomains[i])
                self.continuousInputs.append(sequenceInputs[i])

        self.cseq = wrapper.ghmm_dpseq(
            self.length,
            len(self.discreteDomains),
            len(self.continuousDomains)
        )

        for i in range(len(self.discreteInputs)):
            internalInput = self.discreteDomains[i].internalSequence(self.discreteInputs[i])
            pointerDiscrete = self.cseq.get_discrete(i)
            for j in range(len(self)):
                wrapper.int_array_setitem(pointerDiscrete, j, internalInput[j])
                # self.cseq.set_discrete(i, seq)

        for i in range(len(self.continuousInputs)):
            #seq = [float(x) for x in self.continuousInputs[i]]
            #seq = wrapper.list2double_array(seq)
            pointerContinuous = self.cseq.get_continuous(i)
            for j in range(len(self)):
                wrapper.double_array_setitem(pointerContinuous, j, self.continuousInputs[i][j])
                # self.cseq.set_continuous(i, seq)

    def __del__(self):
        """
        Deallocation of C sequence struct.
        """
        del self.cseq
        self.cseq = None

    def __len__(self):
        """
        @return the length of the sequence.
        """
        return self.length

    def getInternalDiscreteSequence(self, index):
        """
        access the underlying C structure and return the internal
        representation of the discrete sequence number 'index'
        @param index number of the discrete sequence
        @return a python list of ints
        """
        int_pointer = self.cseq.get_discrete(index)
        internal = wrapper.int_array2list(int_pointer, len(self))
        int_pointer = None
        return internal

    def getInternalContinuousSequence(self, index):
        """
        access the underlying C structure and return the internal
        representation of the continuous sequence number 'index'
        @param index number of the continuous sequence
        @return a python list of floats
        """
        d_pointer = self.cseq.get_continuous(index)
        internal = wrapper.double_array2list(d_pointer, len(self))
        return internal

    def getDiscreteSequence(self, index):
        """
        get the 'index'th discrete sequence as it has been given at the input
        @param index number of the discrete sequence
        @return a python sequence
        """
        return self.discreteInputs[index]

    def __getitem__(self, key):
        """
        get a slice of the complex emission sequence
        @param key either int (makes no big sense) or slice object
        @return a new ComplexEmissionSequence containing a slice of the
        original
        """
        domains = []
        for domain in self.discreteDomains:
            domains.append(domain)
        for domain in self.continuousDomains:
            domains.append(domain)
        slicedInput = []
        for input in self.discreteInputs:
            slicedInput.append(input[key])
        for input in self.continuousInputs:
            slicedInput.append(input[key])
        return ComplexEmissionSequence(domains, slicedInput)

    def __str__(self):
        """
        string representation. Access the underlying C-structure and return
        the sequence in all it's encodings (can be quite long)
        @return string representation
        """
        return "<ComplexEmissionSequence>"

    def verboseStr(self):
        """
        string representation. Access the underlying C-structure and return
        the sequence in all it's encodings (can be quite long)
        @return string representation
        """
        s = ("ComplexEmissionSequence (len=%i, discrete=%i, continuous=%i)\n" %
             (self.cseq.length, len(self.discreteDomains),
             len(self.continuousDomains)))
        for i in range(len(self.discreteDomains)):
            s += ("").join([str(self.discreteDomains[i].external(x))
                for x in self.getInternalDiscreteSequence(i)])
            s += "\n"
        for i in range(len(self.continuousDomains)):
            s += (",").join([str(self.continuousDomains[i].external(x))
                for x in self.getInternalContinuousSequence(i)])
            s += "\n"
        return s

