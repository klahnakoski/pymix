################################################################################
#
#       This file is part of the Modified Python Mixture Package, original
#       source code is from https://svn.code.sf.net/p/pymix/code.  Also see
#       http://www.pymix.org/pymix/index.php?n=PyMix.Download
#
#       Changes made by: Kyle Lahnakoski (kyle@lahnakoski.com)
#
################################################################################
#
#       This file is part of the Python Mixture Package
#
#       file:    mixture.py
#       author: Benjamin Georgi
#
#       Copyright (C) 2004-2009 Benjamin Georgi
#       Copyright (C) 2004-2009 Max-Planck-Institut fuer Molekulare Genetik,
#                               Berlin
#
#       Contact: georgi@molgen.mpg.de
#
#       This library is free software; you can redistribute it and/or
#       modify it under the terms of the GNU Library General Public
#       License as published by the Free Software Foundation; either
#       version 2 of the License, or (at your option) any later version.
#
#       This library is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#       Library General Public License for more details.
#
#       You should have received a copy of the GNU Library General Public
#       License along with this library; if not, write to the Free
#       Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
################################################################################

#-------------------------------------------------------------------------------
# NOTE: code in this file is a mild modification of code from the GHMM library (www.ghmm.org)
import copy
from pyLibrary.queries import Q
from pymix.util.ghmm import wrapper
from pymix.util.logs import Log


class EmissionDomain:
    """ Abstract base class for emissions produced by an HMM.

    There can be two representations for emissions:
        1) An internal, used in ghmm.py and the ghmm C-library
        2) An external, used in your particular application

    Example:\n
    The underlying library represents symbols from a finite,
    discrete domain as integers (see Alphabet).

    EmissionDomain is the identity mapping
    """

    def internal(self, emission):
        """ Given a emission return the internal representation
        """
        return emission

    def internalSequence(self, emissionSequence):
        """ Given a emissionSequence return the internal representation
        """
        Log.error("not implemented")

    def external(self, internal):
        """ Given an internal representation return the external representation
        """
        return internal

    def externalSequence(self, internalSequence):
        """ Given a sequence with the internal representation return the external
        representation
        """
        return internalSequence

    def isAdmissable(self, emission):
        """ Check whether \p emission is admissible (contained in) the domain
        raises GHMMOutOfDomain else
        """
        return None


class Alphabet(EmissionDomain):
    """ Discrete, finite alphabet

    """

    def __init__(self, listOfCharacters):
        """
        Creates an alphabet out of a listOfCharacters
        @param listOfCharacters a list of strings (single characters most of
        the time), ints, or other objects that can be used as dictionary keys
        for a mapping of the external sequences to the internal representation
        or can alternatively be a SWIG pointer to a
        C alphabet_s struct

        @note
        Alphabets should be considered as immutable. That means the
        listOfCharacters and the mapping should never be touched after
        construction.
        """
        self.index = {}  # Which index belongs to which character

        self.listOfCharacters = copy.deepcopy(listOfCharacters)

        for i, c in enumerate(self.listOfCharacters):
            self.index[c] = i

        self.CDataType = "int" # flag indicating which C data type should be used
        self.dimension = 1


    def __str__(self):
        strout = ["<Alphabet:"]
        strout.append(str(self.listOfCharacters) + '>')

        return ''.join(strout)

    def verboseStr(self):
        strout = ["GHMM Alphabet:\n"]
        strout.append("Number of symbols: " + str(len(self)) + "\n")
        strout.append("External: " + str(self.listOfCharacters) + "\n")
        strout.append("Internal: " + str(range(len(self))) + "\n")
        return ''.join(strout)


    def __eq__(self, alph):
        if not isinstance(alph, Alphabet):
            return False
        else:
            if self.listOfCharacters == alph.listOfCharacters and self.index == alph.index and self.CDataType == alph.CDataType:
                return True
            else:
                return False

    def __len__(self):
        return len(self.listOfCharacters)

    def __hash__(self):
        #XXX rewrite
        # defining hash and eq is not recommended for mutable types.
        # => listOfCharacters should be considered immutable
        return id(self)

    #obsolete
    def size(self):
        """ @deprecated use len() instead
        """
        Log.warning("Warning: The use of .size() is deprecated. Use len() instead.")
        return len(self.listOfCharacters)

    def internal(self, emission):
        """
        Given a emission return the internal representation
        """
        try:
            return self.index[emission]
        except Exception, e:
            Log.error("key error", e)

    def internalSequence(self, emissionSequence):
        """
        Given a emission_sequence return the internal representation
        """
        return [self.index[c] for c in emissionSequence]


    def external(self, internal):
        """ Given an internal representation return the external representation

        @note the internal code -1 always represents a gap character '-'

        Raises KeyError
        """
        if internal == -1:
            return "-"
        if internal < -1 or len(self.listOfCharacters) < internal:
            Log.error("Internal symbol " + str(internal) + " not recognized.")
        return self.listOfCharacters[internal]

    def externalSequence(self, internalSequence):
        """ Given a sequence with the internal representation return the external
        representation

        Raises KeyError
        """
        result = copy.deepcopy(internalSequence)
        try:
            result = map(lambda i: self.listOfCharacters[i], result)
        except IndexError:
            raise KeyError
        return result

    def isAdmissable(self, emission):
        """ Check whether emission is admissable (contained in) the domain
        """
        return emission in self.listOfCharacters


DNA = Alphabet(['a', 'c', 'g', 't'])
AminoAcids = Alphabet(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])


def IntegerRange(a, b):
    """
    Creates an Alphabet with internal and external representation of range(a,b)
    @return Alphabet
    """
    return Alphabet(range(a, b))


# To be used for labelled HMMs. We could use an Alphabet directly but this way it is more explicit.
class LabelDomain(Alphabet):
    def __init__(self, listOfLabels):
        Alphabet.__init__(self, listOfLabels)


class Float(EmissionDomain):
    """Continuous Alphabet"""

    def __init__(self, dim=1):
        self.dimension=dim
        self.CDataType = "double"  # flag indicating which C data type should be used

    def __eq__(self, other):
        return isinstance(other, Float)

    def __hash__(self):
        # defining hash and eq is not recommended for mutable types.
        # for float it is fine because it is kind of state less
        return id(self)

    def isAdmissable(self, emission):
        """ Check whether emission is admissable (contained in) the domain

        raises GHMMOutOfDomain else
        """
        return isinstance(emission, float)

    def internalSequence(self, emissionSequence):
        """ Given a emissionSequence return the internal representation
        """
        if self.dimension==1:
            return emissionSequence
        else:
            # return zip(*[c for i, c in Q.groupby(emissionSequence, size=len(emissionSequence)/self.dimension)])
            return [c for i, c in Q.groupby(emissionSequence, size=self.dimension)]
