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


# NOTE: code in this file is a mild modification of code from the GHMM library (www.ghmm.org)
import copy


class EmissionDomain:
    """ Abstract base class for emissions produced by an HMM.

        There can be two representations for emissions:
        1) An internal, used in ghmm.py and the ghmm C-library
        2) An external, used in your particular application

        Example:
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
        return emissionSequence


    def external(self, internal):
        """ Given an internal representation return the
            external representation
        """
        return internal

    def externalSequence(self, internalSequence):
        """ Given a sequence with the internal representation return the
            external representation
        """
        return internalSequence


    def isAdmissable(self, emission):
        """ Check whether emission is admissable (contained in) the domain
            raises GHMMOutOfDomain else
        """
        return None


class Alphabet(EmissionDomain):
    """ Discrete, finite alphabet

    """
    def __init__(self, listOfCharacters):
        """ Creates an alphabet out of a listOfCharacters """
        self.listOfCharacters = listOfCharacters
        self.index = {c: i for i, c in enumerate(self.listOfCharacters)}
        self.CDataType = "int"  # flag indicating which C data type should be used

    def __str__(self):
        strout = "Alphabet:\n"
        strout += "Number of symbols: " + str(len(self)) + "\n"
        strout += "External: " + str(self.listOfCharacters) + "\n"
        strout += "Internal: " + str(range(len(self))) + "\n"
        return strout


    def __len__(self):
        return len(self.listOfCharacters)


    def size(self):
        """ Deprecated """
        print "Warning: The use of .size() is deprecated. Use len() instead."
        return len(self.listOfCharacters)


    def internal(self, emission):
        """ Given a emission return the internal representation
        """
        return self.index[emission]


    def internalSequence(self, emissionSequence):
        """ Given a emission_sequence return the internal representation

            Raises KeyError
        """

        result = copy.deepcopy(emissionSequence)
        try:
            result = map(lambda i: self.index[i], result)
        except IndexError:
            raise KeyError
        return result


    def external(self, internal):
        """ Given an internal representation return the
            external representation

            Raises KeyError
        """
        if internal < 0 or len(self.listOfCharacters) <= internal:
            raise KeyError, "Internal symbol "+str(internal)+" not recognized."
        return self.listOfCharacters[internal]

    def externalSequence(self, internalSequence):
        """ Given a sequence with the internal representation return the
            external representation
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



DNA = Alphabet(['a','c','g','t'])
AminoAcids = Alphabet(['A','C','D','E','F','G','H','I','K','L',
                       'M','N','P','Q','R','S','T','V','W','Y'])


def IntegerRange(a,b):
    l = range(a,b)
    for i,s in enumerate(l):
        l[i] = str(s)
    return Alphabet(l)


#  We could use an Alphabet directly but this way it is more explicit.
class LabelDomain(Alphabet):
    def __init__(self, listOfLabels):
        Alphabet.__init__(self, listOfLabels)


class Float(EmissionDomain):

    def __init__(self):
        self.CDataType = "double" # flag indicating which C data type should be used

    def isAdmissable(self, emission):
        """ Check whether emission is admissable (contained in) the domain
            raises GHMMOutOfDomain else
        """
        return isinstance(emission, float)

