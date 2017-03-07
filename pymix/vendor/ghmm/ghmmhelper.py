# From http://sourceforge.net/projects/ghmm/files/latest/download?source=typ_redirect
#
#/*******************************************************************************
#
#        This file is part of the General Hidden Markov Model Library,
#        GHMM version __VERSION__, see http://ghmm.org
#
#        Filename: ghmmhelper.py
#        Authors:  Benjamin Georgi, Janne Grunau
#
#        Copyright (C) 1998-2004 Alexander Schliep
#        Copyright (C) 1998-2001 ZAIK/ZPR, Universitaet zu Koeln
#        Copyright (C) 2002-2004 Max-Planck-Institut fuer Molekulare Genetik,
#                                Berlin
#
#        Contact: schliep@ghmm.org
#
#        This library is free software; you can redistribute it and/or
#        modify it under the terms of the GNU Library General Public
#        License as published by the Free Software Foundation; either
#        version 2 of the License, or (at your option) any later version.
#
#        This library is distributed in the hope that it will be useful,
#        but WITHOUT ANY WARRANTY; without even the implied warranty of
#        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#        Library General Public License for more details.
#
#        You should have received a copy of the GNU Library General Public
#        License along with this library; if not, write to the Free
#        Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
#
#        This file is version $Revision: 2259 $
#                        from $Date: 2009-04-22 04:19:35 -0400 (Wed, 22 Apr 2009) $
#              last change by $Author: grunau $.
#
# *****************************************************************************
from pymix.util.ghmm import wrapper


def double_matrix2list(cmatrix, row, col):
    llist = []
    for i in range(row):
        llist.append(wrapper.double_array2list(cmatrix[i], col))
    return llist


def list2double_matrix(matrix):
    """ Allocation and initialization of a double** based on a
    two dimensional Python list (list of lists).

    The number of elements in each column can vary.
    """
    seq = matrix
    col_len = [len(s) for s in seq]

    return seq, col_len


def list2int_matrix(matrix):
    """ Allocation and initialization of an int** based on a
    two dimensional Python list (list of lists).

    The number of elements in each column can vary.
    """
    seq = matrix
    col_len = [len(r) for r in seq]
    return seq, col_len


def int_matrix2list(cmatrix, row, col):
    llist = []
    for i in range(row):
        llist.append(wrapper.int_array2list(wrapper.int_matrix_get_col(cmatrix, i), col))
    return llist


def extract_out(lisprobs):
    """ Helper function for building HMMs from matrices: Used for
    transition matrices without transition classes.

    Extract out-/ingoing transitions from the row-vector resp. the
    column vector (corresponding to incoming transitions) of the
    transition matrix

    Allocates: .[out|in]_id and .[out|in]_a vectors
    """
    trans_id = range(len(lisprobs))
    return len(lisprobs), trans_id, lisprobs[:]


#def extract_out_probs(lisprobs,cos):
#    """ Helper function for building HMMs from matrices: Used for
#        transition matrices with 'cos' transition classes.
#        Extract out-/ingoing transitions from a matric consiting of
#        the row-vectors resp. the column vectors (corresponding to
#        incoming transitions) of the 'cos' transition matrices.
#        Hence, input is a 'cos' x N matrix.
#        Allocates: .[out|in]_id vector and .[out|in]_a array (of size cos x N)
#    """
#    lis = []
# parsing indixes belonging to postive probabilites
#    for j in range(cos):
#        for i in range(len(lisprobs[0])):
#            if lisprobs[j][i] != 0 and i not in lis:
#                lis.append(i)
#    print "lis: ", lis
#    trans_id   = wrapper.int_array_alloc(len(lis))
#    probsarray = wrapper.double_2d_array(cos, len(lis)) # C-function
# creating list with positive probabilities
#    for k in range(cos):
#        for j in range(len(lis)):
#            wrapper.set_2d_arrayd(probsarray,k,j, lisprobs[k][lis[j]])
#    trans_prob = twodim_double_array(probsarray, cos, len(lis)) # python CLASS, C internal
#
#    print trans_prob
# initializing c state index array
#    for i in range(len(lis)):
#        wrapper.int_array_setitem(trans_id,i,lis[i])
#    return [len(lis),trans_id,trans_prob]

def extract_out_cos(transmat, cos, state):
    """ Helper function for building HMMs from matrices: Used for
    transition matrices with 'cos' transition classes.

    Extract outgoing transitions for 'state' from the complete list
    of transition matrices

    Allocates: .out_a array (of size cos x N)
    """
    trans_id = []
    # parsing indixes belonging to postive probabilites
    for j in range(cos):
        for i in range(len(transmat[j][state])):
            trans_id.append(i)

    probsarray = wrapper.double_matrix_alloc(cos, len(trans_id))

    # creating list with positive probabilities
    for k in range(cos):
        for j in range(len(trans_id)):
            probsarray[k][j]=transmat[k][state][j]

    return len(trans_id), trans_id, probsarray


def extract_in_cos(transmat, cos, state):
    """ Helper function for building HMMs from matrices: Used for
    transition matrices with 'cos' transition classes.

    Extract ingoing transitions for 'state' from the complete list
    of transition matrices

    Allocates: .in_a array (of size cos x N)
    """
    trans_id = []

    # parsing indixes belonging to postive probabilites
    for j in range(cos):
        transmat_col_state = map(lambda x: x[state], transmat[j])
        for i in range(len(transmat_col_state)):
            trans_id.append(i)

    probsarray = wrapper.double_matrix_alloc(cos, len(trans_id)) # C-function

    for k in range(cos):
        for j in range(len(trans_id)):
            probsarray[k][j]=transmat[k][j][state]

    return [len(trans_id), trans_id, probsarray]


class twodim_double_array:
    """ Two-dimensional C-Double Array """

    def __init__(self, array, rows, columns, rowlabels=None, columnlabels=None):
        """
        Constructor
        """
        self.array = array
        self.rows = rows
        self.columns = columns
        self.size = (rows, columns)
        self.rowlabels = rowlabels
        self.columnlabels = columnlabels

    def __getitem__(self, index):
        """
        defines twodim_double_array[index[0],index[1]]
        """
        return wrapper.double_matrix_getitem(self.array, index[0], index[1])

    def __setitem__(self, index, value):
        """
        defines twodim_double_array[index[0],index[1]]
        """
        if len(index) == 2:
            wrapper.set_2d_arrayd(self.array, index[0], index[1], value)

    def __str__(self):
        """
        defines string representation
        """
        strout = "\n"
        if self.columnlabels is not None:
            for k in range(len(self.columnlabels)):
                strout += "\t"
                strout += str(self.columnlabels[k])
                strout += "\n"
        for i in range(self.rows):
            if self.rowlabels is not None:
                strout += str(self.rowlabels[i])
                strout += "\t"
            for j in range(self.columns):
                strout += "%2.4f" % self[i, j]
                strout += "\t"
                strout += "\n"
        return strout


# class double_array:
#     """A C-double array"""

#     def __init__(self, array, columns, columnlabels=None):
#         """Constructor"""
#         self.array = array
#         self.rows = 1
#         self.columns = columns
#         self.size = columns
#         self.columnlabels = columnlabels

#     def __getitem__(self,index):
#         """defines double_array[index] """
#         return wrapper.get_arrayd(self.array,index)

#     def __setitem__(self,index,value):
#         """ double_array[index] = value """
#         wrapper.set_arrayd(self.array,index,value)

#     def __str__(self):
#         """defines string representation"""
#         strout = "\n"
#         if self.columnlabels is not None:
#             for k in range(len(self.columnlabels)):
#                 strout+="\t"
#                 strout+= str(self.columnlabels[k])
#                 strout += "\n"
#         for i in range(self.columns):
#             strout += "%2.4f" % self[i]
#             strout += "\t"
#             strout += "\n"
#         return strout


def classNumber(A):
    """ Returns the number of transition classes in the matrix A   """
    if type(A[0][0]) == list:
        cos = len(A)
    else:
        cos = 1
    return cos

