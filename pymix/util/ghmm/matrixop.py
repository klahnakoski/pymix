#******************************************************************************
#*
#*       This file is part of the General Hidden Markov Model Library,
#*       GHMM version __VERSION__, see http:# ghmm.org
#*
#*       Filename: ghmm/ghmm/matrixop.c
#*       Authors:  Christoph Hafemeister
#*
#*       Copyright (C) 2007-2008 Alexander Schliep
#*        Copyright (C) 2007-2008 Max-Planck-Institut fuer Molekulare Genetik,
#*                               Berlin
#*
#*       Contact: schliep@ghmm.org
#*
#*       This library is free software you can redistribute it and/or
#*       modify it under the terms of the GNU Library General Public
#*       License as published by the Free Software Foundation either
#*       version 2 of the License, or (at your option) any later version.
#*
#*       This library is distributed in the hope that it will be useful,
#*       but WITHOUT ANY WARRANTY without even the implied warranty of
#*       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#*       Library General Public License for more details.
#*
#*       You should have received a copy of the GNU Library General Public
#*       License along with this library if not, write to the Free
#*       Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#*
#******************************************************************************

#============================================================================
from math import sqrt
from util.ghmm.wrapper import matrix_alloc, DBL_MIN


# RETURN inverse, det PAIR
def ighmm_invert_det(length, cov):
    det = ighmm_determinant(cov, length)
    return ighmm_inverse(cov, length, det), det


#============================================================================
# calculate determinant of a square matrix
def ighmm_determinant(cov, n):
    if n == 1:
        return cov[0][0]
    if n == 2:
        return cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0]
        # matrix dimension is bigger than 2 - we have to do some work
    det = 0
    for j1 in range(0, n):
        m = matrix_alloc(n - 1, n - 1)
        for i in range(1, n):
            jm = 0
            for j in range(0, n):
                if j == j1:
                    continue
                m[(i - 1) * (n - 1) + jm] = cov[i * n + j]
                jm += 1

        det += pow(-1.0, 1.0 + j1 + 1.0) * cov[j1] * ighmm_determinant(m, n - 1)

    return det


#============================================================================
#
#  The inverse of a square matrix A with a non zero determinant is the adjoint
#  matrix divided by the determinant.
#  The adjoint matrix is the transpose of the cofactor matrix.
#  The cofactor matrix is the matrix of determinants of the minors Aij
#  multiplied by -1^(i+j).
#  The i,j'th minor of A is the matrix A without the i'th column or
#  the j'th row.
#
def ighmm_inverse(cov, n, det):
    inv = matrix_alloc(n, n)
    if n == 1:
        inv[0][0] = 1 / cov[0][0]
        return inv

    if n == 2:
        inv[0][0] = cov[1][1] / (cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0])
        inv[0][1] = (- cov[0][1]) / (cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0])
        inv[1][0] = (- cov[1][0]) / (cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0])
        inv[1][1] = cov[0][0] / (cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0])
        return inv

    for i in range(0, n):
        for j in range(0, n):
            # calculate minor i,j
            m = matrix_alloc(n - 1, n - 1)
            actrow = 0
            for ic in range(0, n):
                if ic == i:
                    continue
                actcol = 0
                for jc in range(0, n):
                    if jc == j:
                        continue
                    m[actrow][actcol] = cov[ic][jc]
                    actcol += 1

                actrow += 1

            # cofactor i,j is determinant of m times -1^(i+j)
            inv[j][i] = pow(-1.0, i + j + 2.0) * ighmm_determinant(m, n - 1) / det
    return inv

#============================================================================
def ighmm_cholesky_decomposition(sigmacd, dim, cov):
    # copy cov to sigmacd
    for row in range(0, dim):
        for j in range(0, dim):
            sigmacd[row][j] = cov[row][j]

    for row in range(0, dim):
        # First compute U[row][row]
        sum = cov[row][row]
        for j in range(row - 1):
            sum -= sigmacd[j][row] * sigmacd[j][row]
        if sum > DBL_MIN:
            sigmacd[row][row] = sqrt(sum)
            # Now find elements sigmacd[row*dim+k], k > row.
            for k in range(row + 1, dim):
                sum = cov[row][k]
                for j in range(0, (row - 1)):
                    sum -= sigmacd[j][row] * sigmacd[j][k]
                sigmacd[row][k] = sum / sigmacd[row][row]
        else:  # blast off the entire row.
            for k in range(row, dim):
                sigmacd[row][k] = 0.0
