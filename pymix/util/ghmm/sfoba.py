# *****************************************************************************
#
#        This file is part of the General Hidden Markov Model Library,
#        GHMM version __VERSION__, see http:# ghmm.org
#
#        Filename: ghmm/ghmm/sfoba.c
#        Authors:  Bernhard Knab, Benjamin Georgi
#
#        Copyright (C) 1998-2004 Alexander Schliep
#        Copyright (C) 1998-2001 ZAIK/ZPR, Universitaet zu Koeln
#        Copyright (C) 2002-2004 Max-Planck-Institut fuer Molekulare Genetik,
#                                Berlin
#
#        Contact: schliep@ghmm.org
#
#        This library is free software you can redistribute it and/or
#        modify it under the terms of the GNU Library General Public
#        License as published by the Free Software Foundation either
#        version 2 of the License, or (at your option) any later version.
#
#        This library is distributed in the hope that it will be useful,
#        but WITHOUT ANY WARRANTY without even the implied warranty of
#        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#        Library General Public License for more details.
#
#        You should have received a copy of the GNU Library General Public
#        License along with this library if not, write to the Free
#        Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
#
#
#        This file is version $Revision: 2243 $
#                        from $Date: 2008-11-05 08:05:05 -0500 (Wed, 05 Nov 2008) $
#              last change by $Author: christoph_mpg $.
#
# *****************************************************************************
from pymix.util.ghmm.wrapper import DBL_MIN


LOWER_SCALE_BOUND = 3.4811068399043105e-57


def sfoba_initforward(smo, alpha_1, omega, scale, b):
    scale[0] = 0.0
    if b == None:
        for i in range(smo.N):
            alpha_1[i] = smo.s[i].pi * smo.s[i].calc_b(omega)
            scale[0] += alpha_1[i]
    else:
        for i in range(smo.N):
            alpha_1[i] = smo.s[i].pi * b[i][smo.M]
            scale[0] += alpha_1[i]

    if scale[0] > DBL_MIN:
        c_0 = 1 / scale[0]
        for i in range(smo.N):
            alpha_1[i] *= c_0


def sfoba_stepforward(s, alpha_t, osc, b_omega):
    value = 0.0
    for i in range(len(s.in_a[osc])):
        value += s.in_a[osc][i] * alpha_t[i]

    value *= b_omega             # b_omega outside the sum
    return value
    # sfoba_stepforward

