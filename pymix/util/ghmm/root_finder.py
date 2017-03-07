# *****************************************************************************  *
#
#        This file is part of the General Hidden Markov Model Library,
#        GHMM version __VERSION__, see http:# ghmm.org
#
#        Filename: ghmm/ghmm/root_finder.c
#        Authors:  Achim Gaedke
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
#        This file is version $Revision: 2267 $
#                        from $Date: 2009-04-24 11:01:58 -0400 (Fri, 24 Apr 2009) $
#              last change by $Author: grunau $.
#
# *****************************************************************************


#
#  this interface is used in sreestimate.c
#

def ghmm_zbrent_AB(func, x1, x2, tol, A, B, eps):
    a = min(x1, x2)
    fa = func(a)
    c = max(x1, x2)
    fc = func(c)
    b = (c - a) / 2.0
    fb = func(b)

    while abs(c - a) > tol + (tol * min(abs(a), abs(c))):
        r = fb / fc
        s = fb / fa
        t = fa / fc
        p = s * (t * (r - t) * (c - b) - (1.0 - r) * (b - a))
        q = (t - 1.0) * (r - 1.0) * (s - 1.0)

        x = b + p / q

        if x > a and x < c:
            #Accept interpolating point
            if x < b:
                c = b
                fc = fb
                b = x
                fb = func(b)
            elif x > b:
                a = b
                fa = fb
                b = x
                fb = func(b)
        else:
            #Use bisection
            pass
