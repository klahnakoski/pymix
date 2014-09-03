## encoding: utf-8
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



import unittest
from .util.env.logs import Log
from .util.maths import Math
from .util.structs.wraps import wrap


class BaseTest(unittest.TestCase):
    def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        assertAlmostEqual(first, second, places=places, msg=msg, delta=delta)

    def assertEqual(self, first, second, msg=None):
        self.assertAlmostEqual(first, second, msg=msg)


def assertAlmostEqual(first, second, places=None, msg=None, delta=None):
    if isinstance(second, dict):
        first = wrap({"value": first})
        second = wrap(second)
        for k, v2 in second.items():
            v1 = first["value." + unicode(k)]
            assertAlmostEqual(v1, v2)
    elif hasattr(first, "__iter__") and hasattr(second, "__iter__"):
        for a, b in zip(first, second):
            assertAlmostEqual(a, b, places=places, msg=msg, delta=delta)
    else:
        # if self is None:
        #     assertAlmostEqualValue(first, second, places=places, msg=msg, delta=delta)
        # else:
        assertAlmostEqualValue(first, second, places=places, msg=msg, delta=delta)


def assertAlmostEqualValue(first, second, places=None, msg=None, delta=None):
    """
    Snagged from unittest/case.py, then modified (Aug2014)
    """
    if first == second:
        # shortcut
        return
    if delta is not None and places is not None:
        raise TypeError("specify delta or places not both")

    if delta is not None:
        if abs(first - second) <= delta:
            return

        standardMsg = '%s != %s within %s delta' % (
            repr(first),
            repr(second),
            repr(delta)
        )
    else:
        if places is None:
            places = 7

        if Math.round(first, digits=places) == Math.round(second, digits=places):
            return

        standardMsg = '%s != %s within %r places' % (
            repr(first),
            repr(second),
            places
        )
    raise AssertionError(msg + ": (" + standardMsg + ")")




