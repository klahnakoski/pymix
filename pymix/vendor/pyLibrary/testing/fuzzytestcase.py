# encoding: utf-8
#
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Kyle Lahnakoski (kyle@lahnakoski.com)
#

import unittest
from ..maths import Math
from ..structs.wraps import wrap


class FuzzyTestCase(unittest.TestCase):
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
        assertAlmostEqualValue(first, second, places=places, msg=msg, delta=delta)


def assertAlmostEqualValue(first, second, places=None, msg=None, delta=None):
    """
    Snagged from unittest/case.py, then modified (Aug2014)
    """
    if first == second:
        # shortcut
        return
    if delta is not None and places is not None:
        Log.error("specify delta or places not both")

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
            places = 18

        if Math.round(first, digits=places) == Math.round(second, digits=places):
            return

        standardMsg = '%s != %s within %r places' % (
            repr(first),
            repr(second),
            places
        )
    raise AssertionError(msg + ": (" + standardMsg + ")")




