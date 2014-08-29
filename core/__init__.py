import unittest
from numpy import ndarray
from core.util.env.logs import Log
from core.util.structs.wraps import wrap


def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):
    if isinstance(second, dict):
        first = wrap({"value": first})
        second = wrap(second)
        for k, v2 in second.items():
            v1 = first["value." + unicode(k)]
            assertAlmostEqual(self, v1, v2)
    elif isinstance(first, (list, ndarray)) and isinstance(second, (list, ndarray)):
        for a, b in zip(first, second):
            assertAlmostEqual(self, a, b, places=places, msg=msg, delta=delta)
    else:
        if self is None:
            assertAlmostEqualValue(first, second, places=places, msg=msg, delta=delta)
        else:
            unittest.TestCase.assertAlmostEqual(self, first, second, places=places, msg=msg, delta=delta)


def assertAlmostEqualValue(first, second, places=None, msg=None, delta=None):
    """
    Snagged from unittest/case.py
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

        if round(abs(second - first), places) == 0:
            return

        standardMsg = '%s != %s within %r places' % (
            repr(first),
            repr(second),
            places
        )
    raise Log.error(msg + ": (" + standardMsg + ")")
