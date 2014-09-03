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


class MixtureError(Exception):
    """Base class for mixture exceptions."""

    def __init__(self, message):
        self._message = message

    def __str__(self):
        return str(self._message)

    def _get_message(self):
        return self._message

    def _set_message(self, message):
        self._message += message

    message = property(_get_message, _set_message)


class InvalidPosteriorDistribution(MixtureError):
    """
    Raised if an invalid posterior distribution occurs.
    """
    pass


class InvalidDistributionInput(MixtureError):
    """
    Raised if a DataSet is found to be incompatible with a given MixtureModel.
    """
    pass


class ConvergenceFailureEM(MixtureError):
    """
    Raised if a DataSet is found to be incompatible with a given MixtureModel.
    """
    pass


