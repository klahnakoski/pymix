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

from numpy import ndarray
from pyLibrary.testing.fuzzytestcase import FuzzyTestCase

from pymix import mixtureHMM
from pymix.distributions.multinomial import MultinomialDistribution
from pymix.distributions.normal import NormalDistribution
from pymix.distributions.product import ProductDistribution
from pymix.models.mixture import MixtureModel
from pymix.vendor.ghmm.ghmm import SequenceSet
from pymix.vendor.ghmm.emission_domain import Alphabet


class HMMTests(FuzzyTestCase):
    def setUp(self):
        # building generating models
        self.DIAG = Alphabet(['.', '0', '8', '1'])

        A = [[0.3, 0.6, 0.1], [0.0, 0.5, 0.5], [0.4, 0.2, 0.4]]
        B = [[0.5, 0.2, 0.1, 0.2], [0.5, 0.4, 0.05, 0.05], [0.8, 0.1, 0.05, 0.05]]
        pi = [1.0, 0.0, 0.0]
        self.h1 = mixtureHMM.getHMM(mixtureHMM.ghmm.IntegerRange(0,4), mixtureHMM.ghmm.DiscreteDistribution(mixtureHMM.ghmm.IntegerRange(0,4)), A, B, pi)

        A2 = [[0.5, 0.4, 0.1], [0.3, 0.2, 0.5], [0.3, 0.2, 0.5]]
        B2 = [[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.4, 0.5], [0.2, 0.2, 0.3, 0.3]]
        pi2 = [0.6, 0.4, 0.0]
        self.h2 = mixtureHMM.getHMM(mixtureHMM.ghmm.IntegerRange(0,4), mixtureHMM.ghmm.DiscreteDistribution(mixtureHMM.ghmm.IntegerRange(0,4)), A2, B2, pi2)

        n1 = NormalDistribution(2.5, 0.5)
        n2 = NormalDistribution(6.0, 0.8)

        mult1 = MultinomialDistribution(3, 4, [0.23, 0.26, 0.26, 0.25], alphabet=self.DIAG)
        mult2 = MultinomialDistribution(3, 4, [0.7, 0.1, 0.1, 0.1], alphabet=self.DIAG)

        c1 = ProductDistribution([n1, mult1, self.h1])
        c2 = ProductDistribution([n2, mult2, self.h2])

        mpi = [0.4, 0.6]
        self.m = MixtureModel(2, mpi, [c1, c2])

        # mixture for sampling
        gc1 = ProductDistribution([n1, mult1])
        gc2 = ProductDistribution([n2, mult2])
        self.gen = MixtureModel(2, mpi, [gc1, gc2])


    def testinternalinitcomplex(self):
        # complex DataSet with HMM sequences
        dat = self.gen.sampleSet(100)

        # sampling hmm data
        seq1 = self.h1.hmm.sample(40, 10)
        seq2 = self.h2.hmm.sample(60, 10)

        seq1.merge(seq2)

        data = mixtureHMM.SequenceDataSet()
        data.fromGHMM(dat, [seq1])
        data.internalInit(self.m)

        self.assertEqual(str(data.complexFeature), '[0, 0, 1]')
        self.assertEqual(data.dimension, 5)
        self.assertEqual(data.suff_p, 6)


    def testinternalinitcomplexempty(self):
        # complex DataSet with HMM sequences only

        # sampling hmm data
        seq1 = self.h1.hmm.sample(40, 10)
        seq2 = self.h2.hmm.sample(60, 10)
        seq1.merge(seq2)

        data = mixtureHMM.SequenceDataSet()
        data.fromGHMM([], [seq1])

        self.assertRaises(AssertionError, data.internalInit, self.m)

        c1 = ProductDistribution([self.h1])
        c2 = ProductDistribution([self.h2])

        mpi = [0.4, 0.6]
        hm = MixtureModel(2, mpi, [c1, c2])

        data.internalInit(hm)

        self.assertEqual(str(data.complexFeature), '[1]')
        self.assertEqual(data.dimension, 1)
        self.assertEqual(data.suff_p, 1)


    def testgetinternalfeature(self):
        # complex DataSet with HMM sequences
        dat = self.gen.sampleSet(100)

        # sampling hmm data
        seq1 = self.h1.hmm.sample(40, 10)
        seq2 = self.h2.hmm.sample(60, 10)

        seq1.merge(seq2)

        data = mixtureHMM.SequenceDataSet()
        data.fromGHMM(dat, [seq1])
        data.internalInit(self.m)

        f0 = data.getInternalFeature(0)
        self.assertEqual(isinstance(f0, ndarray), True)

        f1 = data.getInternalFeature(1)
        self.assertEqual(isinstance(f1, ndarray), True)

        f2 = data.getInternalFeature(2)
        self.assertEqual(isinstance(f2, SequenceSet), True)


    def testem(self):
        # complex DataSet with HMM sequences and scalar data
        dat = self.gen.sampleSet(100)

        # sampling hmm data
        seq1 = self.h1.hmm.sample(40, 10)
        seq2 = self.h2.hmm.sample(60, 10)

        seq1.merge(seq2)

        data = mixtureHMM.SequenceDataSet()
        data.fromGHMM(dat, [seq1])
        data.internalInit(self.m)

        tA = [[0.5, 0.2, 0.3], [0.2, 0.3, 0.5], [0.1, 0.5, 0.4]]
        tB = [[0.2, 0.4, 0.1, 0.3], [0.5, 0.1, 0.2, 0.2], [0.4, 0.3, 0.15, 0.15]]
        tpi = [0.3, 0.3, 0.4]
        th1 = mixtureHMM.getHMM(mixtureHMM.ghmm.IntegerRange(0, 4), mixtureHMM.ghmm.DiscreteDistribution(mixtureHMM.ghmm.IntegerRange(0, 4)), tA, tB, tpi)

        tA2 = [[0.5, 0.4, 0.1], [0.3, 0.2, 0.5], [0.3, 0.2, 0.5]]
        tB2 = [[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.4, 0.4], [0.2, 0.1, 0.6, 0.1]]
        tpi2 = [0.3, 0.4, 0.3]
        th2 = mixtureHMM.getHMM(mixtureHMM.ghmm.IntegerRange(0, 4), mixtureHMM.ghmm.DiscreteDistribution(mixtureHMM.ghmm.IntegerRange(0, 4)), tA2, tB2, tpi2)

        tn1 = NormalDistribution(-1.5, 1.5)
        tn2 = NormalDistribution(9.0, 1.2)

        tmult1 = MultinomialDistribution(3, 4, [0.1, 0.1, 0.55, 0.25], alphabet=self.DIAG)
        tmult2 = MultinomialDistribution(3, 4, [0.4, 0.3, 0.1, 0.2], alphabet=self.DIAG)

        tc1 = ProductDistribution([tn1, tmult1, th1])
        tc2 = ProductDistribution([tn2, tmult2, th2])

        tmpi = [0.7, 0.3]
        tm = MixtureModel(2, tmpi, [tc1, tc2])

        tm.EM(data, 80, 0.1, silent=1)


    def testememptylist(self):
        # complex DataSet with HMM sequences only

        # sampling hmm data
        seq1 = self.h1.hmm.sample(40, 10)
        seq2 = self.h2.hmm.sample(60, 10)
        seq1.merge(seq2)

        data = mixtureHMM.SequenceDataSet()
        data.fromGHMM([], [seq1])

        tA = [[0.5, 0.2, 0.3], [0.2, 0.3, 0.5], [0.1, 0.5, 0.4]]
        tB = [[0.2, 0.4, 0.1, 0.3], [0.5, 0.1, 0.2, 0.2], [0.4, 0.3, 0.15, 0.15]]
        tpi = [0.3, 0.3, 0.4]
        th1 = mixtureHMM.getHMM(mixtureHMM.ghmm.IntegerRange(0,4), mixtureHMM.ghmm.DiscreteDistribution(mixtureHMM.ghmm.IntegerRange(0,4)), tA, tB, tpi)

        tA2 = [[0.5, 0.4, 0.1], [0.3, 0.2, 0.5], [0.3, 0.2, 0.5]]
        tB2 = [[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.4, 0.4], [0.2, 0.1, 0.6, 0.1]]
        tpi2 = [0.3, 0.4, 0.3]
        th2 = mixtureHMM.getHMM(mixtureHMM.ghmm.IntegerRange(0,4), mixtureHMM.ghmm.DiscreteDistribution(mixtureHMM.ghmm.IntegerRange(0,4)), tA2, tB2, tpi2)

        c1 = ProductDistribution([th1])
        c2 = ProductDistribution([th2])

        mpi = [0.4, 0.6]
        hm = MixtureModel(2, mpi, [c1, c2])

        data.internalInit(hm)

        hm.EM(data, 40, 0.1, silent=1)

    def testsimpleem(self):
        # sampling hmm data
        seq1 = self.h1.hmm.sample(40, 10)
        seq2 = self.h2.hmm.sample(60, 10)
        seq1.merge(seq2)

        data = mixtureHMM.SequenceDataSet()
        data.fromGHMM([], [seq1])

        tA = [[0.5, 0.2, 0.3], [0.2, 0.3, 0.5], [0.1, 0.5, 0.4]]
        tB = [[0.2, 0.4, 0.1, 0.3], [0.5, 0.1, 0.2, 0.2], [0.4, 0.3, 0.15, 0.15]]
        tpi = [0.3, 0.3, 0.4]
        th1 = ProductDistribution([mixtureHMM.getHMM(mixtureHMM.ghmm.IntegerRange(0,4), mixtureHMM.ghmm.DiscreteDistribution(mixtureHMM.ghmm.IntegerRange(0,4)), tA, tB, tpi)])

        tA2 = [[0.5, 0.4, 0.1], [0.3, 0.2, 0.5], [0.3, 0.2, 0.5]]
        tB2 = [[0.1, 0.1, 0.4, 0.4], [0.1, 0.1, 0.4, 0.4], [0.2, 0.1, 0.6, 0.1]]
        tpi2 = [0.3, 0.4, 0.3]
        th2 = ProductDistribution([mixtureHMM.getHMM(mixtureHMM.ghmm.IntegerRange(0,4), mixtureHMM.ghmm.DiscreteDistribution(mixtureHMM.ghmm.IntegerRange(0,4)), tA2, tB2, tpi2)])

        mpi = [0.4, 0.6]
        hm = MixtureModel(2, mpi, [th1, th2])

        data.internalInit(hm)

        hm.EM(data, 80, 0.1, silent=1)

# Run ALL tests (comment out to deactivate)
if __name__ == '__main__':
    unittest.main()

suiteHMMTests = unittest.makeSuite(HMMTests, 'test')


# Call to individual test suites, uncomment to activate as needed.
runner = unittest.TextTestRunner(verbosity=2)

