################################################################################
#
#       This file is part of the Python Mixture Package
#
#       file:    mixtureunittests.py
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
#
#
################################################################################
"""
Unittests for the Pymix package.

"""
import unittest
import copy
import random

from pymix import _C_mixextend
import numpy
import core

from core.distributions.discrete import DiscreteDistribution
from core.distributions.exponential import ExponentialDistribution
from core.distributions.multinomial import MultinomialDistribution
from core.distributions.multinormal import MultiNormalDistribution
from core.distributions.normal import NormalDistribution
from core.distributions.product import ProductDistribution
from core.priors.normal_gamma import NormalGammaPrior
from core.pymix_util.errors import InvalidDistributionInput
from core.mixture import modelSelection
from core.models.bayes import BayesMixtureModel
from core.models.constrained import ConstrainedMixtureModel
from core.models.labeled import LabeledMixtureModel
from core.models.mixture import MixtureModel
from core.parse import readMixture, writeMixture
from core.priors.dirichlet import DirichletPrior
from core.priors.dirichlet_mixture import DirichletMixturePrior
from core.priors.mixture_model import MixtureModelPrior
from core.priors.product import ProductDistributionPrior
from core.pymix_util.alphabet import Alphabet
from core.pymix_util.candidate_group import CandidateGroup
from core.pymix_util.constrained_dataset import ConstrainedDataSet
from core.pymix_util.dataset import DataSet



# XXX
# XXX     def sufficientStatistics(self, posterior, data):
#           -> add unitests and check numpy.dot(., .)[0]




#class TestGenerator:
#    """
#    Simple class for automatic generation of unittest code for a given model
#
#    """
#    def __init__(self, model):
#        self.model = model
#
#    def printTest(self,testModelName):
#
#        print ' '*8 +'self.assertEqual(str('+str(testModelName)+'.pi), "'+str(self.model.pi)+'" )'
#        for i in range(self.model.G):
#            for j in range(self.model.components[0].dist_nr):
#                dist = self.model.components[i].distList[j]
#                # lower hierarchy mixture
#                if isinstance(dist,mixture.MixtureModel):
#                    print ' '*8 +'self.assertEqual(str('+str(testModelName)+'.components['+str(i)+'].distList['+ \
#                               str(j)+'].pi), "'+str(dist.pi)+'" )'
#                    for i2 in range(dist.G):
#                        for j2 in range(dist.components[0].dist_nr):
#                            print ' '*8 +'self.assertEqual(str('+str(testModelName)+'.components['+str(i)+'].distList['+ \
#                            str(j)+'].components['+str(i2)+'].distList['+str(j2)+']), "' + str(dist.components[i2].distList[j2]) +'" )'
#                else:
#                    print ' '*8 +'self.assertEqual(str('+str(testModelName)+ \
#                    '.components['+str(i)+'].distList['+str(j)+']), "'+ str(dist) +'" )'
#
##        T = TestGenerator(train)
#        T.printTest('train')

#def dataAsList(data):
#    print ' '*8 +'data = DataSet()'
#    print ' '*8 +'l = [',
#    for i,row in enumerate(data.dataMatrix):
#        if i == data.N-1:
#            print ' '*14 ,row,']'
#        elif i == 0:
#            print row,','
#        else:
#            print ' '*14 ,row,','
#    print ' '*8 +'data.fromList(l)'
#    print ' '*8 +'data.internalInit(gen)'

def _testLists(tcase, list1, list2, places):
    tcase.assertEqual(len(list1), len(list2))

    for i in range(len(list1)):
        tcase.assertAlmostEqual(list1[i], list2[i], places)


def _testTupleLists(tcase, list1, list2, places):
    """
    Assumes two element tuples. First element is float, second element is int
    """
    tcase.assertEqual(len(list1), len(list2))

    for i in range(len(list1)):
        tcase.assertAlmostEqual(list1[i][0], list2[i][0], places)
        tcase.assertEqual(list1[i][1], list2[i][1], places)


class BaseTest(unittest.TestCase):
    def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        core.assertAlmostEqual(self, first, second, places=places, msg=msg, delta=delta)

    def assertEqual(self, first, second, msg=None):
        self.assertAlmostEqual(first, second, msg=msg)


class DataSetTests(BaseTest):
    """
    Tests for class DataSet.
    """

    def setUp(self):
        self.d1 = DataSet()  # empty DataSet
        self.d2 = DataSet()

    def testfromfile(self):
        self.d1.fromFile('./tests/test_fromfile0.txt')

        self.assertEqual(self.d1.headers, ['DLCL-0042', 'DLCL-0007', 'DLCL-0031', '"DLCL-0036;OCT"', 'DLCL-0030', 'DLCL-0004',
            'DLCL-0029', 'DLCL-0008', 'DLCL-0052', 'DLCL-0034', 'DLCL-0051', 'DLCL-0011', 'DLCL-0032', 'DLCL-0006',
            'DLCL-0049', 'DLCL-0039', 'DLCL-0001', 'DLCL-0018', 'DLCL-0037', 'DLCL-0010', 'DLCL-0015', 'DLCL-0026',
            'DLCL-0005', 'DLCL-0023', 'DLCL-0027', 'DLCL-0024', 'DLCL-0013', 'DLCL-0002', 'DLCL-0016', 'DLCL-0020',
            'DLCL-0003', 'DLCL-0014', 'DLCL-0048', 'DLCL-0033', 'DLCL-0025', 'DLCL-0040', 'DLCL-0017', 'DLCL-0028',
            'DLCL-0012', 'DLCL-0021', 'DLCL-0041', 'DLCL-0009'])

        self.assertEqual(self.d1.sampleIDs, ['1357915', '1358277', '1358064', '1072873', '627173', '1357623', '1072356',
            '1335756', '125294', '1268758', '1252094', '701768', '52564', '815774', 'LC4125', 'LC4103', '428103', '1371997',
            '428103___1', '90', '703791', '682938', '1272464', '1336130', '128088', '1250670', '1369098', '504877', '1356583',
            '713347', '705272', '1355435', '814768', '1367394', '1320051', '1287454', '1355426', '1287889', '1353111',
            '727551', '824602', '1300358', '298320', '298320___1', '815541', '429234', '701152', '301649', '1351622', '1340918',
            '1353149', '1339405', '1285581', '713080', '1251353', '1371030', '278808', '1286750', '1352434', '824198', '592498',
            '685371', '684361', '1354190', '153010', '1368529', '1355174', '809851', '686038', '1143183', '1301588', '490387',
            '1354710', '815845', '341889', '711771', '815555', '815555___1', '1333834', '469370', '1341161', '345032',
            '345032___1', '1316750', '1356425', '1272279', '1183989', '364941', '1290386', '47475', '1340120', '1353711',
            '490387___1', '813460', '1184052', '1351987', '704783', '752785', '549218', '728241', '1289318', '1339105',
            '1341422', '1355289', '1370612', '1356091', '1353782', '1334943', '1356654', '1234404', '685761', '1305138',
            '1354224', '1354224___1', '1288235', '1240756', '115281', '1240750', '1283152', '1270393', '1351623', '1305158',
            '815623', '1290438', '1335255', '1355045', '1334380', '1335782', '1356993', '1338245', '1355098', '686352',
            '1318612', '1671251', '1672022', '1357222', '1370670', '1368415', '1368505', '814360', '1372090', '1370776',
            '1369330', '1333775', '1357940', '1367790', '154493', '684328', '1241117', '1340139', '826131', '93', '1367632',
            '1130479', '162174', '259241', '207715', '1242115', '1240638', '1241211', '1350713', '491166', '221619', '1318920',
            '491166___1', '35356', '701504', '714394', '1305171', '1300154', '814979', '814979___1', '1241854', '686199',
            '213502', '1307025', '1351990', '740604', '293425___1', '1240813', '1172268', '840451', '725263', '254080', '1367995',
            '42062', '814841', '292524', '1233864', '1339768', '1351095', '1270669', '754080', '1235193', '344134___1', '1240959',
            '1240590', '1240959___1', '1241741', '771065', '162165', '162772', '162772___1', '746300', '510467', '1356323',
            '1340461', '1335780', '701622', '824639', '666448', '1300528', '1341297', '455210', '1353133', '1340443', '30209',
            '1234475', '510002', '1354154', '52408', '713213', '511387', '116801', '1352981', '1371026', '1372343', '1372323',
            '1352964', '527073', '1326908', '1335634', '1370519', '1335796'])

        self.assertEqual(self.d1.col_headers, [])

        self.assertEqual(self.d1.N, 234)

        self.assertEqual(self.d1.p, 42)

        self.d1.fromFile('./tests/test_fromfile1.txt', col_headers=True)

        self.assertEqual(self.d1.headers, ['DLCL-0042', 'DLCL-0007', 'DLCL-0031', '"DLCL-0036;OCT"', 'DLCL-0030', 'DLCL-0004',
            'DLCL-0029', 'DLCL-0008', 'DLCL-0052', 'DLCL-0034', 'DLCL-0051', 'DLCL-0011', 'DLCL-0032', 'DLCL-0006',
            'DLCL-0049', 'DLCL-0039', 'DLCL-0001', 'DLCL-0018', 'DLCL-0037', 'DLCL-0010', 'DLCL-0015', 'DLCL-0026',
            'DLCL-0005', 'DLCL-0023', 'DLCL-0027', 'DLCL-0024', 'DLCL-0013', 'DLCL-0002', 'DLCL-0016', 'DLCL-0020',
            'DLCL-0003', 'DLCL-0014', 'DLCL-0048', 'DLCL-0033', 'DLCL-0025', 'DLCL-0040', 'DLCL-0017', 'DLCL-0028',
            'DLCL-0012', 'DLCL-0021', 'DLCL-0041', 'DLCL-0009'])

        self.assertEqual(self.d1.col_headers, ['DLBCL2', 'DLBCL2', 'DLBCL2', 'DLBCL2', 'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL1',
            'DLBCL1', 'DLBCL1', 'DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL2', 'DLBCL2', 'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL1',
            'DLBCL1', 'DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL2', 'DLBCL2', 'DLBCL1', 'DLBCL1', 'DLBCL2', 'DLBCL2',
            'DLBCL1', 'DLBCL2', 'DLBCL2', 'DLBCL2', 'DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL2', 'DLBCL1'])

        self.assertEqual(len(self.d1.headers), len(self.d1.col_headers))

        self.assertEqual(len(self.d1.dataMatrix[0]), len(self.d1.headers))

        #self.assertEqual(self.d1.dataMatrix, '')

    def testfromarray(self):
        a = numpy.array([[0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0, 0, 0]], dtype='Float64')

        sID = ['s1', 's2', 's3', 's4', 's5']
        cH = ['f1', 'f2', 'f3']

        self.d1.fromArray(a, IDs=sID, headers=cH)

        self.assertEqual(self.d1.N, 5)
        self.assertEqual(self.d1.p, 3)
        self.assertEqual(self.d1.sampleIDs, sID)
        self.assertEqual(self.d1.headers, cH)

        dat = numpy.array([[0.25, 0.70, -0.22, -0.80, -0.10, -0.23, -0.37, -0.74, 0.33, 0.15],
            [0.31, 1.12, -0.25, 1.24, 1.06, 0.49, 0.46, -0.82, 0.68, 1.03],
            [-0.4, -0.35, 0.11, 0.39, 0.9, 0.0, 0.1, 0.6, -0.67, -0.52],
            [0.48, 1.39, 0.62, -0.06, -0.5, 0.06, -0.86, -0.77, -0.82, 0.76]], dtype='Float64')

        datID = ['g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9']
        datHeaders = ['s1', 's2', 's3', 's4']
        datClass = ['DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL2']

        self.d2.fromArray(dat, IDs=datID, headers=datHeaders, col_headers=datClass)

        self.assertEqual(self.d2.N, 4)
        self.assertEqual(self.d2.p, 10)
        self.assertEqual(self.d2.sampleIDs, datID)
        self.assertEqual(self.d2.headers, datHeaders)
        self.assertEqual(self.d2.col_headers, datClass)


    def testfromlist(self):
        l = [['A', 'G', 'T', 'C'],
            ['A', 'A', 'A', 'C'],
            ['C', 'G', 'C', 'C'],
            ['G', 'G', 'T', 'G'],
            ['T', 'A', 'T', 'A']]

        sID = ['s1', 's2', 's3', 's4', 's5']
        cH = ['b1', 'b2', 'b3', 'b4']

        self.d1.fromList(l, IDs=sID, headers=cH)

        self.assertEqual(self.d1.N, 5)
        self.assertEqual(self.d1.p, 4)
        self.assertEqual(self.d1.sampleIDs, sID)
        self.assertEqual(self.d1.headers, cH)

        dat = [[0.25, 0.70, -0.22, -0.80, -0.10, -0.23, -0.37, -0.74, 0.33, 0.15],
            [0.31, 1.12, -0.25, 1.24, 1.06, 0.49, 0.46, -0.82, 0.68, 1.03],
            [-0.4, -0.35, 0.11, 0.39, 0.9, 0.0, 0.1, 0.6, -0.67, -0.52],
            [0.48, 1.39, 0.62, -0.06, -0.5, 0.06, -0.86, -0.77, -0.82, 0.76]]

        datID = ['g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9']
        datHeaders = ['s1', 's2', 's3', 's4']
        datClass = ['DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL2']

        self.d2.fromList(dat, IDs=datID, headers=datHeaders, col_headers=datClass)

        self.assertEqual(self.d2.N, 4)
        self.assertEqual(self.d2.p, 10)
        self.assertEqual(self.d2.sampleIDs, datID)
        self.assertEqual(self.d2.headers, datHeaders)
        self.assertEqual(self.d2.col_headers, datClass)

    def testtranspose(self):
        self.d1.fromFile('./tests/test_fromfile1.txt', col_headers=True)

        self.assertEqual(self.d1.col_headers, ['DLBCL2', 'DLBCL2', 'DLBCL2',
            'DLBCL2', 'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL1',
            'DLBCL1', 'DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL2', 'DLBCL2', 'DLBCL1',
            'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL2', 'DLBCL1',
            'DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL2', 'DLBCL2', 'DLBCL1', 'DLBCL1',
            'DLBCL2', 'DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL2', 'DLBCL2', 'DLBCL2',
            'DLBCL1', 'DLBCL2', 'DLBCL2', 'DLBCL1'])

        self.assertEqual(self.d1.row_headers, [])

        self.assertEqual(self.d1.N, 1095)
        self.assertEqual(self.d1.p, 42)

        self.d1.transpose()

        self.assertEqual(self.d1.N, 42)
        self.assertEqual(self.d1.p, 1095)

        self.assertEqual(self.d1.row_headers, ['DLBCL2', 'DLBCL2', 'DLBCL2',
            'DLBCL2', 'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL1',
            'DLBCL1', 'DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL2', 'DLBCL2', 'DLBCL1',
            'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL1', 'DLBCL2', 'DLBCL1',
            'DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL2', 'DLBCL2', 'DLBCL1', 'DLBCL1',
            'DLBCL2', 'DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL2', 'DLBCL2', 'DLBCL2',
            'DLBCL1', 'DLBCL2', 'DLBCL2', 'DLBCL1'])


    def testinternalinit(self):
        # basic case
        DNA = Alphabet(['A', 'C', 'G', 'T'])
        c1 = ProductDistribution([MultinomialDistribution(1, 4, [0.5, 0.15, 0.15, 0.2], alphabet=DNA), MultinomialDistribution(1, 4, [0.21, 0.27, 0.27, 0.25], alphabet=DNA)])
        c2 = ProductDistribution([MultinomialDistribution(1, 4, [0.3, 0.25, 0.35, 0.1], alphabet=DNA), MultinomialDistribution(1, 4, [0.7, 0.1, 0.1, 0.1], alphabet=DNA)])
        c3 = ProductDistribution([MultinomialDistribution(1, 4, [0.7, 0.1, 0.1, 0.1], alphabet=DNA), MultinomialDistribution(1, 4, [0.2, 0.15, 0.15, 0.5], alphabet=DNA)])
        m = MixtureModel(3, [0.4, 0.2, 0.4], [c1, c2, c3])
        l = [['A', 'A'], ['T', 'A'], ['T', 'T'], ['A', 'G'], ['T', 'T'], ['A', 'C'],
            ['G', 'C'], ['T', 'C'], ['C', 'C'], ['A', 'T'], ['A', 'A'], ['G', 'A'], ['G', 'T'],
            ['A', 'G'], ['G', 'A'], ['C', 'A'], ['A', 'C'], ['T', 'T'], ['A', 'G'], ['C', 'A'],
            ['A', 'G'], ['A', 'T'], ['C', 'A'], ['T', 'A'], ['A', 'T'], ['T', 'T'], ['A', 'T'],
            ['T', 'T'], ['T', 'A'], ['G', 'G']]

        dat = DataSet()
        dat.fromList(l)

        dat.internalInit(m)

        self.assertEqual(dat.internalData[1].tolist(), [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(dat.internalData[12].tolist(), [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertEqual(dat.internalData[19].tolist(), [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


    def testsimpledatasetgaussian(self):
        n1 = ProductDistribution([NormalDistribution(-2.5, 0.5)])
        n2 = ProductDistribution([NormalDistribution(6.0, 0.8)])

        pi = [0.4, 0.6]
        gen = MixtureModel(2, pi, [n1, n2])

        random.seed(3586662)
        data = gen.sampleDataSet(10)

        f0 = data.getInternalFeature(0)

        self.assertEqual(f0.tolist(),
            [[-2.6751369624764929], [3.5541907883613257], [4.950009051623474], [4.7405049982596053], [-2.2523083183357127], [6.1391797079580561], [7.5525326592401942], [-1.8669732387994338],
                [4.8362379747532103], [-3.1207976176705601]])


    def testsimpledatasetdiscrete(self):
        DIAG = Alphabet(['.', '0', '8', '1'])

        d1 = ProductDistribution([DiscreteDistribution(4, [0.23, 0.26, 0.26, 0.25], alphabet=DIAG)])
        d2 = ProductDistribution([DiscreteDistribution(4, [0.7, 0.1, 0.1, 0.1], alphabet=DIAG)])

        #print "111",d1.sample()
        #print "222",d1.sampleSet(10)

        pi = [0.4, 0.6]
        gen = MixtureModel(2, pi, [d1, d2])

        random.seed(3586662)
        data = gen.sampleDataSet(10)
        #print "------discrete---------"
        #print data.dataMatrix
        #print data.internalData

        f0 = data.getInternalFeature(0)
        self.assertEqual(f0.tolist(), [[1.0], [2.0], [3.0], [0.0], [2.0], [2.0], [1.0], [0.0], [0.0], [3.0]])


    def testremovefeatures(self):
    #        n1 = NormalDistribution(2.5,0.5)
    #        n2 = NormalDistribution(6.0,0.8)
    #
    #        DIAG = Alphabet(['.','0','8','1'])
    #        mult1 = MultinomialDistribution(3,4,[0.23,0.26,0.26,0.25],alphabet = DIAG)
    #        mult2 = MultinomialDistribution(3,4,[0.7,0.1,0.1,0.1],alphabet = DIAG)
    #
    #        c1 = ProductDistribution([n1,mult1])
    #        c2 = ProductDistribution([n2,mult2])
    #
    #        mpi = [0.4,0.6]
    #        m = MixtureModel(2,mpi,[c1,c2])

        data = DataSet()
        l = [[2.3248630375235071, '8', '0', '1'],
            [4.950009051623474, '.', '.', '8'],
            [2.7476916816642873, '1', '8', '1'],
            [7.5525326592401942, '.', '1', '.'],
            [5.2438207409464539, '1', '.', '.'],
            [4.8362379747532103, '.', '.', '.'],
            [2.5870901172554128, '.', '0', '0'],
            [5.344556553752307, '1', '.', '.'],
            [6.9475338057098321, '.', '.', '.'],
            [2.5984482845693329, '0', '8', '1'],
            [2.6594447517982589, '1', '0', '8'],
            [2.515868200989412, '1', '8', '0'],
            [2.4332357214576641, '0', '.', '.']]
        data.fromList(l)

        data.removeFeatures([0], silent=1)
        self.assertEqual(data.dataMatrix,
            [['8', '0', '1'], ['.', '.', '8'], ['1', '8', '1'], ['.', '1', '.'], ['1', '.', '.'], ['.', '.', '.'], ['.', '0', '0'], ['1', '.', '.'], ['.', '.', '.'], ['0', '8', '1'], ['1', '0', '8'],
                ['1', '8', '0'], ['0', '.', '.']])
        self.assertEqual(data.headers, [1, 2, 3])
        self.assertEqual(data.p, 3)

        data.removeFeatures([1, 3], silent=1)
        self.assertEqual(data.dataMatrix, [['0'], ['.'], ['8'], ['1'], ['.'], ['.'], ['0'], ['.'], ['.'], ['8'], ['0'], ['8'], ['.']])
        self.assertEqual(data.headers, [2])
        self.assertEqual(data.p, 1)

        dat = [['0.25', '0.70', '-0.22', '-0.80', '-0.10', '-0.23', '-0.37', '-0.74', '0.33', '0.15'],
            ['0.31', '1.12', '-0.25', '1.24', '1.06', '0.49', '0.46', '-0.82', '0.68', '1.03'],
            ['-0.4', '-0.35', '0.11', '0.39', '0.9', '0.0', '0.1', '0.6', '-0.67', '-0.52'],
            ['0.48', '1.39', '0.62', '-0.06', '-0.5', '0.06', '-0.86', '-0.77', '-0.82', '0.76'],
            ['-0.35', '0.20', '-0.17', '-0.05', '0.03', '0.0', '-0.35', '-0.24', '-0.32', '0.32'],
            ['0.26', '-0.05', '0.23', '0.19', '0.0', '-0.45', '-0.01', '-0.3', '-0.78', '-0.24'],
            ['0.09', '-0.04', '-0.08', '0.92', '0.58', '0.02', '0.81', '-0.12', '0.12', '0.35'],
            ['0.01', '0.80', '1.04', '-0.08', '-0.18', '-0.01', '0.20', '-0.77', '0.01', '-0.01']]

        datHeaders = ['g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9']
        datID = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7']
        datClass = ['DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL2', 'DLBCL1', 'DLBCL1', 'DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL1']

        self.d1.fromList(dat, IDs=datID, headers=datHeaders, col_headers=datClass)

        self.d1.removeFeatures(['g1'], silent=1)
        self.assertEqual(self.d1.dataMatrix,
            [['0.25', '-0.22', '-0.80', '-0.10', '-0.23', '-0.37', '-0.74', '0.33', '0.15'], ['0.31', '-0.25', '1.24', '1.06', '0.49', '0.46', '-0.82', '0.68', '1.03'],
                ['-0.4', '0.11', '0.39', '0.9', '0.0', '0.1', '0.6', '-0.67', '-0.52'], ['0.48', '0.62', '-0.06', '-0.5', '0.06', '-0.86', '-0.77', '-0.82', '0.76'],
                ['-0.35', '-0.17', '-0.05', '0.03', '0.0', '-0.35', '-0.24', '-0.32', '0.32'], ['0.26', '0.23', '0.19', '0.0', '-0.45', '-0.01', '-0.3', '-0.78', '-0.24'],
                ['0.09', '-0.08', '0.92', '0.58', '0.02', '0.81', '-0.12', '0.12', '0.35'], ['0.01', '1.04', '-0.08', '-0.18', '-0.01', '0.20', '-0.77', '0.01', '-0.01']])
        self.assertEqual(self.d1.headers, ['g0', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9'])
        self.assertEqual(self.d1.col_headers, ['DLBCL2', 'DLBCL2', 'DLBCL2', 'DLBCL1', 'DLBCL1', 'DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL1'])
        self.assertEqual(self.d1.p, 9)

        self.d1.removeFeatures(['g0', 'g3'], silent=1)
        self.assertEqual(self.d1.dataMatrix,
            [['-0.22', '-0.10', '-0.23', '-0.37', '-0.74', '0.33', '0.15'], ['-0.25', '1.06', '0.49', '0.46', '-0.82', '0.68', '1.03'], ['0.11', '0.9', '0.0', '0.1', '0.6', '-0.67', '-0.52'],
                ['0.62', '-0.5', '0.06', '-0.86', '-0.77', '-0.82', '0.76'], ['-0.17', '0.03', '0.0', '-0.35', '-0.24', '-0.32', '0.32'], ['0.23', '0.0', '-0.45', '-0.01', '-0.3', '-0.78', '-0.24'],
                ['-0.08', '0.58', '0.02', '0.81', '-0.12', '0.12', '0.35'], ['1.04', '-0.18', '-0.01', '0.20', '-0.77', '0.01', '-0.01']])
        self.assertEqual(self.d1.headers, ['g2', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9'])
        self.assertEqual(self.d1.col_headers, ['DLBCL2', 'DLBCL1', 'DLBCL1', 'DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL1'])
        self.assertEqual(self.d1.p, 7)

    def testremovesamples(self):
        data = DataSet()
        l = [[2.3248630375235071, '8', '0', '1'],
            [4.950009051623474, '.', '.', '8'],
            [2.7476916816642873, '1', '8', '1'],
            [7.5525326592401942, '.', '1', '.'],
            [5.2438207409464539, '1', '.', '.'],
            [4.8362379747532103, '.', '.', '.'],
            [2.5870901172554128, '.', '0', '0'],
            [5.344556553752307, '1', '.', '.'],
            [6.9475338057098321, '.', '.', '.'],
            [2.5984482845693329, '0', '8', '1'],
            [2.6594447517982589, '1', '0', '8'],
            [2.515868200989412, '1', '8', '0'],
            [2.4332357214576641, '0', '.', '.']]
        data.fromList(l)

        data.removeSamples([0, 2, 4, 5], silent=1)
        self.assertEqual(data.dataMatrix,
            [[4.950009051623474, '.', '.', '8'], [7.5525326592401942, '.', '1', '.'], [2.5870901172554128, '.', '0', '0'], [5.344556553752307, '1', '.', '.'], [6.9475338057098321, '.', '.', '.'],
                [2.5984482845693329, '0', '8', '1'], [2.6594447517982589, '1', '0', '8'], [2.515868200989412, '1', '8', '0'], [2.4332357214576641, '0', '.', '.']])
        self.assertEqual(data.N, 9)

        data.removeSamples([6, 7, 9, 12], silent=1)
        self.assertEqual(data.dataMatrix,
            [[4.950009051623474, '.', '.', '8'], [7.5525326592401942, '.', '1', '.'], [6.9475338057098321, '.', '.', '.'], [2.6594447517982589, '1', '0', '8'], [2.515868200989412, '1', '8', '0']])
        self.assertEqual(data.N, 5)

        dat = [['0.25', '0.70', '-0.22', '-0.80', '-0.10', '-0.23', '-0.37', '-0.74', '0.33', '0.15'],
            ['0.31', '1.12', '-0.25', '1.24', '1.06', '0.49', '0.46', '-0.82', '0.68', '1.03'],
            ['-0.4', '-0.35', '0.11', '0.39', '0.9', '0.0', '0.1', '0.6', '-0.67', '-0.52'],
            ['0.48', '1.39', '0.62', '-0.06', '-0.5', '0.06', '-0.86', '-0.77', '-0.82', '0.76'],
            ['-0.35', '0.20', '-0.17', '-0.05', '0.03', '0.0', '-0.35', '-0.24', '-0.32', '0.32'],
            ['0.26', '-0.05', '0.23', '0.19', '0.0', '-0.45', '-0.01', '-0.3', '-0.78', '-0.24'],
            ['0.09', '-0.04', '-0.08', '0.92', '0.58', '0.02', '0.81', '-0.12', '0.12', '0.35'],
            ['0.01', '0.80', '1.04', '-0.08', '-0.18', '-0.01', '0.20', '-0.77', '0.01', '-0.01']]

        datHeaders = ['g0', 'g1', 'g2', 'g3', 'g4', 'g5', 'g6', 'g7', 'g8', 'g9']
        datID = ['s0', 's1', 's2', 's3', 's4', 's5', 's6', 's7']
        datClass = ['DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL2', 'DLBCL1', 'DLBCL1', 'DLBCL2', 'DLBCL1', 'DLBCL2', 'DLBCL1']

        self.d1.fromList(dat, IDs=datID, headers=datHeaders, col_headers=datClass)

        self.d1.removeSamples(['s0', 's2', 's4', 's5'], silent=1)
        self.assertEqual(self.d1.dataMatrix,
            [['0.31', '1.12', '-0.25', '1.24', '1.06', '0.49', '0.46', '-0.82', '0.68', '1.03'], ['0.48', '1.39', '0.62', '-0.06', '-0.5', '0.06', '-0.86', '-0.77', '-0.82', '0.76'],
                ['0.09', '-0.04', '-0.08', '0.92', '0.58', '0.02', '0.81', '-0.12', '0.12', '0.35'], ['0.01', '0.80', '1.04', '-0.08', '-0.18', '-0.01', '0.20', '-0.77', '0.01', '-0.01']])

# XXX to be implemented...
#
#    def testgetinternalfeature(self):
#        raise NotImplementedError
#
#    def testfromfiles(self):
#        pass
#
#    def testmaskdataset(self):
#        pass
#
#    def testmaskfeatures(self):
#        pass
#
#    def testgetexternalfeature(self):
#        pass
#
#    def testextractsubset(self):
#        pass
#

#    def testsinglefeaturesubset(self):
#        l =  [ ['A','G','T','C'],
#               ['A','A','A','C'],
#               ['C','G','C','C'],
#               ['G','G','T','G'],
#               ['T','A','T','A'] ]
#
#        sID = ['s1','s2','s3','s4','s5']
#        cH = ['b1','b2','b3','b4']
#
#        self.d1.fromList(l,IDs = sID, col_header = cH)
#
#
#        s1 = self.d1.singleFeatureSubset(0)
#        print s1
#        print s1.dataMatrix



class FormatDataTests(BaseTest):
    """
    Test functions for formatData
    """

    def testformatData(self):
        DNA = Alphabet(['A', 'C', 'G', 'T'])

        # single normal distribution
        dist = NormalDistribution(0.0, 1.0)
        s = dist.formatData(0.5)
        self.assertEqual(s, [1, [0.5]])

        # single multinomial distribution p = 1
        dist = MultinomialDistribution(1, 4, [0.1, 0.1, 0.2, 0.6], alphabet=DNA)
        s = dist.formatData(['A'])
        self.assertEqual(s, [4, [1, 0, 0, 0]])

        # single multinomial distribution p > 1
        DNA = Alphabet(['A', 'C', 'G', 'T'])
        dist = MultinomialDistribution(6, 4, [0.1, 0.1, 0.2, 0.6], alphabet=DNA)
        s = dist.formatData(['A', 'A', 'T', 'C', 'G', 'G'])
        self.assertEqual(s, [4, [2, 1, 2, 1]])

        # single discrete distribution
        DNA = Alphabet(['A', 'C', 'G', 'T'])
        dist = DiscreteDistribution(4, [0.1, 0.1, 0.2, 0.6], alphabet=DNA)
        s = dist.formatData(['A'])
        self.assertEqual(s, [1, [0]])

        # single multivariate Normal distribution
        dist = MultiNormalDistribution(2, [0.0, 1.0], [[0.3, 0.2], [0.1, 1.0]])
        s = dist.formatData([0.3, 1.2])
        self.assertEqual(s, [2, [0.29999999999999999, 1.2]])


        # ProductDistribution(Normal, Multinom, Discrete )
        n1 = NormalDistribution(2.5, 0.5)
        d1 = DiscreteDistribution(4, [0.23, 0.26, 0.26, 0.25], alphabet=DNA)
        m1 = MultinomialDistribution(6, 4, [0.1, 0.1, 0.2, 0.6], alphabet=DNA)
        p = ProductDistribution([n1, d1, m1])

        s = p.formatData([0.5, 'A', 'A', 'T', 'C', 'G', 'G', 'C'])
        self.assertEqual(s, [6, [0.5, 0, 1, 2, 2, 1]])

        # ProductDistribution(Mix(Normal), Normal, Multinom, Discrete )
        n1 = NormalDistribution(2.5, 0.5)
        n2 = NormalDistribution(-2.0, 1.0)
        c1 = ProductDistribution([n1])
        c2 = ProductDistribution([n2])
        pi = [0.4, 0.6]
        mix = MixtureModel(2, pi, [c1, c2])
        n1 = NormalDistribution(2.5, 0.5)
        d1 = DiscreteDistribution(4, [0.23, 0.26, 0.26, 0.25], alphabet=DNA)
        m1 = MultinomialDistribution(6, 4, [0.1, 0.1, 0.2, 0.6], alphabet=DNA)
        p = ProductDistribution([mix, n1, d1, m1])

        s = p.formatData([1.0, 0.5, 'A', 'A', 'T', 'C', 'G', 'G', 'C'])
        self.assertEqual(s, [7, [1.0, 0.5, 0, 1, 2, 2, 1]])


        # ProductDistribution(Mix(Multinom), Normal, Multinom, Discrete )
        n1 = MultinomialDistribution(3, 4, [0.1, 0.1, 0.2, 0.6], alphabet=DNA)
        n2 = MultinomialDistribution(3, 4, [0.1, 0.1, 0.2, 0.6], alphabet=DNA)
        c1 = ProductDistribution([n1])
        c2 = ProductDistribution([n2])
        pi = [0.4, 0.6]
        mix = MixtureModel(2, pi, [c1, c2])
        n1 = NormalDistribution(2.5, 0.5)
        d1 = DiscreteDistribution(4, [0.23, 0.26, 0.26, 0.25], alphabet=DNA)
        m1 = MultinomialDistribution(6, 4, [0.1, 0.1, 0.2, 0.6], alphabet=DNA)
        p = ProductDistribution([mix, n1, d1, m1])

        s = p.formatData(['A', 'G', 'T', 0.5, 'A', 'A', 'T', 'C', 'G', 'G', 'C'])
        self.assertEqual(s, [10, [1, 0, 1, 1, 0.5, 0, 1, 2, 2, 1]])


        # ProductDistribution(Mix(Discrete), Normal, Multinom, Discrete )
        n1 = DiscreteDistribution(4, [0.23, 0.26, 0.26, 0.25], alphabet=DNA)
        n2 = DiscreteDistribution(4, [0.23, 0.26, 0.26, 0.25], alphabet=DNA)
        c1 = ProductDistribution([n1])
        c2 = ProductDistribution([n2])
        pi = [0.4, 0.6]
        mix = MixtureModel(2, pi, [c1, c2])
        n1 = NormalDistribution(2.5, 0.5)
        d1 = DiscreteDistribution(4, [0.23, 0.26, 0.26, 0.25], alphabet=DNA)
        m1 = MultinomialDistribution(6, 4, [0.1, 0.1, 0.2, 0.6], alphabet=DNA)
        p = ProductDistribution([mix, n1, d1, m1])

        s = p.formatData(['T', 0.5, 'A', 'A', 'T', 'C', 'G', 'G', 'C'])
        self.assertEqual(s, [7, [3, 0.5, 0, 1, 2, 2, 1]])


class NormalDistributionTests(BaseTest):
    """
    Tests for class NormalDistribution
    """

    def setUp(self):
        self.dist = NormalDistribution(0.0, 1.0)


    def testeq(self):
        tdist1 = NormalDistribution(0.0, 1.0)
        self.assertEqual(self.dist, tdist1)

        tdist2 = NormalDistribution(2.43, 0.753)
        self.assertNotEqual(self.dist, tdist2)


    def testcopy(self):
        cp = copy.copy(self.dist)

        self.dist.mu = 5.0
        self.assertEqual(self.dist.mu, 5.0)
        self.dist.sigma = 3.2
        self.assertEqual(self.dist.sigma, 3.2)

        self.assertEqual(cp.mu, 0.0)
        self.assertEqual(cp.sigma, 1.0)


    def testpdf(self):
        a = numpy.array([[0.5], [1.0], [3.2], [-2.3], [7.8]], dtype='Float64')
        p = self.dist.pdf(a)

        self.assertEqual(p, [-1.04393853, -1.41893853, -6.03893853, -3.56393853, -31.33893853])


    def testmstep(self):
        a = numpy.array([[0.5], [1.0], [3.2], [-2.3], [7.8]], dtype='Float64')
        post1 = numpy.array([0.5, 0.5, 0.5, 0.5, 0.5])   # dummy posterior

        self.dist.MStep(post1, a)
        self.assertEqual(self.dist.mu, 2.04)
        self.assertEqual(self.dist.sigma, 3.37081592497)

        post2 = numpy.array([0.1, 0.1, 0.8, 0.1, 0.8])   # dummy posterior
        self.dist.MStep(post2, a)
        self.assertEqual(self.dist.mu, 4.58947368421)
        self.assertEqual(self.dist.sigma, 3.03469321034)

    def testsample(self):
        random.seed(3586662)
        x = self.dist.sample()
        self.assertEqual(x, -1.130581561)


    def testsampleset(self):
        random.seed(3586662)
        x = self.dist.sampleSet(10)
        self.assertEqual(x, [-1.13058156, -1.66891075, -3.05726151, -0.63949235, -1.57436875, -0.46658398, 0.8928686, -1.97693209, 0.17397463, 0.48817142])


    def testisvalid(self):
        self.dist.isValid(0.5)
        self.assertRaises(InvalidDistributionInput, self.dist.isValid, 'A')

    def testflatstr(self):
        f = self.dist.flatStr(1)
        self.assertEqual(f, '\t\t;Norm;0.0;1.0\n')


    #    def testposteriortraceback(self):
    #        pass

    #    def testupdatesuffp(self):
    #        pass

    def testmerge(self):
        pass


class MultiNormalDistributionTests(BaseTest):
    def setUp(self):
        random.seed(3586662)

        self.dist = MultiNormalDistribution(3, [0.0, 1.0, 2.0], [[1.0, 0.1, 0.04], [0.1, 1.0, 0.2], [0.2, 0.12, 1.0]])
        #print self.dist

        self.data = self.dist.sampleSet(10)
        #?!# self.data


    #    def testeq(self):
    #        pass
    #
    #    def testcopy(self):
    #        pass

    def testpdf(self):
        res = self.dist.pdf(self.data)
        sres = str(res)
        self.assertEqual(res, [-9.36609609, -4.27624557, -5.13430985, -3.7094928, -3.56997005, -3.95061984, -3.02135587, -3.7760178, -4.28146567, -3.95570885])


    def testmstep(self):
        post = numpy.ones(10, dtype='Float64')
        self.dist.MStep(post, self.data)

        self.assertEqual([round(m, 14) for m in self.dist.mu], [-0.35102902303187, 0.41163352335608999, 1.16925349374317])
        self.assertEqual([round(s, 14) for s in self.dist.sigma[0]], [0.58092143386218997, -0.017474956383180001, 0.58669594241713996])
        self.assertEqual([round(s, 14) for s in self.dist.sigma[1]], [-0.017474956383180001, 1.08875448993966, 0.44125993864780999])
        self.assertEqual([round(s, 14) for s in self.dist.sigma[2]], [0.58669594241713996, 0.44125993864780999, 1.1047533761494699])


    def testsample(self):
        s = self.dist.sample()
        self.assertEqual([round(v, 14) for v in s], [-1.3330318465269, 1.2010161194550599, 0.35547714589259])

    def testsampleset(self):
        s = self.dist.sampleSet(3)
        self.assertEqual(s, [[-1.33303185, 1.20101612, 0.35547715], [0.45069163, 0.39710096, 3.32416995], [0.29382531, 1.33373668, 2.58634925]])


    def testisvalid(self):
        pass


    def testflatstr(self):
        self.assertEqual(self.dist.p, 3)
        self.assertEqual(self.dist.mu, [0.0, 1.0, 2.0])
        self.assertEqual(self.dist.sigma, [[1.0, 0.10000000000000001, 0.040000000000000001], [0.10000000000000001, 1.0, 0.20000000000000001], [0.20000000000000001, 0.12, 1.0]])
        # f = self.dist.flatStr(1)
        # self.assertEqual(f, '\t\t;MultiNormal;3;[0.0,1.0,2.0];[[1.0,0.10000000000000001,0.040000000000000001], [0.10000000000000001,1.0,0.20000000000000001], [0.20000000000000001,0.12,1.0]]\n')

    def testposteriortraceback(self):
        pass

    def testupdatesuffp(self):
        pass

    def testmerge(self):
        pass

#class ExponentialDistributionTests(BaseTest):
#    def setUp(self):
#        # some setup
#        pass
#
#    def testeq(self):
#        pass
#
#    def testcopy(self):
#        pass
#
#    def testpdf(self):
#        pass
#
#    def testmstep(self):
#        pass
#
#    def testsample(self):
#        pass
#
#    def testsampleset(self):
#        pass
#

#    def testisvalid(self):
#        pass
#

#    def testflatstr(self):
#        pass
#
#    def testposteriortraceback(self):
#        pass
#
#    def testupdatesuffp(self):
#        pass
#
#    def testmerge(self):
#        pass
#
#
#class UniformDistributionTests(BaseTest):
#    def setUp(self):
#        # some setup
#        pass
#
#    def testeq(self):
#        pass
#
#    def testcopy(self):
#        pass
#
#    def testpdf(self):
#        pass
#
#    def testmstep(self):
#        pass
#
#    def testsample(self):
#        pass
#
#    def testsampleset(self):
#        pass
#

#    def testisvalid(self):
#        pass
#
#
#    def testflatstr(self):
#        pass
#
#    def testposteriortraceback(self):
#        pass
#
#    def testupdatesuffp(self):
#        pass
#
#    def testmerge(self):
#        pass
#
#class MultinomialDistributionTests(BaseTest):
#    def setUp(self):
#        # some setup
#        pass
#
#    def testeq(self):
#        pass
#
#    def testcopy(self):
#        pass
#
#    def testpdf(self):
#        pass
#
#    def testmstep(self):
#        pass
#
#    def testsample(self):
#        pass
#
#    def testsampleset(self):
#        pass
#
#    def testisvalid(self):
#        pass
#

#    def testflatstr(self):
#        pass
#
#    def testposteriortraceback(self):
#        pass
#
#    def testupdatesuffp(self):
#        pass
#
#    def testmerge(self):
#        pass
#


class DiscreteDistributionTests(BaseTest):
    """
    Tests for class DiscreteDistribution.
    """

    def setUp(self):
        self.DNA = Alphabet(['A', 'C', 'G', 'T'])

        self.d = DiscreteDistribution(4, [0.2, 0.3, 0.4, 0.1], self.DNA)
        self.dat = numpy.array(([[0], [1], [0], [2], [3]]))

    def testeq(self):
        d2 = DiscreteDistribution(4, [0.2, 0.3, 0.4, 0.1], self.DNA)
        self.assertEqual(self.d, d2)


    def testcopy(self):
        d2 = copy.copy(self.d)
        self.assertEqual(self.d, d2)
        del self.d
        self.assertEqual(d2.phi, [0.2, 0.3, 0.4, 0.1])
        self.assertEqual(d2.M, 4)

    def testpdf(self):
        self.assertEqual(self.d.pdf(self.dat), [-1.60943791, -1.2039728, -1.60943791, -0.91629073, -2.30258509])

    def testmstep(self):
        post = numpy.array([0.4, 0.2, 0.1, 0.8, 0.9])
        self.d.MStep(post, self.dat)
        self.assertEqual(self.d.phi, [0.20833333, 0.08333333, 0.33333333, 0.375])

    def testsample(self):
        random.seed(3586662)
        s = self.d.sample()
        self.assertEqual(s, 'A')


    def testsampleset(self):
        random.seed(3586662)
        s = self.d.sampleSet(10)
        self.assertEqual(s, ['A', 'C', 'A', 'G', 'C', 'T', 'G', 'A', 'C', 'G'])

    def testisvalid(self):
        self.d.isValid('A')
        self.assertRaises(InvalidDistributionInput, self.d.isValid, 'U')
        self.d.isValid(['T'])
        self.assertRaises(InvalidDistributionInput, self.d.isValid, ['U'])


    def testflatstr(self):
        # self.d.
        self.assertEqual(self.d.M, 4)
        self.assertEqual(self.d.phi, [0.20000000000000001, 0.29999999999999999, 0.40000000000000002, 0.10000000000000001])
        self.assertEqual(self.d.alphabet.listOfCharacters, ['A', 'C', 'G', 'T'])
        self.assertEqual(self.d.parFix, [0, 0, 0, 0])
        # f = self.d.flatStr(1)
        # self.assertEqual(f, "\t\t;Discrete;4;[0.20000000000000001,0.29999999999999999,0.40000000000000002,0.10000000000000001];['A', 'C', 'G', 'T'];[0,0,0,0]\n")

#
#    def testposteriortraceback(self):
#        pass
#
#    def testupdatesuffp(self):
#        pass
#
#    def testmerge(self):
#        pass
#


#class DirichletDistributionTests(BaseTest):
#    def setUp(self):
#        self.d1 = DirichletDistribution(4,[1.0,1.0,1.0,1.0])  # uniform prior
#        self.d2 = DirichletDistribution(4,[1.5,1.5,1.5,1.5])  # prior with emphasis on uniform phi
#        self.DNA = Alphabet(['A','C','G','T'])
#
#
#    def testeq(self):
#        self.assertEqual(self.d1 == self.d2, False)
#        self.assertEqual(self.d1 == 'hallo', False)
#        self.assertEqual(self.d1 == self.d1, True)
#
#    def testpdf(self):
#        pass
#




#  XXX more...


class DirichletPriorTests(BaseTest):
    """
    Tests for class DirichletPrior.
    """

    def setUp(self):
        self.d1 = DirichletPrior(4, [1.0, 1.0, 1.0, 1.0])  # uniform prior
        self.d2 = DirichletPrior(4, [1.5, 1.5, 1.5, 1.5])  # prior with emphasis on uniform phi
        self.d3 = DirichletPrior(4, [1.01, 1.01, 1.01, 3.0])  # prior with emphasis on the last symbol
        self.d4 = DirichletPrior(4, [1.5, 1.5, 1.5, 1.5])  # prior with emphasis on uniform phi
        self.DNA = Alphabet(['A', 'C', 'G', 'T'])


    def testeq(self):
        self.assertEqual(self.d1 == self.d2, False)
        self.assertEqual(self.d1 == 'hallo', False)
        self.assertEqual(self.d1 == self.d1, True)
        self.assertEqual(self.d2 == self.d4, True)


    def testpdf(self):
        m = MultinomialDistribution(3, 4, [0.4, 0.1, 0.1, 0.4], alphabet=self.DNA)
        disc = DiscreteDistribution(4, [0.25, 0.25, 0.25, 0.25], alphabet=self.DNA)

        # uniform prior
        self.assertEqual(self.d1.pdf(m), 1.79175946923)
        self.assertEqual(self.d1.pdf(disc), 1.79175946923)

        self.assertEqual(self.d2.pdf(m), 2.05174486845)
        self.assertEqual(self.d2.pdf(disc), 2.49803197108)

        # array valued input
        disc2 = DiscreteDistribution(4, [0.2, 0.1, 0.1, 0.6], alphabet=self.DNA)
        p = self.d2.pdf([m, disc, disc2])
        self.assertEqual(p, [2.05174487, 2.49803197, 1.90790383])

    def testmapmstep(self):

        # input is DiscreteDistribution
        dist = DiscreteDistribution(4, [0.4, 0.1, 0.1, 0.4], alphabet=self.DNA)
        dat1 = numpy.array([0, 0, 1, 2, 3, 3])
        post1 = numpy.ones(6, dtype='Float64')

        self.d1.mapMStep(dist, post1, dat1)
        self.assertEqual(dist.phi.tolist(), [0.33333333333333331, 0.16666666666666666, 0.16666666666666666, 0.33333333333333331])

        self.d2.mapMStep(dist, post1, dat1)
        self.assertEqual(dist.phi.tolist(), [0.3125, 0.1875, 0.1875, 0.3125])

        self.d3.mapMStep(dist, post1, dat1)
        self.assertEqual(dist.phi.tolist(), [0.25031133250311333, 0.12577833125778329, 0.12577833125778329, 0.4981320049813201])

        # input is MultinomialDistribution
        dist = MultinomialDistribution(3, 4, [0.4, 0.1, 0.1, 0.4], alphabet=self.DNA)
        dat1 = numpy.array([[3, 0, 0, 0], [1, 1, 0, 1], [0, 2, 0, 1], [0, 1, 1, 1], [1, 0, 0, 2], [0, 0, 0, 3]])
        post1 = numpy.ones(6, dtype='Float64')

        self.d1.mapMStep(dist, post1, dat1)
        self.assertEqual(dist.phi.tolist(), [0.27777777777777779, 0.22222222222222221, 0.055555555555555552, 0.44444444444444442])

        self.d2.mapMStep(dist, post1, dat1)
        self.assertEqual(dist.phi.tolist(), [0.27500000000000002, 0.22500000000000001, 0.074999999999999997, 0.42499999999999999])

        self.d3.mapMStep(dist, post1, dat1)
        self.assertEqual(dist.phi.tolist(), [0.25012481278082871, 0.200199700449326, 0.05042436345481776, 0.49925112331502741])

    def testmarginal(self):
        res1 = self.d1.marginal([1.0, 2.0, 11.0, 3.0])
        self.assertEqual(res1, -20.5566424959)
        res2 = self.d2.marginal([5.0, 8.0, 6.0, 1.0])
        self.assertEqual(res2, -27.5620419418)
        res3 = self.d3.marginal([5.0, 4.0, 5.0, 1.0])
        self.assertEqual(res3, -23.6695641578)
        res4 = self.d4.marginal([5.0, 2.0, 1.0, 3.0])
        self.assertEqual(res4, -15.7992843092)

    def testposterior(self):
        m = DiscreteDistribution(4, [0.1, 0.3, 0.2, 0.4])

        res1 = self.d1.posterior(m, [1.0, 2.0, 11.0, 3.0])
        self.assertEqual(res1, [2.49287482, 2.66514604, 0.4383393, 2.63537388])
        res2 = self.d2.posterior(m, [5.0, 8.0, 6.0, 1.0])
        self.assertEqual(res2, [2.175781, 1.29884317, 1.9097195, 2.61466619])
        res3 = self.d3.posterior(m, [5.0, 4.0, 5.0, 1.0])
        self.assertEqual(res3, [3.1167637, 3.30068711, 3.1167637, 3.14885983])
        res4 = self.d4.posterior(m, [5.0, 2.0, 1.0, 3.0])
        self.assertEqual(res4, [2.175781, 2.66755309, 2.61466619, 2.57712427])


    def testcopy(self):
        cp = copy.copy(self.d1)
        self.assertEqual(cp, self.d1)
        self.d1.alpha = numpy.zeros(4)
        self.assertEqual(str(cp.alpha), str(numpy.ones(4)))


    def testmapmstepmerge(self):
        post1 = numpy.array([0.3, 0.2, 0.5, 0.12, 0.5], dtype='Float64')
        post2 = numpy.array([0.4, 0.1, 0.1, 0.7, 0.3], dtype='Float64')

        data = numpy.array([0, 1, 2, 3, 0])

        d = DiscreteDistribution(4, [0.25] * 4)

        self.d2.mapMStep(d, post1 + post2, data)

        self.assertEqual(d, {"M": 4, "phi": [0.38314176, 0.1532567, 0.21072797, 0.25287356]})
        d2 = DiscreteDistribution(4, [0.25] * 4)

        req_stat1 = numpy.zeros(d2.M, dtype='Float64')
        req_stat2 = numpy.zeros(d2.M, dtype='Float64')
        for i in range(d2.M):
            i_ind = numpy.where(data == i)[0]
            req_stat1[i] = numpy.sum(post1[i_ind])
            req_stat2[i] = numpy.sum(post2[i_ind])

        dum1 = DiscreteDistribution(4, [0.25] * 4)
        dum2 = DiscreteDistribution(4, [0.1, 0.2, 0.3, 0.4])

        cm1 = CandidateGroup(dum1, sum(req_stat1), 'dum', req_stat1)
        cm2 = CandidateGroup(dum2, sum(req_stat2), 'dum', req_stat2)

        cmr = self.d2.mapMStepMerge([cm1, cm2])

        self.assertEqual(str(cmr.dist), str(d))  #


    #    def testsample(self):
    #        pass
    #
    #    def testsampleset(self):
    #        pass
    #
    #    def testisvalid(self):
    #        pass
    #
    def testflatstr(self):
        self.assertEqual(self.d1.M, 4)
        self.assertEqual(self.d1.alpha, [1.0, 1.0, 1.0, 1.0])
        # f = self.d1.flatStr(1)
        # self.assertEqual(f, '\t\t;DirichletPr;4;[1.0,1.0,1.0,1.0]\n')

#
#    def testposteriortraceback(self):
#        pass
#
#    def testupdatesuffp(self):
#        pass
#
#    def testmerge(self):
#        pass
#

class NormalGammaPriorTests(BaseTest):
    """
    Tests for class NormalGammaPrior.
    """

    def setUp(self):
        mu = 2.0
        kappa = 0.1
        dof = 2
        scale = 2.0

        self.ng = NormalGammaPrior(mu, kappa, dof, scale)


    def testpdf(self):
        n1 = NormalDistribution(0.5, 0.7)
        p1 = self.ng.pdf(n1)
        self.assertEqual(p1, -2.55726452327)

        n2 = NormalDistribution(-3.5, 2.7)
        p2 = self.ng.pdf(n2)
        self.assertEqual(p2, -7.38114015051)

        n3 = NormalDistribution(1.9, 0.1)
        p3 = self.ng.pdf(n3)
        self.assertEqual(p3, -90.6073056147)

        n4 = NormalDistribution(9.9, 0.01)  # -inf result raises exception
        self.assertRaises(ValueError, self.ng.pdf, n4)

        # array valued input
        p5 = self.ng.pdf([n1, n2, n3])
        self.assertEqual(p5, [-2.55726452, -7.38114015, -90.60730561])
        self.assertRaises(ValueError, self.ng.pdf, [n1, n2, n3, n4])

    def testmapmstepmerge(self):
        post1 = numpy.array([0.3, 0.2, 0.5, 0.12, 0.5], dtype='Float64')
        post2 = numpy.array([0.4, 0.1, 0.1, 0.7, 0.3], dtype='Float64')

        data = numpy.array([[1.2], [2.0], [1.1], [3.1], [0.4]])

        d = NormalDistribution(0.0, 1.0)

        self.ng.mapMStep(d, post1 + post2, data)

        self.assertEqual(d, {"mu": 1.55481927711, "sigma": 0.803640227896})
        d2 = DiscreteDistribution(4, [0.25] * 4)

        req_stat1 = d.sufficientStatistics(post1, data)
        req_stat2 = d.sufficientStatistics(post2, data)

        dum1 = DiscreteDistribution(4, [0.25] * 4)
        dum2 = DiscreteDistribution(4, [0.1, 0.2, 0.3, 0.4])

        cm1 = CandidateGroup(dum1, sum(post1), 'dum', req_stat1)
        cm2 = CandidateGroup(dum2, sum(post2), 'dum', req_stat2)

        cmr = self.ng.mapMStepMerge([cm1, cm2])

        self.assertEqual(str(cmr.dist), str(d))  #


class DirichletMixturePriorTests(BaseTest):
    """
    Tests for class DirichletMixturePrior.
    """

    def setUp(self):

        self.DNA = Alphabet(['A', 'C', 'G', 'T'])

        dPrior = [DirichletPrior(4, [1.3, 1.6, 1.1, 4.0]),
            DirichletPrior(4, [3.1, 1.2, 1.1, 1.0])]

        self.dmixPrior = DirichletMixturePrior(2, 4, [0.5, 0.5], dPrior)

        #mixprior = MixtureModelPrior(0.1,0.1,piPrior, [cPrior,cPrior,cPrior] )
        #m = BayesMixtureModel(4,[0.25,0.25,0.25,0.25],[p1,p2,p3,p4],mixprior,struct=1)


    def testeq(self):
        dPrior = [DirichletPrior(4, [1.3, 1.6, 1.1, 4.0]),
            DirichletPrior(4, [3.1, 1.2, 1.1, 1.0])]

        dmp2 = DirichletMixturePrior(2, 4, [0.5, 0.5], dPrior)

        self.assertEqual(self.dmixPrior == dmp2, True)

        dPrior = [DirichletPrior(4, [1.3, 1.6, 1.1, 4.0]),
            DirichletPrior(4, [3.1, 1.2, 1.1, 1.0])]

        dmp3 = DirichletMixturePrior(2, 4, [0.7, 0.3], dPrior)
        self.assertEqual(self.dmixPrior == dmp3, False)

        dPrior = [DirichletPrior(4, [2.3, 1.6, 1.1, 4.0]),
            DirichletPrior(4, [3.1, 1.2, 2.1, 1.0])]

        dmp4 = DirichletMixturePrior(2, 4, [0.5, 0.5], dPrior)
        self.assertEqual(self.dmixPrior == dmp4, False)


    def testpdf(self):
        m = MultinomialDistribution(3, 4, [0.4, 0.1, 0.1, 0.4], alphabet=self.DNA)
        disc = DiscreteDistribution(4, [0.25, 0.25, 0.2, 0.3], alphabet=self.DNA)

        # single distribution input
        self.assertEqual(self.dmixPrior.pdf(m), 2.29559766753)
        self.assertEqual(self.dmixPrior.pdf(disc), 1.76408269819)

        # list input
        _testLists(self, self.dmixPrior.pdf([m, disc]).tolist(), [2.2955976675334093, 1.7640826981888198], 14)
        _testLists(self, self.dmixPrior.pdf([disc, m, m, disc]).tolist(), [1.7640826981888198, 2.2955976675334093, 2.2955976675334093, 1.7640826981888198], 14)


    def testmapmstep(self):

        # input is DiscreteDistribution
        dist = DiscreteDistribution(4, [0.4, 0.1, 0.1, 0.4], alphabet=self.DNA)
        dat1 = numpy.array([0, 0, 1, 2, 3, 3])
        post1 = numpy.ones(6, dtype='Float64')

        self.dmixPrior.mapMStep(dist, post1, dat1)
        _testLists(self, dist.phi.tolist(), [0.31686594399861573, 0.18188034910242759, 0.15894585209433559, 0.342307854804621], 14)

        # input is MultinomialDistribution
        dist = MultinomialDistribution(3, 4, [0.4, 0.1, 0.1, 0.4], alphabet=self.DNA)
        dat1 = numpy.array([[3, 0, 0, 0], [1, 1, 0, 1], [0, 2, 0, 1], [0, 1, 1, 1], [1, 0, 0, 2], [0, 0, 0, 3]])
        post1 = numpy.ones(6, dtype='Float64')

        self.dmixPrior.mapMStep(dist, post1, dat1)
        _testLists(self, dist.phi.tolist(), [0.28374853426045621, 0.2143354801453049, 0.083217212994288553, 0.41869877259995031], 14)

    def testconsistencywithdirichletprior(self):
        """
        There are special cases where a dirichlet mixture prior (dmp) is equivalent to a single
        dirichlet prior. To test the computations consistency of results for some of these cases
        is checked.
        """


        # a dmp with equal parameters for all components is equivalent to
        # a single dirichlet with the same parameters
        dPrior = [DirichletPrior(4, [1.3, 1.6, 1.1, 4.0]),
            DirichletPrior(4, [1.3, 1.6, 1.1, 4.0]),
            DirichletPrior(4, [1.3, 1.6, 1.1, 4.0])]
        dmp1 = DirichletMixturePrior(3, 4, [0.2, 0.3, 0.5], dPrior)
        d1 = DirichletPrior(4, [1.3, 1.6, 1.1, 4.0])

        # check pdf results
        dist1 = DiscreteDistribution(4, [0.1, 0.6, 0.1, 0.2], alphabet=self.DNA)
        self.assertEqual(dmp1.pdf(dist1), d1.pdf(dist1))
        dist2 = MultinomialDistribution(3, 4, [0.5, 0.3, 0.1, 0.1], alphabet=self.DNA)
        self.assertEqual(dmp1.pdf(dist2), d1.pdf(dist2))

        # check parameter estimates

        # XXX different scale for alphas in DirichletPrior and DirichletMixturePrior
        # DirichletPrior uses to true MAP estimates, DirichletMixturePrior follows
        # the approach usually taken in the literature of using the least-squares estimates
        # of the MAP. The difference is a -1 term in the numerator of the estimation formula

        # Therefore adding 1 to each alpha yields the same parameter estimates
        d1 = DirichletPrior(4, [2.3, 2.6, 2.1, 5.0])

        dat1 = numpy.array([0, 0, 1, 1, 2, 0, 3, 2, 2, 3])
        post1 = numpy.ones(10, dtype='Float64')
        temp1 = DiscreteDistribution(4, [0.2, 0.2, 0.2, 0.4], alphabet=self.DNA)
        temp2 = DiscreteDistribution(4, [0.2, 0.2, 0.2, 0.4], alphabet=self.DNA)

        dmp1.mapMStep(temp1, post1, dat1)
        d1.mapMStep(temp2, post1, dat1)

        self.assertEqual(str(temp1.phi), str(temp2.phi))


    #        dat2 = numpy.array( [[3,0,0,0], [1,0,0,2], [0,2,0,1], [1,1,0,1], [1,0,0,2], [0,0,0,3], [0,0,0,3], [1,0,0,2], [1,1,0,1], [1,0,0,2]])
    #        post2 = numpy.ones(10,dtype='Float64')
    #        temp1 = MultinomialDistribution(3,4, [1.0,0.0,0.0,0.0], alphabet = self.DNA)
    #        temp2 = MultinomialDistribution(3,4, [0.0,0.0,0.0,1.0], alphabet = self.DNA)

    def testposterior(self):
        dist1 = DiscreteDistribution(4, [0.1, 0.6, 0.1, 0.2], alphabet=self.DNA)
        p1 = self.dmixPrior.posterior(dist1)
        self.assertAlmostEqual(p1[0], 0.78314206506543915, 14)
        self.assertAlmostEqual(p1[1], 0.21685793493456079, 14)

        dist2 = MultinomialDistribution(3, 4, [0.5, 0.3, 0.1, 0.1], alphabet=self.DNA)
        p2 = self.dmixPrior.posterior(dist2)
        self.assertAlmostEqual(p2[0], 0.018530789429022847, 14)
        self.assertAlmostEqual(p2[1], 0.98146921057097714, 14)


    def testmapmstepmerge(self):
        post1 = numpy.array([0.3, 0.2, 0.5, 0.12, 0.5], dtype='Float64')
        post2 = numpy.array([0.4, 0.1, 0.1, 0.7, 0.3], dtype='Float64')

        data = numpy.array([0, 1, 2, 3, 0])

        d = DiscreteDistribution(4, [0.25] * 4)

        self.dmixPrior.mapMStep(d, post1 + post2, data)

        self.assertEqual(d, {"M": 4, "phi": [0.36649153, 0.16247853, 0.16440497, 0.30662497]})
        d2 = DiscreteDistribution(4, [0.25] * 4)

        req_stat1 = numpy.zeros(d2.M, dtype='Float64')
        req_stat2 = numpy.zeros(d2.M, dtype='Float64')
        for i in range(d2.M):
            i_ind = numpy.where(data == i)[0]
            req_stat1[i] = numpy.sum(post1[i_ind])
            req_stat2[i] = numpy.sum(post2[i_ind])

        dum1 = DiscreteDistribution(4, [0.25] * 4)
        dum2 = DiscreteDistribution(4, [0.1, 0.2, 0.3, 0.4])

        cm1 = CandidateGroup(dum1, sum(req_stat1), 0.3, req_stat1)
        cm2 = CandidateGroup(dum2, sum(req_stat2), 0.7, req_stat2)

        cmr = self.dmixPrior.mapMStepMerge([cm1, cm2])

        self.assertEqual(cmr.dist, {"M": 4, "phi": [0.54933743, 0.28053815, 0.2713648, 0.59875962]})  #  str(d)


class ProductDistributionPriorTests(BaseTest):
    """
    Tests for class ProductDistributionPrior.
    """

    def setUp(self):
        self.comp_pr = ProductDistributionPrior([DirichletPrior(4, [1.0, 2.0, 2.0, 1.0]),
            DirichletPrior(4, [1.0, 1.0, 4.0, 1.0]),
            NormalGammaPrior(2.0, 0.1, 2, 1.0),
            NormalGammaPrior(2.0, 0.1, 2, 1.0)])

    def testeq(self):
        comp_pr2 = ProductDistributionPrior([DirichletPrior(4, [1.0, 2.0, 2.0, 1.0]),
            DirichletPrior(4, [1.0, 4.0, 1.0, 1.0]),
            NormalGammaPrior(2.0, 0.1, 2, 1.0),
            NormalGammaPrior(2.0, 0.1, 2, 1.0)])

        self.assertEqual(self.comp_pr == comp_pr2, False)

        comp_pr2[1] = DirichletPrior(4, [1.0, 1.0, 4.0, 1.0])
        self.assertEqual(self.comp_pr == comp_pr2, True)

        # XXX ...


class MixtureModelPriorTests(BaseTest):
    """
    Tests for class MixtureModelPrior.
    """

    def setUp(self):
        piPrior = DirichletPrior(1, [1.0])

        mu = 2.0
        kappa = 0.1
        dof = 2
        scale = 2.0

        compPrior = [
            DirichletPrior(4, [1.0, 2.0, 2.0, 1.0]),
            DirichletPrior(4, [1.0, 1.0, 4.0, 1.0]),
            NormalGammaPrior(mu, kappa, dof, scale),
            NormalGammaPrior(mu, kappa, dof, scale)
        ]

        self.prior = MixtureModelPrior(0.5, 0.6, piPrior, compPrior)

    def testeq(self):
        piPrior = DirichletPrior(1, [1.0])

        mu = 2.0
        kappa = 0.1
        dof = 2
        scale = 2.0

        compPrior = [DirichletPrior(4, [1.0, 2.0, 2.0, 1.0]),
            DirichletPrior(4, [1.0, 1.0, 4.0, 1.0]),
            NormalGammaPrior(mu, kappa, dof, scale),
            NormalGammaPrior(mu, kappa, dof, scale)]

        prior2 = MixtureModelPrior(0.0, 0.6, piPrior, compPrior)

        self.prior == prior2


    def testpdf(self):
        c = ProductDistribution([DiscreteDistribution(4, [0.25] * 4),
            MultinomialDistribution(3, 4, [0.1, 0.6, 0.1, 0.2]),
            NormalDistribution(0.0, 1.0),
            NormalDistribution(-2.0, 0.5)])
        m = MixtureModel(1, [1.0], [c])

        self.assertEqual(self.prior.pdf(m), -12.4635011183)

    def testcopy(self):
        cp = copy.copy(self.prior)
        self.assertEqual(self.prior == cp, True)

        self.prior.compPrior.priorList[0].alpha = [0.0]

        self.assertEqual(str(cp.compPrior.priorList[0].alpha), str(numpy.array([1.0, 2.0, 2.0, 1.0])))

    def testisvalid(self):
        c = ProductDistribution([DiscreteDistribution(4, [0.25] * 4),
            MultinomialDistribution(3, 4, [0.1, 0.6, 0.1, 0.2]),
            NormalDistribution(0.0, 1.0),
            NormalDistribution(-2.0, 0.5)])
        m1 = MixtureModel(1, [1.0], [c])
        self.prior.isValid(m1)

        m2 = MixtureModel(2, [0.3, 0.7], [c, c])
        self.assertRaises(InvalidDistributionInput, self.prior.isValid, m2)

        c2 = ProductDistribution([DiscreteDistribution(4, [0.25] * 4),
            NormalDistribution(0.0, 1.0),
            MultinomialDistribution(3, 4, [0.1, 0.6, 0.1, 0.2]),
            NormalDistribution(-2.0, 0.5)])
        m3 = MixtureModel(1, [1.0], [c2])
        self.assertRaises(InvalidDistributionInput, self.prior.isValid, m3)

        c3 = ProductDistribution([DiscreteDistribution(4, [0.25] * 4),
            MultinomialDistribution(3, 4, [0.1, 0.6, 0.1, 0.2]),
            NormalDistribution(-2.0, 0.5)])
        m4 = MixtureModel(1, [1.0], [c3])
        self.assertRaises(InvalidDistributionInput, self.prior.isValid, m4)


class ProductDistributionTests(BaseTest):
    """
    Tests for class ProductDistribution.
    """

    def setUp(self):
        self.prod = ProductDistribution([DiscreteDistribution(4, [0.25] * 4),
            MultinomialDistribution(3, 4, [0.1, 0.6, 0.1, 0.2]),
            NormalDistribution(-2.0, 0.5),
            ExponentialDistribution(1.0),
            MultiNormalDistribution(2, [0.0, 1.0], [[0.3, 0.2], [0.1, 1.0]])])

    def testpdf(self):
        l = [['0', '0', '1', '2', -2.3, 0.1, 0.2, 1.1], ['1', '1', '1', '1', 0.3, 0.14, 0.6, 1.3], ['2', '0', '1', '2', -3.1, 0.4, -0.2, 0.8]]

        data = DataSet()
        data.fromList(l)

        m = MixtureModel(1, [1.0], [self.prod])
        data.internalInit(m)
        self.assertEqual(self.prod.pdf(data), [-8.27554718, -15.66059967, -10.82090432])

    def testisvalid(self):
        l = [['0', '0', '1', '2', -2.3, 0.1, 0.2, 1.1], ['4', '1', '1', '1', 0.3, 0.14, 0.6, 1.3],
            ['2', '0', '1', '2', -3.1, 0.4, 0.8], ['1', '0', '2', -3.1, 0.4, 0.8, 0.78],
            ['2', '0', '1', '2', -3.1, -1.0, 0.4, 0.8]]

        self.prod.isValid(l[0])
        self.assertRaises(InvalidDistributionInput, self.prod.isValid, l[1])
        self.assertRaises(InvalidDistributionInput, self.prod.isValid, l[2])
        self.assertRaises(InvalidDistributionInput, self.prod.isValid, l[3])
        self.assertRaises(InvalidDistributionInput, self.prod.isValid, l[4])

#    def testeq(self):
#        raise NotImplementedError
#
#    def testcopy(self):
#        raise NotImplementedError


#    def testmstep(self):
#        raise NotImplementedError
#
#    def testsample(self):
#        raise NotImplementedError





class MixtureModelPartialLearningTests(BaseTest):
    """
    Tests for semi-supervised learning in classes LabeledMixtureModel and ConstrainedMixtureModel.
    """

    def testemfixlabels(self):
        l = [[3.6704548984145786], [7.4850458858888267], [5.5679870244380885], [1.8948738587656102], [6.3796597596325251], [6.7740700319588667], [5.3324147712221572], [6.0053993914010819],
            [6.0004977866626836], [2.3912721528162093], [1.2617720906388765], [2.2965179694073492], [6.2011150415322831]]
        dat = ConstrainedDataSet()
        dat.fromList(l)

        fixedlabels = [[0, 2, 6]]
        dat.setConstrainedLabels(fixedlabels)

        n1 = ProductDistribution([NormalDistribution(4.5, 1.5)])
        n2 = ProductDistribution([NormalDistribution(8.0, 1.8)])

        mpi = [0.4, 0.6]
        train = LabeledMixtureModel(2, mpi, [n1, n2])

        train.EM(dat, 10, 0.1, silent=1)
        c = train.classify(dat, silent=1)

        self.assertEqual(c, [0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1])

        self.assertEqual(train.pi, [0.66878953, 0.33121047])
        self.assertEqual(train.components[0][0], {"mu": 3.84041070987, "sigma": 1.93327701521})
        self.assertEqual(train.components[1][0], {"mu": 6.47311765018, "sigma": 0.50132494554})


    def testemconstraints(self):
        l = [[3.6704548984145786], [7.4850458858888267], [4.8679870244380885], [1.8948738587656102], [6.3796597596325251], [6.7740700319588667], [5.3324147712221572], [6.0053993914010819],
            [6.0004977866626836], [2.3912721528162093], [1.2617720906388765], [2.2965179694073492], [6.2011150415322831]]
        dat = ConstrainedDataSet()
        dat.fromList(l)

        pos_constr = numpy.zeros((13, 13), dtype='Float64')
        pos_constr[0, 2] = 1.0
        pos_constr[2, 0] = 1.0
        #        pos_constr[0,6] = 1.0
        #        pos_constr[6,0] = 1.0
        #        pos_constr[2,6] = 1.0
        #        pos_constr[6,2] = 1.0

        neg_constr = numpy.zeros((13, 13), dtype='Float64')
        #        neg_constr[0,2] = 1.0
        #        neg_constr[2,0] = 1.0
        neg_constr[0, 6] = 1.0
        neg_constr[6, 0] = 1.0
        neg_constr[2, 6] = 1.0
        neg_constr[6, 2] = 1.0

        dat.setPairwiseConstraints(pos_constr, neg_constr)

        n1 = ProductDistribution([NormalDistribution(4.5, 1.5)])
        n2 = ProductDistribution([NormalDistribution(6.6, 1.8)])

        mpi = [0.5, 0.5]

        train = ConstrainedMixtureModel(2, mpi, [n1, n2])
        random.seed(1)
        p = train.modelInitialization(dat, 100, 100, 3, rtype=1)

        #posterior = numpy.zeros((2,13),dtype='Float64')

        [log_l, log_p] = train.EM(dat, 100, 0.1, 100, 100, p, 3, silent=1)

        c = train.classify(dat, 100, 100, numpy.exp(log_l), 3, silent=1)
        self.assertEqual(c, [0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1])
        self.assertEqual(train.pi, [0.46997507, 0.53002493])
        self.assertEqual(train.components[0][0], {"mu": 2.79334411699, "sigma": 1.27604735923})
        self.assertEqual(train.components[1][0], {"mu": 6.31242489602, "sigma": 0.630612790811})


class BayesMixtureModelTests(BaseTest):
    """
    Tests for class BayesMixtureModel.
    """

    def setUp(self):
        G = 3
        p = 4
        # Bayesian Mixture with three components and four discrete features
        piPrior = DirichletPrior(G, [1.0] * G)

        compPrior = [DirichletPrior(4, [2.02, 1.02, 3.02, 1.42]), DirichletPrior(4, [1.02, 2.02, 3.02, 1.72]),
            DirichletPrior(4, [1.52, 1.72, 2.02, 1.02]), DirichletPrior(4, [1.52, 2.02, 1.02, 5.02])]

        mixPrior = MixtureModelPrior(0.1, 0.1, piPrior, compPrior)

        self.DNA = Alphabet(['A', 'C', 'G', 'T'])
        random.seed(3586662)
        _C_mixextend.set_gsl_rng_seed(3586662)

        comps = [ProductDistribution([DiscreteDistribution(4, [0.12753376538325797, 0.50616311046923801, 0.20698023329453394, 0.15932289085297005], self.DNA),
            DiscreteDistribution(4, [0.42479119171723906, 0.30837836771484017, 0.24099154348478288, 0.025838897083137909], self.DNA),
            DiscreteDistribution(4, [0.072678020806599297, 0.74303442328419766, 0.17342774138519285, 0.010859814524010233], self.DNA),
            DiscreteDistribution(4, [0.41290501940439173, 0.26045689115571397, 0.026675238506973845, 0.29996285093292041], self.DNA)]),
            ProductDistribution([DiscreteDistribution(4, [0.47242332711053769, 0.05503582379855871, 0.032325320190801962, 0.44021552890010163], self.DNA),
                DiscreteDistribution(4, [0.26615339532148219, 0.34344782681526098, 0.26154605901673872, 0.12885271884651808], self.DNA),
                DiscreteDistribution(4, [0.26917515170707557, 0.057407992738203517, 0.60606234492675792, 0.067354510627962966], self.DNA),
                DiscreteDistribution(4, [0.10597559898421849, 0.54316942248339628, 0.011512954838178956, 0.3393420236942063], self.DNA)]),
            ProductDistribution([DiscreteDistribution(4, [0.34086877560660928, 0.17411602775602431, 0.19669116610293322, 0.28832403053443317], self.DNA),
                DiscreteDistribution(4, [0.38241462847792057, 0.16967035154094584, 0.29945566811349167, 0.14845935186764186], self.DNA),
                DiscreteDistribution(4, [0.057210950173245387, 0.34948659755634709, 0.57122905601480312, 0.022073396255604422], self.DNA),
                DiscreteDistribution(4, [0.0081482169238704712, 0.85961621432433211, 0.034233188370958081, 0.098002380380839305], self.DNA)])]

        pi = [0.11792710681100915, 0.27129691282064256, 0.61077598036834835]

        self.m = BayesMixtureModel(G, pi, comps, mixPrior, struct=1)

        self.data = DataSet()
        l = [['A', 'G', 'A', 'T'], ['A', 'G', 'A', 'A'], ['G', 'T', 'G', 'A'], ['G', 'G', 'A', 'G'], ['C', 'T', 'T', 'A'],
            ['C', 'T', 'G', 'A'], ['T', 'G', 'G', 'T'], ['G', 'C', 'G', 'T'], ['T', 'A', 'G', 'G'], ['A', 'G', 'G', 'A'],
            ['G', 'G', 'G', 'T'], ['C', 'G', 'G', 'A'], ['A', 'T', 'G', 'T'], ['C', 'G', 'G', 'A'], ['C', 'G', 'G', 'T'],
            ['A', 'C', 'A', 'T'], ['C', 'G', 'G', 'C'], ['C', 'T', 'A', 'A'], ['A', 'G', 'G', 'A'], ['A', 'C', 'T', 'A'],
            ['A', 'G', 'A', 'A'], ['A', 'G', 'T', 'G'], ['G', 'T', 'G', 'A'], ['T', 'C', 'C', 'A'], ['A', 'C', 'G', 'T'],
            ['C', 'G', 'G', 'T'], ['T', 'A', 'G', 'T'], ['T', 'C', 'T', 'C'], ['G', 'T', 'G', 'T'], ['T', 'G', 'A', 'A'],
            ['A', 'T', 'C', 'A'], ['G', 'G', 'C', 'G'], ['G', 'C', 'A', 'T'], ['A', 'C', 'G', 'T'], ['A', 'G', 'C', 'G'],
            ['C', 'C', 'C', 'T'], ['G', 'G', 'T', 'A'], ['C', 'T', 'G', 'T'], ['A', 'G', 'G', 'A'], ['G', 'G', 'A', 'G']]
        self.data.fromList(l)
        self.data.internalInit(self.m)

        l2 = [[0.84415734210279203, 3.4831252651390257, -2.0114691015214223, '1'],
            [0.57222942131134247, -6.3745496428062598, -2.8601051092050338, '3'],
            [1.1034642666571195, -5.3577275348876148, -3.1146891814304389, '2'],
            [-0.19927292611462444, -1.1937256801018536, -2.7907449952182222, '1'],
            [0.97243650579331242, 1.7350289551506874, -3.6354683780255805, '2'],
            [0.99177414545803999, -5.7626163860784603, -3.3466535599578462, '2'],
            [1.0060590118349366, -5.7890643159612312, -2.2450385318798691, '3'],
            [1.5835914152104689, 1.5961551732250574, -2.7523093515605876, '3'],
            [1.0031284888519361, -7.1523637186251907, -3.8985888193923208, '2'],
            [1.4908256849813117, 1.3877005773065594, -3.1067879105457417, '2'],
            [0.69214327179122326, -0.30344817528702617, -2.7973949321854077, '2'],
            [0.91916318214912751, 3.1000346363444757, -2.8666720793510647, '0'],
            [2.345633352473679, 0.16087200224272347, -2.6257350900065619, '3'],
            [0.95235835374739464, 1.4732654415792585, -3.2545450565085483, '0'],
            [1.602157719545996, 2.8661497991259446, -3.0635175786389524, '1'],
            [0.69299355409901375, -5.3246941979332627, -3.1479143643427232, '2'],
            [1.0892917304906975, -6.1040097897649792, -2.8655423737527177, '3'],
            [0.2005153638949998, 0.52479824903375505, -3.8636374393910309, '1'],
            [1.1678705443107265, -0.37585185162193335, -3.0787761392845416, '3'],
            [0.88892153171856414, -5.5167558427884877, -2.6124201493076677, '3'],
            [-0.041974139334297167, 4.2507769624060083, -3.4465950645179126, '2'],
            [1.1515887636499875, 2.7972154367508821, -3.2448434191382911, '3'],
            [1.3680765819454179, -6.1971982186874248, -2.3337352134059639, '3'],
            [0.64791981966297785, -5.3058036013081082, -1.7402431491674624, '2'],
            [1.3379548269757844, 2.5366270204479782, -2.9907335954687384, '0'],
            [0.73718903725569729, 1.7476289380420684, -2.7259761936547089, '2'],
            [0.51046926111166369, 0.30877915224043528, -2.7541222082323213, '2'],
            [0.55635267700522939, 3.0615003341802636, -1.5241436184074817, '2'],
            [1.3010558053932728, -6.4069872160494779, -3.6973625379097883, '3'],
            [1.7623420534183178, -6.2936580073616213, -2.1875623544962508, '2'],
            [0.45923118840055943, 0.34066275417435632, -2.6072343220803438, '1'],
            [0.56386731177265648, 5.1028744187156487, -3.1729979696411719, '0'],
            [0.63053217827641972, 3.1550913701986953, -2.8000747628496803, '1'],
            [0.74349439801247064, 5.4136966332598302, -2.6245681527263272, '3']]

        self.data2 = DataSet()
        self.data2.fromList(l2)

        tn11 = NormalDistribution(1.6, 0.65)
        tn12 = NormalDistribution(1.0, 0.35)
        tn13 = NormalDistribution(-3.2, 0.58)
        td14 = DiscreteDistribution(4, [0.25] * 4)
        tc1 = ProductDistribution([tn11, tn12, tn13, td14])

        tn21 = NormalDistribution(5.0, 0.9)
        tn22 = NormalDistribution(-6.4, 0.54)
        tn23 = NormalDistribution(1.2, 0.5)
        td24 = DiscreteDistribution(4, [0.25] * 4)
        tc2 = ProductDistribution([tn21, tn22, tn23, td24])

        tpi = [0.8, 0.2]
        sp1 = NormalGammaPrior(1.0, 1.0, 1.0, 1.0)
        sp2 = NormalGammaPrior(1.0, 1.0, 1.0, 1.0)
        sp3 = NormalGammaPrior(1.0, 1.0, 1.0, 1.0)
        sp4 = DirichletPrior(4, [1.02] * 4)
        pipr = DirichletPrior(2, [1.0, 1.0])
        prior = MixtureModelPrior(0.03, 0.03, pipr, [sp1, sp2, sp3, sp4])
        self.m2 = BayesMixtureModel(2, tpi, [tc1, tc2], prior, struct=1)
        self.data2.internalInit(self.m2)
        prior.compPrior.priorList[0].setParams(self.data2.getInternalFeature(0), 2)
        prior.compPrior.priorList[1].setParams(self.data2.getInternalFeature(1), 2)
        prior.compPrior.priorList[2].setParams(self.data2.getInternalFeature(2), 2)


    def testcopy(self):
        cp = copy.copy(self.m)

    def testeq(self):
        G = 3
        p = 4
        # Bayesian Mixture with three components and four discrete features
        piPrior = DirichletPrior(G, [1.0] * G)
        compPrior = [DirichletPrior(4, [2.02, 1.02, 3.02, 1.42]), DirichletPrior(4, [1.02, 2.02, 3.02, 1.72]),
            DirichletPrior(4, [1.52, 1.72, 2.02, 1.02]), DirichletPrior(4, [1.52, 2.02, 1.02, 5.02])]
        mixPrior = MixtureModelPrior(0.15, 0.15, piPrior, compPrior)

        self.DNA = Alphabet(['A', 'C', 'G', 'T'])

        comps = [ProductDistribution([DiscreteDistribution(4, [0.12753376538325797, 0.50616311046923801, 0.20698023329453394, 0.15932289085297005], self.DNA),
            DiscreteDistribution(4, [0.42479119171723906, 0.30837836771484017, 0.24099154348478288, 0.025838897083137909], self.DNA),
            DiscreteDistribution(4, [0.072678020806599297, 0.74303442328419766, 0.17342774138519285, 0.010859814524010233], self.DNA),
            DiscreteDistribution(4, [0.41290501940439173, 0.26045689115571397, 0.026675238506973845, 0.29996285093292041], self.DNA)]),
            ProductDistribution([DiscreteDistribution(4, [0.47242332711053769, 0.05503582379855871, 0.032325320190801962, 0.44021552890010163], self.DNA),
                DiscreteDistribution(4, [0.26615339532148219, 0.34344782681526098, 0.26154605901673872, 0.12885271884651808], self.DNA),
                DiscreteDistribution(4, [0.26917515170707557, 0.057407992738203517, 0.60606234492675792, 0.067354510627962966], self.DNA),
                DiscreteDistribution(4, [0.10597559898421849, 0.54316942248339628, 0.011512954838178956, 0.3393420236942063], self.DNA)]),
            ProductDistribution([DiscreteDistribution(4, [0.34086877560660928, 0.17411602775602431, 0.19669116610293322, 0.28832403053443317], self.DNA),
                DiscreteDistribution(4, [0.38241462847792057, 0.16967035154094584, 0.29945566811349167, 0.14845935186764186], self.DNA),
                DiscreteDistribution(4, [0.057210950173245387, 0.34948659755634709, 0.57122905601480312, 0.022073396255604422], self.DNA),
                DiscreteDistribution(4, [0.0081482169238704712, 0.85961621432433211, 0.034233188370958081, 0.098002380380839305], self.DNA)])]

        pi = [0.11792710681100915, 0.27129691282064256, 0.61077598036834835]

        m2 = BayesMixtureModel(G, pi, comps, mixPrior, struct=1)

        self.assertEqual(self.m == m2, False)

        m2.prior.nrCompPrior = numpy.log(0.1)
        m2.prior.structPrior = numpy.log(0.1)

        self.assertEqual(self.m == m2, True)

    def testbayesmixturemapem(self):
        #print self.m

        self.m.mapEM(self.data, 40, 0.1, silent=1)

        #print self.m
        #self.assertEqual(self.m.pi, [0.27471411,0.28258675,0.44269914])
        self.assertEqual(self.m.components[1].distList[2].phi, [0.31015039, 0.2236957, 0.34025197, 0.12590194])
        self.assertEqual(self.m.components[2].distList[3].phi, [0.53186597, 0.07784179, 0.00337653, 0.38691572])

    def testbayesmixrandmaxmapem(self):
        self.m.randMaxTraining(self.data, 3, 40, 0.1, silent=1)

    def testbayesupdatestructure(self):
        # use uniform prior
        piPrior = DirichletPrior(3, [1.0] * 3)
        compPrior = [DirichletPrior(4, [1.02] * 4)] * 4
        mixPrior = MixtureModelPrior(0.01, 0.01, piPrior, compPrior)
        self.m.prior = mixPrior

        self.m.mapEM(self.data, 40, 0.1, silent=1)
        c = self.m.updateStructureBayesian(self.data, silent=1)
        self.assertEqual(c, 6)
        self.assertEqual(self.m.leaders, [[0, 2], [0, 2], [0], [0]])
        self.assertEqual(self.m.groups, [{0: [1], 2: []}, {0: [1], 2: []}, {0: [1, 2]}, {0: [1, 2]}])

        self.assertEqual(self.m.pi, [0.16045824, 0.38332446, 0.4562173])
        self.assertEqual(self.m.components[0].distList[1].phi, [0.09299761, 0.41397815, 0.49067351, 0.00235072])
        self.assertEqual(self.m.components[1].distList[3].phi, [0.4246507, 0.0503992, 0.1501996, 0.3747505])

        logp = self.m.mapEM(self.data, 1, 0.1, silent=1)
        self.assertEqual(logp[1], -210.317357795)

    def testbayesnormgamma(self):
        #print self.m2
        self.m2.mapEM(self.data2, 30, 0.1, silent=1)
        #print self.m2

        self.assertEqual(self.m2.pi, [0.35293918, 0.64706082])
        self.assertEqual(self.m2.components[0].distList[0], {"mu": 1.03551857564, "sigma": 0.267379671329})
        self.assertEqual(self.m2.components[0].distList[1], {"mu": -5.96118416543, "sigma": 0.655891604822})
        self.assertEqual(self.m2.components[0].distList[2], {"mu": -2.83752030449, "sigma": 0.508436811645})
        self.assertEqual(self.m2.components[0].distList[3], {"M": 4, "phi": [0.00165564, 0.00165564, 0.49834223, 0.49834649]})

        self.assertEqual(self.m2.components[1].distList[0], {"mu": 0.873675185386, "sigma": 0.504040070827})
        self.assertEqual(self.m2.components[1].distList[1], {"mu": 1.96093124967, "sigma": 1.59585315625})
        self.assertEqual(self.m2.components[1].distList[2], {"mu": -2.89718774482, "sigma": 0.434882981521})
        self.assertEqual(self.m2.components[1].distList[3], {"M": 4, "phi": [0.18206466, 0.27264409, 0.31793651, 0.22735474]})

    def testbayesstructureem(self):
        self.m.bayesStructureEM(self.data, 2, 2, 40, 0.1, silent=1)

        self.assertEqual(self.m.G, 3)
        self.assertEqual(self.m.leaders, [[0, 1], [0, 1], [0], [0, 2]])
        self.assertEqual(self.m.groups, [{0: [2], 1: []}, {0: [2], 1: []}, {0: [1, 2]}, {0: [1], 2: []}])
        self.assertEqual(self.m.components[0][0], {"M": 4, "phi": [0.37830054, 0.27666408, 0.28029575, 0.06473962]})
        self.assertEqual(self.m.components[0][1], {"M": 4, "phi": [0.00066029, 0.18665376, 0.55106449, 0.26162146]})
        self.assertEqual(self.m.components[0][3], {"M": 4, "phi": [0.11779314, 0.13479561, 0.14112911, 0.60628214]})

    def testklfeatureranks(self):
    #        print '--------------------'
    #        for g in range(self.m.G):
    #            print '[',
    #            for j in range(self.m.dist_nr):
    #                print 'mixture.DiscreteDistribution(4,',list(self.m.components[g][j].phi),',self.DNA),',
    #            print ']'
    #
    #        print list(self.m.pi)
    #
    #        print self.data
    #        print '--------------------'


        ranks = self.m.KLFeatureRanks(self.data, [0], silent=1)
        _testTupleLists(self, ranks, [(0.14427839406237647, 0), (0.13474946073624822, 2), (0.074990561644492151, 3), (0.058614197665394219, 1)], 14)

        ranks = self.m.KLFeatureRanks(self.data, [0, 2], silent=1)
        _testTupleLists(self, ranks, [(0.26090883547682586, 0), (0.17545191359121187, 3), (0.15731239730737478, 2), (0.024072778105738678, 1)], 14)

        ranks = self.m.KLFeatureRanks(self.data, [1, 2], silent=1)
        _testTupleLists(self, ranks, [(0.14427839406237647, 0), (0.13474946073624822, 2), (0.074990561644492151, 3), (0.058614197665394219, 1)], 14)

    def testminimalstructure(self):
        self.m.minimalStructure()
        self.assertEqual(self.m.leaders, [[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]])
        self.assertEqual(self.m.groups, [{0: [], 1: [], 2: []}, {0: [], 1: [], 2: []}, {0: [], 1: [], 2: []}, {0: [], 1: [], 2: []}])

        self.m.leaders = [[0, 2], [0], [0, 2], [0]]
        self.m.groups = [{0: [1], 2: []}, {0: [1, 2]}, {0: [1], 2: []}, {0: [1, 2]}]
        self.m.minimalStructure()
        self.assertEqual(self.m.leaders, [[0, 1], [0], [0, 1], [0]])
        self.assertEqual(self.m.groups, [{0: [], 1: []}, {0: [1]}, {0: [], 1: []}, {0: [1]}])
        self.assertEqual(self.m.G, 2)
        self.assertEqual(self.m.freeParams, 19)

        self.setUp() # reset m

        self.m.leaders = [[0], [0], [0], [0]]
        self.m.groups = [{0: [1, 2]}, {0: [1, 2]}, {0: [1, 2]}, {0: [1, 2]}]
        self.m.minimalStructure()
        self.assertEqual(self.m.leaders, [[0], [0], [0], [0]])
        self.assertEqual(self.m.groups, [{0: []}, {0: []}, {0: []}, {0: []}])
        self.assertEqual(self.m.G, 1)
        self.assertEqual(self.m.freeParams, 12)


class MixtureModelTests(BaseTest):
    """
    Tests for class MixtureModel.
    """

    def setUp(self):
        self.DIAG = Alphabet(['.', '0', '8', '1'])
        n1 = NormalDistribution(2.5, 0.5)
        n2 = NormalDistribution(6.0, 0.8)

        mult1 = MultinomialDistribution(3, 4, [0.23, 0.26, 0.26, 0.25], alphabet=self.DIAG)
        mult2 = MultinomialDistribution(3, 4, [0.7, 0.1, 0.1, 0.1], alphabet=self.DIAG)

        c1 = ProductDistribution([n1, mult1])
        c2 = ProductDistribution([n2, mult2])

        mpi = [0.4, 0.6]
        self.m = MixtureModel(2, mpi, [c1, c2])

        n1 = NormalDistribution(2.5, 0.5)
        mult1 = MultinomialDistribution(3, 4, [0.23, 0.26, 0.26, 0.25], alphabet=self.DIAG)
        d1 = DiscreteDistribution(4, [0.23, 0.26, 0.26, 0.25], alphabet=self.DIAG)
        e1 = ExponentialDistribution(1.0)
        mn1 = MultiNormalDistribution(2, [1.0, 2.0], [[1.0, 0.5], [0.5, 1.0]])

        n2 = NormalDistribution(-2.0, 1.0)
        mult2 = MultinomialDistribution(3, 4, [0.5, 0.2, 0.2, 0.1], alphabet=self.DIAG)
        d2 = DiscreteDistribution(4, [0.1, 0.3, 0.1, 0.5], alphabet=self.DIAG)
        e2 = ExponentialDistribution(2.0)
        mn2 = MultiNormalDistribution(2, [1.5, 2.0], [[0.8, 0.1], [0.2, 1.0]])

        c1 = ProductDistribution([n1, mult1, d1, e1, mn1])
        c2 = ProductDistribution([n2, mult2, d2, e2, mn2])

        pi = [0.4, 0.6]
        self.m_all = MixtureModel(2, pi, [c1, c2])


    #    def testmixturesuffstatattributes(self):
    #        print "----------------------"
    #        print "p:",self.m.p
    #        print "suff_p:",self.m.suff_p
    #        print self.m.components[0].dataRange
    #        print self.m.components[0].suff_dataRange


    def testeq(self):
        n1 = NormalDistribution(1.5, 0.5)
        n2 = NormalDistribution(6.0, 0.8)

        mult1 = MultinomialDistribution(3, 4, [0.23, 0.26, 0.26, 0.25], alphabet=self.DIAG)
        mult2 = MultinomialDistribution(3, 4, [0.1, 0.1, 0.7, 0.1], alphabet=self.DIAG)

        c1 = ProductDistribution([n1, mult1])
        c2 = ProductDistribution([n2, mult2])

        mpi = [0.5, 0.5]
        m2 = MixtureModel(2, mpi, [c1, c2])

        self.assertEqual(self.m == m2, False)

        m2.pi = numpy.array([0.4, 0.6], dtype='Float64')
        self.assertEqual(self.m == m2, False)

        m2.components[0][0].mu = 2.5
        self.assertEqual(self.m == m2, False)

        m2.components[1][1] = MultinomialDistribution(3, 4, [0.7, 0.1, 0.1, 0.1], alphabet=self.DIAG)
        self.assertEqual(self.m == m2, True)

    def testmixtureallem(self):
        """
        Mixture with all (non-sequence) atomar distributions available
        """

        data = DataSet()
        l = [[1.8918773044356572, '.', '8', '1', '0', 2.35442010455411, 0.90857251768321845, 2.9084647635570593],
            [2.6859039639792819, '.', '0', '8', '8', 0.89717111717432541, -1.1136763322054235, 0.37710939034334223],
            [-4.2181695000125945, '0', '8', '.', '0', 0.0005856354546493693, 1.3853193731036311, -2.6185019231713618],
            [-2.8585062321976942, '0', '.', '.', '1', 0.0076519556727788502, 0.60415784410716444, -2.8440534334104672],
            [-3.4956005043854663, '.', '.', '.', '0', 0.37513230719776602, 1.3041954018639283, -3.6982663911648652],
            [2.5688378507218079, '0', '0', '0', '0', 0.88412393351305663, 0.063378905069815916, 2.3743538328032545],
            [-1.8637506878905052, '8', '.', '0', '1', 2.457269626806335, 1.2937192355764773, -3.0510625377277831],
            [-2.1194235665956582, '8', '.', '0', '.', 0.12208677615248997, 1.4900664959149097, -1.0497481687211503],
            [-3.0127831596421233, '8', '8', '1', '1', 0.10916121666047895, 1.5688341195062365, -2.0213614014671828],
            [-1.8208451754577171, '.', '.', '.', '1', 0.33004234771593577, 1.2932501549941697, -1.5949438801128235],
            [-2.3703822818537432, '8', '1', '0', '1', 0.22552033717768097, 1.6757118060694636, -1.7695330070738791],
            [-2.1159485034003769, '8', '1', '0', '0', 0.61328249735749418, 0.4234334770382131, -2.315505721339401],
            [-2.1347125472920219, '.', '.', '.', '0', 0.020479663095155396, 0.99321104597773879, -1.1629724459472346]]
        data.fromList(l)

        gn1 = NormalDistribution(-1.5, 0.9)
        gmult1 = MultinomialDistribution(3, 4, [0.1, 0.4, 0.4, 0.1], alphabet=self.DIAG)
        gd1 = DiscreteDistribution(4, [0.5, 0.1, 0.1, 0.3], alphabet=self.DIAG)
        ge1 = ExponentialDistribution(0.6)
        gmn1 = MultiNormalDistribution(2, [0.7, 2.2], [[1.0, 0.6], [0.8, 1.2]])

        gn2 = NormalDistribution(3.0, 1.1)
        gmult2 = MultinomialDistribution(3, 4, [0.2, 0.2, 0.2, 0.4], alphabet=self.DIAG)
        gd2 = DiscreteDistribution(4, [0.3, 0.3, 0.1, 0.3], alphabet=self.DIAG)
        ge2 = ExponentialDistribution(2.8)
        gmn2 = MultiNormalDistribution(2, [1.3, 1.0], [[0.9, 0.5], [0.6, 1.3]])

        gc1 = ProductDistribution([gn1, gmult1, gd1, ge1, gmn1])
        gc2 = ProductDistribution([gn2, gmult2, gd2, ge2, gmn2])

        gpi = [0.4, 0.6]
        g = MixtureModel(2, gpi, [gc1, gc2])

        g.EM(data, 40, 0.1, silent=1)
        self.assertEqual(g.pi, [0.23076923, 0.76923077])


        #        print g.components[0].distList[0]
        #        print g.components[0].distList[1]
        #        print g.components[0].distList[2]
        #        print g.components[0].distList[3]
        #        print g.components[0].distList[4]


        self.assertEqual(g.components[0].distList[0], {"mu": 2.38220637305, "sigma": 0.34999339552})
        self.assertEqual(g.components[0].distList[1], {"M": 4, "p": 3, "phi": [0.22222222, 0.44444444, 0.22222222, 0.11111111]})
        self.assertEqual(g.components[0].distList[2], {"M": 4, "phi": [2.49875062e-04, 6.66333500e-01, 3.33166750e-01, 2.49875062e-04]})
        self.assertEqual(g.components[0].distList[3], {"lambd": [0.725388448525]})
        self.assertAlmostEqual(g.components[0].distList[4], {
            "mu": [-0.047241636484129997, 1.88664266223455],
            "sigma": [[0.68770018725283, 0.88014718559524996], [0.88014718559524996, 1.1868910971984099]]
        }, places=14)
        # self.assertEqual( [ round(m,14) for m in g.components[0].distList[4].mu], [-0.047241636484129997,1.88664266223455])
        # self.assertEqual( [round(s,14) for s in g.components[0].distList[4].sigma[0]] , [0.68770018725283,0.88014718559524996])
        # self.assertEqual( [round(s,14) for s in g.components[0].distList[4].sigma[1]] , [0.88014718559524996,1.1868910971984099])




        self.assertEqual(g.components[1].distList[0], {"mu": -2.60101221587, "sigma": 0.744405770895})
        self.assertEqual(g.components[1].distList[1], {"M": 4, "p": 3, "phi": [0.46666667, 0.2, 0.23333333, 0.1]})
        self.assertEqual(g.components[1].distList[2], {"M": 4, "phi": [9.99750062e-02, 3.99900025e-01, 2.49937516e-04, 4.99875031e-01]})
        self.assertEqual(g.components[1].distList[3], {"lambd": [2.34674997335]})
        self.assertAlmostEqual(g.components[1].distList[4], {
            "mu": [1.2031898954151901, -2.2125948910136199],
            "sigma": [[0.15099010970408, 0.060671540603359997], [0.060671540603359997, 0.65526254779920001]]
        }, places=14)
        # self.assertEqual( [ round(m,14) for m in g.components[1].distList[4].mu], [1.2031898954151901,-2.2125948910136199])
        # self.assertEqual( [round(s,14) for s in g.components[1].distList[4].sigma[0]] , [0.15099010970408,0.060671540603359997])
        # self.assertEqual( [round(s,14) for s in g.components[1].distList[4].sigma[1]] , [0.060671540603359997,0.65526254779920001])

    def testmixgaussdisc(self):
        n1 = NormalDistribution(2.5, 0.5)
        d1 = DiscreteDistribution(4, [0.23, 0.26, 0.26, 0.25], alphabet=self.DIAG)
        n2 = NormalDistribution(-2.0, 1.0)
        d2 = DiscreteDistribution(4, [0.1, 0.3, 0.1, 0.5], alphabet=self.DIAG)

        c1 = ProductDistribution([n1, d1])
        c2 = ProductDistribution([n2, d2])

        pi = [0.4, 0.6]
        m = MixtureModel(2, pi, [c1, c2])

        data = DataSet()
        l = [[2.3067148530759196, '1'],
            [-3.3937899938608864, '0'],
            [2.1453548445915058, '1'],
            [2.5782106621412639, '.'],
            [3.2695657225076191, '1'],
            [-3.7947205133889188, '1'],
            [2.6609272635699091, '.'],
            [-3.037791980230244, '1'],
            [-2.7921060507157986, '0'],
            [-1.005585825166154, '0'],
            [-3.9715187562361702, '.'],
            [3.0876112202307135, '8'],
            [-2.3418461383413289, '1'],
            [-3.2783952111130494, '1'],
            [-1.0453054086838787, '0']]
        data.fromList(l)
        data.internalInit(m)

        gn1 = NormalDistribution(2.5, 0.5)
        gd1 = DiscreteDistribution(4, [0.23, 0.26, 0.26, 0.25], alphabet=self.DIAG)
        gn2 = NormalDistribution(-2.0, 1.0)
        gd2 = DiscreteDistribution(4, [0.1, 0.3, 0.1, 0.5], alphabet=self.DIAG)
        gc1 = ProductDistribution([gn1, gd1])
        gc2 = ProductDistribution([gn2, gd2])
        gpi = [0.2, 0.8]
        g = MixtureModel(2, gpi, [gc1, gc2])
        g.EM(data, 40, 0.1, silent=1)

        self.assertEqual(g.pi, [0.39999861, 0.60000139])
        self.assertEqual(g.components[0].distList[0], {"mu": 2.67473242494, "sigma": 0.397801961454})
        self.assertEqual(g.components[0].distList[1], {"M": 4, "phi": [3.33251094e-01, 2.49937516e-04, 1.66625589e-01, 4.99873379e-01]})
        self.assertEqual(g.components[1].distList[0], {"mu": -2.74010633306, "sigma": 1.02615649886})
        self.assertEqual(g.components[1].distList[1], {"M": 4, "phi": [1.11083139e-01, 4.44332332e-01, 2.49937516e-04, 4.44334591e-01]})


    def testem(self):
        l = [[3.6704548984145786, '8', '1', '1'], [7.4850458858888267, '1', '.', '.'], [5.5679870244380885, '.', '.', '.'], [1.8948738587656102, '0', '0', '0'], [6.3796597596325251, '.', '.', '.'],
            [6.7740700319588667, '.', '1', '.'], [5.3324147712221572, '.', '.', '.'], [6.0053993914010819, '.', '.', '8'], [6.0004977866626836, '.', '.', '.'], [2.3912721528162093, '1', '0', '8'],
            [1.2617720906388765, '8', '1', '.'], [2.2965179694073492, '1', '8', '0'], [6.2011150415322831, '1', '.', '.']]
        dat = DataSet()
        dat.fromList(l)

        n1 = NormalDistribution(4.5, 1.5)
        n2 = NormalDistribution(8.0, 1.8)

        mult1 = MultinomialDistribution(3, 4, [0.4, 0.2, 0.2, 0.2], alphabet=self.DIAG)
        mult2 = MultinomialDistribution(3, 4, [0.1, 0.1, 0.1, 0.7], alphabet=self.DIAG)

        c1 = ProductDistribution([n1, mult1])
        c2 = ProductDistribution([n2, mult2])

        mpi = [0.4, 0.6]
        train = MixtureModel(2, mpi, [c1, c2])

        train.EM(dat, 40, 0.1, silent=1)

        self.assertEqual(train.pi[0], 0.384611609936)
        self.assertEqual(train.components[0].distList[0].mu, 2.30296658077)
        self.assertEqual(train.components[1].distList[0].sigma, 0.636459666505)

    def testrandmaxem(self):
        random.seed(3586662)

        l = [[3.6704548984145786, '8', '1', '1'], [7.4850458858888267, '1', '.', '.'],
            [5.5679870244380885, '.', '.', '.'], [1.8948738587656102, '0', '0', '0'],
            [6.3796597596325251, '.', '.', '.'], [6.7740700319588667, '.', '1', '.'],
            [5.3324147712221572, '.', '.', '.'], [6.0053993914010819, '.', '.', '8'],
            [6.0004977866626836, '.', '.', '.'], [2.3912721528162093, '1', '0', '8'],
            [1.2617720906388765, '8', '1', '.'], [2.2965179694073492, '1', '8', '0'],
            [6.2011150415322831, '1', '.', '.']]
        dat = DataSet()
        dat.fromList(l)

        n1 = NormalDistribution(4.5, 1.5)
        n2 = NormalDistribution(8.0, 1.8)

        mult1 = MultinomialDistribution(3, 4, [0.4, 0.2, 0.2, 0.2], alphabet=self.DIAG)
        mult2 = MultinomialDistribution(3, 4, [0.1, 0.1, 0.1, 0.7], alphabet=self.DIAG)

        c1 = ProductDistribution([n1, mult1])
        c2 = ProductDistribution([n2, mult2])

        mpi = [0.4, 0.6]
        train = MixtureModel(2, mpi, [c1, c2])
        train.randMaxEM(dat, 3, 40, 0.1, silent=1)

        self.assertEqual(train.pi, [0.38461049, 0.61538951])
        self.assertEqual(train.components[0].distList[0], {"mu": 2.30296118599, "sigma": 0.791001832397})
        self.assertEqual(train.components[0].distList[1], {"M": 4, "p": 3, "phi": [0.06666768, 0.33333757, 0.26666578, 0.33332897]})
        self.assertEqual(train.components[1].distList[0], {"mu": 6.21825321628, "sigma": 0.63646629643})
        self.assertEqual(train.components[1].distList[1], {"M": 4, "p": 3, "phi": [8.33118326e-01, 2.49937516e-04, 4.16585950e-02, 1.24973141e-01]})

    def testrandmaxemsimple(self):
        random.seed(3586662)

        # simple gaussian model
        n1 = ProductDistribution([NormalDistribution(-2.5, 0.5)])
        n2 = ProductDistribution([NormalDistribution(6.0, 0.8)])

        pi = [0.4, 0.6]
        gen = MixtureModel(2, pi, [n1, n2])

        data = DataSet()
        l = [[-2.6751369624764929],
            [3.5541907883613257],
            [4.950009051623474],
            [4.7405049982596053],
            [-2.2523083183357127],
            [6.1391797079580561],
            [7.5525326592401942],
            [-1.8669732387994338],
            [4.8362379747532103],
            [-3.1207976176705601],
            [-2.4129098827445872],
            [-2.8350290817806276],
            [5.344556553752307],
            [5.7325743305747787],
            [6.9475338057098321],
            [5.34149828634225],
            [-2.4015517154306671],
            [-1.9519106148228067],
            [-2.3405552482017411],
            [5.8262556749424412],
            [-2.484131799010588],
            [6.2688028343995477],
            [-2.5667642785423359],
            [5.4790139643682672],
            [8.0636235373059382],
            [6.2350602489435483],
            [7.0105696968484548],
            [6.4078204530854457],
            [-2.4074733713349112],
            [5.9780662579560744]]
        data.fromList(l)
        data.internalInit(gen)

        n1 = ProductDistribution([NormalDistribution(4.5, 1.5)])
        n2 = ProductDistribution([NormalDistribution(8.0, 1.8)])
        pi = [0.7, 0.3]
        train = MixtureModel(2, pi, [n1, n2])
        train.randMaxEM(data, 3, 10, 0.1, silent=1)
        self.assertEqual(train.pi, [0.43290601, 0.56709399])
        self.assertEqual(train.components[0].distList[0], {"mu": 3.448722685, "sigma": 3.93454729029})
        self.assertEqual(train.components[1].distList[0], {"mu": 1.89876256628, "sigma": 4.23502285951})

        # simple multinomial model
        mult1 = ProductDistribution([MultinomialDistribution(3, 4, [0.23, 0.26, 0.26, 0.25], alphabet=self.DIAG)])
        mult2 = ProductDistribution([MultinomialDistribution(3, 4, [0.7, 0.1, 0.1, 0.1], alphabet=self.DIAG)])

        pi = [0.4, 0.6]
        gen = MixtureModel(2, pi, [mult1, mult2])

        data = DataSet()
        l = [['0', '8', '8'],
            ['.', '.', '.'],
            ['8', '0', '.'],
            ['8', '0', '0'],
            ['.', '8', '0'],
            ['1', '0', '8'],
            ['8', '0', '0'],
            ['.', '.', '.'],
            ['.', '0', '8'],
            ['.', '.', '.'],
            ['8', '.', '.'],
            ['0', '8', '.'],
            ['0', '.', '.'],
            ['.', '0', '.'],
            ['1', '8', '8'],
            ['8', '8', '8'],
            ['0', '8', '0'],
            ['0', '.', '.'],
            ['.', '0', '8'],
            ['.', '8', '0'],
            ['.', '.', '0'],
            ['.', '1', '.'],
            ['1', '8', '1'],
            ['8', '0', '1'],
            ['.', '.', '.'],
            ['1', '.', '1'],
            ['0', '8', '1'],
            ['0', '.', '8'],
            ['1', '.', '.'],
            ['.', '8', '1']]
        data.fromList(l)
        data.internalInit(gen)
        mult1 = ProductDistribution([MultinomialDistribution(3, 4, [0.4, 0.2, 0.2, 0.2], alphabet=self.DIAG)])
        mult2 = ProductDistribution([MultinomialDistribution(3, 4, [0.1, 0.1, 0.1, 0.7], alphabet=self.DIAG)])
        pi = [0.7, 0.3]
        train = MixtureModel(2, pi, [mult1, mult2])

        train.randMaxEM(data, 3, 10, 0.1, silent=1)
        self.assertEqual(train.pi, [0.48717885, 0.51282115])
        self.assertEqual(train.components[0].distList[0], {"M": 4, "p": 3, "phi": [0.14164446, 0.30563449, 0.40080779, 0.15191326]})
        self.assertEqual(train.components[1].distList[0], {"M": 4, "p": 3, "phi": [0.62377049, 0.16464741, 0.11756629, 0.09401581]})

    def testsimplegaussian(self):
        n1 = ProductDistribution([NormalDistribution(-2.5, 0.5)])
        n2 = ProductDistribution([NormalDistribution(6.0, 0.8)])
        pi = [0.4, 0.6]
        gen = MixtureModel(2, pi, [n1, n2])

        data = DataSet()
        l = [[6.5445727620632344],
            [5.6348882068585668],
            [6.7921989830498877],
            [4.9101521351042514],
            [5.630628653137931],
            [4.5147580317315601],
            [5.7671344316967561],
            [-1.0980438466562152],
            [6.8098198516377479],
            [-2.6674442486121523],
            [6.7101233232999657],
            [-2.0481975241250003],
            [5.7515047118056293],
            [5.7415446659617899],
            [5.9845992826802386],
            [-3.043028955276009],
            [4.3453297022194874],
            [6.4999354021546782],
            [-2.3976434882316031],
            [6.4321559627701648],
            [6.6872710683408076],
            [-2.3021334453464943],
            [-2.4347664699656537],
            [6.1960339094588992],
            [-2.4965231831296428],
            [6.3421940955410188],
            [5.4253400776950356],
            [7.4667346630068305],
            [7.4199761965051261],
            [5.7051517369462657],
            [-2.9178082614396632],
            [-2.1937366207548026],
            [-2.7578746165839543],
            [7.4680564166282482],
            [-2.8246850063884859],
            [4.7688879222961011],
            [4.9517086476170844],
            [6.2938027130469552],
            [-2.8442237489103301],
            [-2.6247374748847623]]
        data.fromList(l)
        data.internalInit(gen)

        n1 = ProductDistribution([NormalDistribution(4.5, 1.5)])
        n2 = ProductDistribution([NormalDistribution(8.0, 1.8)])
        pi = [0.7, 0.3]
        train = MixtureModel(2, pi, [n1, n2])
        train.EM(data, 40, 0.1, silent=1)

        self.assertEqual(train.pi, [0.35, 0.65])
        self.assertEqual(train.components[0].distList[0], {"mu": -2.47506049216, "sigma": 0.471475077261})
        self.assertEqual(train.components[1].distList[0], {"mu": 6.03055782897, "sigma": 0.858213254015})


    def testsimplemultinom(self):
        mult1 = ProductDistribution([MultinomialDistribution(3, 4, [0.23, 0.26, 0.26, 0.25], alphabet=self.DIAG)])
        mult2 = ProductDistribution([MultinomialDistribution(3, 4, [0.7, 0.1, 0.1, 0.1], alphabet=self.DIAG)])

        pi = [0.4, 0.6]
        gen = MixtureModel(2, pi, [mult1, mult2])

        data = DataSet()
        l = [['.', '1', '1'],
            ['.', '.', '.'],
            ['.', '1', '.'],
            ['.', '1', '.'],
            ['0', '.', '.'],
            ['8', '.', '.'],
            ['1', '.', '8'],
            ['8', '1', '0'],
            ['8', '8', '8'],
            ['.', '8', '0'],
            ['.', '.', '1'],
            ['1', '0', '8'],
            ['.', '8', '0'],
            ['.', '.', '.'],
            ['.', '.', '.'],
            ['.', '.', '.'],
            ['.', '1', '1'],
            ['.', '8', '0'],
            ['.', '.', '1'],
            ['8', '.', '0']]
        data.fromList(l)
        data.internalInit(gen)

        mult1 = ProductDistribution([MultinomialDistribution(3, 4, [0.4, 0.2, 0.2, 0.2], alphabet=self.DIAG)])
        mult2 = ProductDistribution([MultinomialDistribution(3, 4, [0.1, 0.1, 0.1, 0.7], alphabet=self.DIAG)])
        pi = [0.7, 0.3]
        train = MixtureModel(2, pi, [mult1, mult2])

        train.EM(data, 40, 0.1, silent=1)

        self.assertEqual(train.pi, [0.4191391, 0.5808609])
        self.assertEqual(train.components[0].distList[0], {"M": 4, "p": 3, "phi": [7.28784278e-01, 2.49948455e-04, 2.49948455e-04, 2.70715826e-01]})
        self.assertEqual(train.components[1].distList[0], {"M": 4, "p": 3, "phi": [0.36349785, 0.20076187, 0.31550092, 0.12023937]})


    def testsimplediscrete(self):
        d1 = ProductDistribution([DiscreteDistribution(4, [0.23, 0.26, 0.26, 0.25], alphabet=self.DIAG)])
        d2 = ProductDistribution([DiscreteDistribution(4, [0.7, 0.1, 0.1, 0.1], alphabet=self.DIAG)])

        pi = [0.4, 0.6]
        gen = MixtureModel(2, pi, [d1, d2])

        data = DataSet()
        l = [['0'], ['8'], ['1'], ['.'], ['.'], ['1'], ['1'], ['0'], ['0'], ['0'], ['8'], ['1'], ['.'], ['.'], ['1'], ['8'],
            ['.'], ['0'], ['.'], ['8'], ['.'], ['.'], ['.'], ['8'], ['0'], ['.'], ['0'], ['0'], ['0'], ['.']]
        data.fromList(l)
        data.internalInit(gen)

        d1 = ProductDistribution([DiscreteDistribution(4, [0.4, 0.2, 0.2, 0.2], alphabet=self.DIAG)])
        d2 = ProductDistribution([DiscreteDistribution(4, [0.1, 0.1, 0.1, 0.7], alphabet=self.DIAG)])
        pi = [0.7, 0.3]
        train = MixtureModel(2, pi, [d1, d2])

        print '######'
        train.EM(data, 40, 0.1, silent=1)
        print '######'

        self.assertEqual(train.pi, [0.21783681, 0.78216319])
        self.assertEqual(train.components[0].distList[0], {"M": 4, "phi": [0.16289199, 0.24303136, 0.13501742, 0.45905923]})
        self.assertEqual(train.components[1].distList[0], {"M": 4, "phi": [0.42341905, 0.31586608, 0.17548116, 0.08523371]})


    def testmixmixsimple(self):
        miss1 = ProductDistribution([NormalDistribution(-9999.9, 0.00001)])
        n1 = ProductDistribution([NormalDistribution(2.0, 1.0)])
        c1 = ProductDistribution([MixtureModel(2, [0.999, 0.001], [n1, miss1], compFix=[0, 2])])

        miss2 = ProductDistribution([NormalDistribution(-9999.9, 0.00001)])
        n2 = ProductDistribution([NormalDistribution(-2.0, 1.0)])
        c2 = ProductDistribution([MixtureModel(2, [0.999, 0.001], [n2, miss2], compFix=[0, 2])])

        pi = [0.6, 0.4]
        m = MixtureModel(2, pi, [c1, c2])

        data = DataSet()
        mat = [[3.1004406552920902], [2.0999408880314876], [3.50709400911362],
            [-1.8471191498679163], [-1.8037737760991528], [2.5982446668848018],
            [-1.9985090283079578], [1.4791597811545607], [1.8755112236896474],
            [2.6197544975483544], [-2.9493850898381195], [-0.55162909778828628],
            [1.3884447552994725], [-3.4885202101471768], [0.60564106678284402],
            [3.4912612006857096], [-1.1587236124542732], [-1.310937228183378],
            [4.4849354835783242], [-3.3831906133826672], [-2.0497596324770457],
            [2.727173233520563], [0.32825460618394531], [-2.5408454590389868],
            [-2.4282365841477254], [-2.4628481252730583], [-1.0836546076535605],
            [-1.754995395305089], [1.6242019049131544], [-0.18520975405074624],
            [4.0913096380314986], [0.88554498496615341], [-2.8625187627249886],
            [2.579306289032004], [-0.67060150137842078], [1.6519454599598533],
            [0.10472779434885759], [-1.7047226440976959], [-1.6347421696422852],
            [1.7480452785087588], [1.8471619833752921], [1.2828447596548935],
            [-1.0984291623693552], [3.197410907124477], [-1.700133507408105],
            [1.6021458690733921], [-3.0797128639796809], [3.3397325089877423],
            [-2.5128773156532209], [-2.4689570185772127]]

        data.fromList(mat)

        m.EM(data, 40, 0.1, silent=1)

        self.assertEqual(m.components[0].distList[0].components[0].distList[0].mu, -9999.9)
        self.assertEqual(m.components[1].distList[0].components[0].distList[0].sigma, 1e-05)
        self.assertEqual(m.components[0].distList[0].components[1].distList[0].mu, -1.95423960866)
        self.assertEqual(m.components[0].distList[0].components[1].distList[0].sigma, 0.877001461343)
        self.assertEqual(m.components[1].distList[0].components[1].distList[0].mu, 2.12222668098)
        self.assertEqual(m.components[1].distList[0].components[1].distList[0].sigma, 1.18534559105)

    def testmixmixgauss(self):
        n11 = ProductDistribution([NormalDistribution(2.0, 0.5)])
        n12 = ProductDistribution([NormalDistribution(6.0, 1.0)])
        c1 = ProductDistribution([MixtureModel(2, [0.5, 0.5], [n11, n12], compFix=[0, 0])])

        n21 = ProductDistribution([NormalDistribution(-2.0, 0.5)])
        n22 = ProductDistribution([NormalDistribution(-8.0, 1.0)])
        c2 = ProductDistribution([MixtureModel(2, [0.5, 0.5], [n21, n22], compFix=[0, 0])])

        pi = [0.6, 0.4]
        m = MixtureModel(2, pi, [c1, c2])

        data = DataSet()
        mat = [[2.5502203276460449], [2.0499704440157438], [2.7535470045568102],
            [-7.8471191498679165], [-7.8037737760991526], [2.2991223334424009],
            [-7.9985090283079581], [5.4791597811545607], [1.9377556118448236], [6.619754497548354],
            [-2.4746925449190598], [-6.5516290977882861], [5.3884447552994725],
            [-2.7442601050735886], [1.302820533391422], [2.7456306003428548],
            [-7.1587236124542732], [-7.310937228183378], [8.4849354835783242],
            [-2.6915953066913336], [-8.0497596324770466], [6.7271732335205634],
            [4.3282546061839451], [-8.5408454590389873], [-2.2141182920738625],
            [-8.4628481252730587], [-1.5418273038267802], [-1.8774976976525444],
            [1.8121009524565772], [-6.1852097540507458], [3.0456548190157493],
            [4.8855449849661534], [-8.8625187627249886], [2.2896531445160022],
            [-1.3353007506892105], [1.8259727299799267], [1.0523638971744287],
            [-1.852361322048848], [-1.8173710848211426], [1.8740226392543793],
            [5.8471619833752921], [5.2828447596548935], [-7.0984291623693547],
            [7.1974109071244765], [-7.700133507408105], [1.801072934536696],
            [-2.5398564319898407], [2.6698662544938712], [-8.5128773156532205],
            [-8.4689570185772123]]

        data.fromList(mat)

        tn11 = ProductDistribution([NormalDistribution(0.0, 1.1)])
        tn12 = ProductDistribution([NormalDistribution(9.0, 0.6)])
        tc1 = ProductDistribution([MixtureModel(2, [0.5, 0.5], [tn11, tn12], compFix=[0, 0])])

        tn21 = ProductDistribution([NormalDistribution(-2.0, 2.0)])
        tn22 = ProductDistribution([NormalDistribution(-5.0, 1.0)])
        tc2 = ProductDistribution([MixtureModel(2, [0.5, 0.5], [tn21, tn22], compFix=[0, 0])])

        tpi = [0.3, 0.7]
        tm = MixtureModel(2, tpi, [tc1, tc2])

        tm.EM(data, 40, 0.1, silent=1)

        self.assertEqual(tm.components[0].distList[0].components[0].distList[0].mu, -2.10888808641)
        self.assertEqual(tm.components[1].distList[0].components[0].distList[0].sigma, 1.22362958167)
        self.assertEqual(tm.components[0].distList[0].components[1].distList[0].mu, -7.77015137509)
        self.assertEqual(tm.components[0].distList[0].components[1].distList[0].sigma, 0.75197914841)
        self.assertEqual(tm.components[1].distList[0].components[1].distList[0].mu, 2.12813631991)
        self.assertEqual(tm.components[1].distList[0].components[1].distList[0].sigma, 0.538936095872)


    def testclassify(self):
        n1 = ProductDistribution([NormalDistribution(2.5, 1.5)])
        n2 = ProductDistribution([NormalDistribution(6.0, 1.8)])

        pi = [0.4, 0.6]
        m = MixtureModel(2, pi, [n1, n2])

        l = [[0.5], [3.2], [4.5], [5.5], [4.8], [2.0]]
        data = DataSet()
        data.fromList(l)

        c = m.classify(data, silent=1)
        self.assertEqual(c, [0, 0, 1, 1, 1, 0])

        c2 = m.classify(data, silent=1, entropy_cutoff=0.5)

        self.assertEqual(c2, [0, -1, -1, 1, -1, 0])

    def testisvalid(self):
        data = self.m.sampleDataSet(10)

        self.m.isValid(data.dataMatrix[0])
        self.m.isValid(data)

        data.dataMatrix[5][1] = 'A'

        self.assertRaises(InvalidDistributionInput, self.m.isValid, data)

    def testwritemixture(self):
        writeMixture(self.m_all, 'testwrite.mix', silent=True)

    def testreadmixture(self):
        m2 = readMixture('./testwrite.mix')
        self.assertEqual(m2, self.m_all)


    def testremovecomponent(self):
        g11 = NormalDistribution(1.2, 0.5)
        g12 = NormalDistribution(2.2, 0.5)
        g21 = NormalDistribution(-4.2, 0.5)
        g22 = NormalDistribution(-6.2, 0.5)
        g31 = NormalDistribution(0.0, 0.5)
        g32 = NormalDistribution(0.0, 0.5)
        g41 = NormalDistribution(0.0, 0.5)
        g42 = NormalDistribution(0.0, 0.5)
        g43 = NormalDistribution(0.0, 0.5)

        gc1 = ProductDistribution([g11, g21, g41])
        gc2 = ProductDistribution([g12, g22, g42])
        gc3 = ProductDistribution([g32, g32, g43])
        gpi = [0.0, 0.5, 0.5]
        g = MixtureModel(3, gpi, [gc1, gc2, gc3], struct=1)

        g.leaders = [[0], [0, 1], [0, 1]]
        g.groups = [{0: [1, 2]}, {0: [2], 1: []}, {0: [], 1: [2]}]
        g.removeComponent(0)
        self.assertEqual(g.leaders, [[0], [0, 1], [0]])
        self.assertEqual(g.groups, [{0: [1]}, {0: [], 1: []}, {0: [1]}])

    def testidentifiable(self):
        self.m.pi = numpy.array([0.9, 0.1], dtype='Float64')
        self.m.identifiable()
        self.assertEqual(self.m.pi, [0.1, 0.9])
        self.assertEqual(self.m.components[1][0].mu, 2.5)

        self.m.pi = numpy.array([0.9, 0.1], dtype='Float64')
        self.m.identifiable()
        self.assertEqual(self.m.pi, [0.1, 0.9])
        self.assertEqual(self.m.components[0][0].mu, 2.5)


class ModelInitTests(BaseTest):
    """
    Tests for function modelInitialization().
    """

    def setUp(self):
        self.DIAG = Alphabet(['.', '0', '8', '1'])


    def testmixturecomponentsproductatomar(self):
        random.seed(3586662)
        n1 = NormalDistribution(2.5, 0.5)
        n2 = NormalDistribution(6.0, 0.8)

        mult1 = MultinomialDistribution(3, 4, [0.23, 0.26, 0.26, 0.25], alphabet=self.DIAG)
        mult2 = MultinomialDistribution(3, 4, [0.7, 0.1, 0.1, 0.1], alphabet=self.DIAG)

        c1 = ProductDistribution([n1, mult1])
        c2 = ProductDistribution([n2, mult2])

        mpi = [0.4, 0.6]
        m = MixtureModel(2, mpi, [c1, c2])

        data = DataSet()
        l = [[2.3248630375235071, '8', '0', '1'],
            [4.950009051623474, '.', '.', '8'],
            [2.7476916816642873, '1', '8', '1'],
            [7.5525326592401942, '.', '1', '.'],
            [5.2438207409464539, '1', '.', '.'],
            [4.8362379747532103, '.', '.', '.'],
            [2.5870901172554128, '.', '0', '0'],
            [5.344556553752307, '1', '.', '.'],
            [6.9475338057098321, '.', '.', '.'],
            [2.5984482845693329, '0', '8', '1'],
            [2.6594447517982589, '1', '0', '8'],
            [2.515868200989412, '1', '8', '0'],
            [2.4332357214576641, '0', '.', '.'],
            [8.0636235373059382, '8', '.', '.'],
            [7.0105696968484548, '8', '0', '.'],
            [2.5925266286650888, '0', '0', '.'],
            [6.2954398350548653, '0', '.', '.'],
            [2.8361660731097387, '0', '8', '0'],
            [5.2581563085814205, '.', '0', '8'],
            [5.7629666039318046, '.', '.', '8'],
            [4.8952611828696151, '0', '8', '.'],
            [2.409261823396061, '.', '.', '.'],
            [2.1972954776899365, '1', '8', '8'],
            [2.7535734806909216, '8', '.', '0'],
            [4.9544722048862599, '0', '.', '.'],
            [1.9789354518110067, '8', '0', '.'],
            [5.7955733032762664, '.', '.', '0'],
            [4.7742014292462898, '.', '.', '1'],
            [6.5445727620632344, '.', '.', '1'],
            [5.3552893128649091, '.', '.', '1']]
        data.fromList(l)
        data.internalInit(m)
        m.modelInitialization(data)

        self.assertEqual(m.pi, [0.43333333, 0.56666667])
        self.assertEqual(m.components[0].distList[0], {"mu": 4.25527682308, "sigma": 1.87053050907})
        self.assertEqual(m.components[0].distList[1], {"M": 4, "p": 3, "phi": [0.46153846, 0.17948718, 0.17948718, 0.17948718]})
        self.assertEqual(m.components[1].distList[0], {"mu": 4.52356582315, "sigma": 1.78556613542})
        self.assertEqual(m.components[1].distList[1], {"M": 4, "p": 3, "phi": [0.47058824, 0.23529412, 0.17647059, 0.11764706]})


    def testmixturecomponentsproductmixtures(self):
        n1 = NormalDistribution(2.5, 0.5)
        n2 = NormalDistribution(6.0, 0.8)

        mult1 = MultinomialDistribution(3, 4, [0.23, 0.26, 0.26, 0.25], alphabet=self.DIAG)
        mult2 = MultinomialDistribution(3, 4, [0.7, 0.1, 0.1, 0.1], alphabet=self.DIAG)

        n11 = ProductDistribution([NormalDistribution(2.5, 0.5)])
        n12 = ProductDistribution([NormalDistribution(6.0, 0.8)])
        mix1 = MixtureModel(2, [0.5, 0.5], [n11, n12])

        n21 = ProductDistribution([NormalDistribution(6.5, 0.5)])
        n22 = ProductDistribution([NormalDistribution(-6.0, 0.8)])
        mix2 = MixtureModel(2, [0.5, 0.5], [n21, n22])

        c1 = ProductDistribution([n1, mult1, mix1])
        c2 = ProductDistribution([n2, mult2, mix2])

        mpi = [0.4, 0.6]
        m = MixtureModel(2, mpi, [c1, c2])

        data = DataSet()
        l = [[5.2438207409464539, '1', '.', '.', -7.1637620252467897],
            [1.8792023823294399, '0', '8', '0', 2.1649709182193724],
            [5.344556553752307, '1', '.', '.', -5.0524661942901679],
            [5.34149828634225, '.', '.', '1', 5.9412570838998988],
            [2.0369989328152429, '1', '0', '8', 2.9306800114588474],
            [5.4148098617138638, '.', '8', '.', 5.7757818020529363],
            [5.944842127762513, '.', '0', '0', -4.7397771273625633],
            [2.6529437131745466, '1', '1', '8', 2.2906032948034705],
            [5.7602211048038363, '.', '8', '0', 6.9119681369546875],
            [5.2365859568039825, '.', '.', '.', 6.7120250815230671],
            [5.7711330174727546, '8', '8', '.', 6.2034016653607091],
            [5.8529731334742205, '.', '0', '8', -6.7002382023544831],
            [1.9992217598127615, '.', '0', '1', 6.4123274427555117],
            [6.5924297433727288, '.', '.', '0', 7.3883704834824995],
            [2.1255558956808072, '.', '0', '8', 1.995801658217295],
            [6.0266962798519321, '.', '0', '1', 7.174956738390784],
            [2.6205322531250133, '1', '1', '.', 2.9951243644061796],
            [4.9101521351042514, '.', '0', '.', 5.5717237698322251],
            [5.7671344316967561, '.', '.', '1', 7.0061374072735925],
            [2.3325557513878477, '8', '8', '1', 5.8157542290203939],
            [5.2887935530774541, '.', '.', '8', 7.1808623582497288],
            [2.0282345625051019, '.', '0', '.', 2.4322368715994229],
            [6.5222974123847477, '1', '.', '.', 6.6023565117683969],
            [6.4321559627701648, '8', '.', '1', -5.8642599159587192],
            [5.6764147228437327, '.', '.', '8', -5.8751930220933684],
            [3.0830771701129849, '.', '1', '8', 2.9343697770375177],
            [7.4199761965051261, '8', '.', '.', 6.0821917385603372],
            [2.8062633792451974, '.', '0', '0', 7.4680564166282482],
            [2.1753149936115141, '1', '.', '8', 4.9517086476170844],
            [6.2938027130469552, '.', '.', '.', 6.3752625251152377],
            [6.1696694061749131, '.', '.', '.', -6.4089155934689295],
            [5.5472659807533136, '.', '.', '.', -6.2534054762325848],
            [6.3966448955380573, '.', '.', '.', -6.9295577720356736],
            [2.0609931982568579, '1', '1', '.', 2.8400285808693102],
            [6.1030972164686252, '.', '.', '1', 5.9293870980697712],
            [5.7951974552543692, '8', '.', '.', 6.04591521610259],
            [5.9656349475756842, '1', '8', '0', -6.0520395335625494],
            [3.3203062319904424, '0', '8', '1', 2.9768405076274922],
            [3.0419575890969055, '1', '8', '0', 2.4951861058624623],
            [7.013969201855673, '0', '1', '.', -5.2636476160305765]]
        data.fromList(l)
        data.internalInit(m)

        m.modelInitialization(data)

        self.assertEqual(m.pi, [0.625, 0.375])
        self.assertEqual(m.components[0].distList[0], {"mu": 4.6127407387, "sigma": 1.73433950115})
        self.assertEqual(m.components[0].distList[1], {"M": 4, "p": 3, "phi": [0.48, 0.16, 0.17333333, 0.18666667]})
        self.assertEqual(m.components[0].distList[2].pi, [0.45, 0.55])
        self.assertEqual(m.components[0].distList[2].components[0].distList[0], {"mu": 2.21433669974, "sigma": 5.225569022})
        self.assertEqual(m.components[0].distList[2].components[1].distList[0], {"mu": 1.88381651676, "sigma": 5.2347237821})
        self.assertEqual(m.components[1].distList[0], {"mu": 4.84509415887, "sigma": 1.74646782345})
        self.assertEqual(m.components[1].distList[1], {"M": 4, "p": 3, "phi": [0.46666667, 0.15555556, 0.17777778, 0.2]})
        self.assertEqual(m.components[1].distList[2].components[0].distList[0], {"mu": 0.943819933008, "sigma": 5.82352459352})
        self.assertEqual(m.components[1].distList[2].components[1].distList[0], {"mu": 3.50553914735, "sigma": 3.8447406861})


    def testbayesmixturecomponentsproductatomar(self):
        n1 = NormalDistribution(2.5, 0.5)
        n2 = NormalDistribution(6.0, 0.8)

        mult1 = MultinomialDistribution(3, 4, [0.23, 0.26, 0.26, 0.25], alphabet=self.DIAG)
        mult2 = MultinomialDistribution(3, 4, [0.7, 0.1, 0.1, 0.1], alphabet=self.DIAG)

        c1 = ProductDistribution([n1, mult1])
        c2 = ProductDistribution([n2, mult2])

        piPr = DirichletPrior(2, [1.0] * 2)
        cPr = [NormalGammaPrior(0.0, 0.1, 3.0, 1.0), DirichletPrior(4, [1.02] * 4)]
        mPrior = MixtureModelPrior(0.5, 0.5, piPr, cPr)

        mpi = [0.4, 0.6]
        m = BayesMixtureModel(2, mpi, [c1, c2], mPrior)

        data = DataSet()
        l = [[6.5445727620632344, '.', '.', '1'],
            [5.3552893128649091, '.', '.', '1'],
            [2.2183140202185716, '1', '.', '0'],
            [2.7391760600887061, '1', '8', '0'],
            [1.6538286754213469, '1', '0', '1'],
            [1.9645581511408992, '0', '8', '8'],
            [6.7101233232999657, '.', '8', '.'],
            [5.7515047118056293, '8', '.', '0'],
            [5.9845992826802386, '.', '.', '.'],
            [4.3453297022194874, '8', '.', '1'],
            [6.2282056476162397, '.', '.', '.'],
            [6.6872710683408076, '.', '.', '.'],
            [2.5652335300343463, '1', '8', '8'],
            [2.5034768168703572, '1', '8', '.'],
            [5.4253400776950356, '8', '8', '.'],
            [7.4199761965051261, '8', '.', '.'],
            [2.0821917385603368, '.', '0', '1'],
            [5.9872760795825508, '.', '.', '8'],
            [6.3478044273088532, '.', '0', '1'],
            [1.8448179047606779, '1', '8', '8'],
            [2.1557762510896699, '0', '0', '0'],
            [6.1696694061749131, '.', '.', '.'],
            [5.5910844065310705, '8', '.', '0'],
            [1.8505012017066282, '.', '.', '8'],
            [2.1666252614289743, '8', '8', '8'],
            [1.919026392477704, '0', '.', '8'],
            [2.3291175208299393, '1', '.', '0'],
            [6.3397464814438118, '.', '.', '.'],
            [7.0821575316682974, '.', '.', '1'],
            [7.2884403696771898, '.', '.', '.']]
        data.fromList(l)
        data.internalInit(m)

        m.modelInitialization(data)

        self.assertEqual(m.pi, [0.5, 0.5])
        self.assertEqual(m.components[0].distList[0], {"mu": 4.09877480669, "sigma": 1.72702515813})
        self.assertEqual(m.components[0].distList[1], {"M": 4, "p": 3, "phi": [0.35536823, 0.22227152, 0.2444543, 0.17790594]})
        self.assertEqual(m.components[1].distList[0], {"mu": 4.72579700206, "sigma": 1.83827093592})
        self.assertEqual(m.components[1].distList[1], {"M": 4, "p": 3, "phi": [0.5771961, 0.06699201, 0.22227152, 0.13354037]})


    def testbayesmixturecomponentsproductmixtures(self):
        n1 = NormalDistribution(2.5, 0.5)
        n2 = NormalDistribution(6.0, 0.8)

        mult1 = MultinomialDistribution(3, 4, [0.23, 0.26, 0.26, 0.25], alphabet=self.DIAG)
        mult2 = MultinomialDistribution(3, 4, [0.7, 0.1, 0.1, 0.1], alphabet=self.DIAG)

        n11 = ProductDistribution([NormalDistribution(2.5, 0.5)])
        n12 = ProductDistribution([NormalDistribution(6.0, 0.8)])
        mix1 = MixtureModel(2, [0.5, 0.5], [n11, n12])

        n21 = ProductDistribution([NormalDistribution(6.5, 0.5)])
        n22 = ProductDistribution([NormalDistribution(-6.0, 0.8)])
        mix2 = MixtureModel(2, [0.5, 0.5], [n21, n22])

        c1 = ProductDistribution([n1, mult1, mix1])
        c2 = ProductDistribution([n2, mult2, mix2])

        smixpiPr = DirichletPrior(2, [1.0] * 2)
        smixcompPr = [NormalGammaPrior(0.0, 0.1, 3, 1.0)]
        smixPr = MixtureModelPrior(0.8, 0.8, smixpiPr, smixcompPr)

        piPr = DirichletPrior(2, [1.0] * 2)
        cPr = [NormalGammaPrior(0.0, 0.1, 3.0, 1.0), DirichletPrior(4, [1.02] * 4), smixPr]
        mPrior = MixtureModelPrior(0.5, 0.5, piPr, cPr)

        mpi = [0.4, 0.6]
        m = BayesMixtureModel(2, mpi, [c1, c2], mPrior)

        data = DataSet()
        l = [[6.8098198516377479, '.', '.', '.', -5.2898766767000343],
            [2.9518024758749997, '8', '0', '0', 5.7415446659617899],
            [5.9845992826802386, '.', '.', '.', 5.4658310638871797],
            [6.4999354021546782, '.', '.', '.', -5.5678440372298352],
            [6.6872710683408076, '.', '.', '.', 6.5652335300343463],
            [6.1960339094588992, '.', '.', '8', -5.067076527819224],
            [3.2854612890487811, '.', '1', '0', 3.3874851228157037],
            [5.7051517369462657, '.', '.', '.', 6.8062633792451974],
            [2.2421253834160457, '1', '1', '8', 2.1753149936115141],
            [4.7688879222961011, '1', '.', '.', -5.7061972869530448],
            [2.1557762510896699, '0', '0', '0', 6.1696694061749131],
            [2.6989379470216446, '8', '0', '1', 5.5472659807533136],
            [6.3862222115374525, '.', '.', '.', -5.6973132761108705],
            [2.6336488789410022, '1', '0', '.', 5.2975891172109728],
            [5.1918578345128363, '.', '.', '.', 6.1006384104946614],
            [5.8746601885794805, '.', '1', '.', -4.5889879007373686],
            [5.6646552888245107, '.', '.', '.', 6.914114655862857],
            [5.352112279485266, '0', '0', '.', -5.3044760707577598],
            [3.1861561083607257, '1', '0', '1', 6.8671321425550484],
            [6.3282486244433231, '.', '.', '.', 7.1337307511597956],
            [7.6503080546750954, '.', '.', '0', -6.4191799172344055],
            [2.1352293729831207, '1', '8', '1', 3.381682310029694],
            [1.5662362754054673, '8', '8', '8', 6.4239353061587572],
            [5.5319979507882211, '.', '.', '0', -7.0744057157000197],
            [2.0604053378541511, '1', '0', '.', 2.8256296756867658],
            [6.3548844417129313, '.', '.', '.', -6.2440326017533163],
            [5.2973602943981968, '.', '.', '1', -6.1795971712981919],
            [4.6208552313082008, '1', '.', '8', -4.5458375637415731],
            [7.1544530669784043, '0', '.', '8', 6.6146297939744763],
            [1.6239993540756092, '8', '8', '0', 2.1987204598700427]]
        data.fromList(l)
        data.internalInit(m)

        m.modelInitialization(data)
        self.assertEqual(m.pi, [0.66666667, 0.33333333])
        self.assertEqual(m.components[0].distList[0], {"mu": 4.51427440311, "sigma": 1.75358833435})
        self.assertEqual(m.components[0].distList[1], {"M": 4, "p": 3, "phi": [0.49966711, 0.16677763, 0.16677763, 0.16677763]})
        self.assertEqual(m.components[0].distList[2].pi, [0.36666667, 0.63333333])
        self.assertEqual(m.components[0].distList[2].components[0].distList[0], {"mu": 1.18018741932, "sigma": 5.30822827248})
        self.assertEqual(m.components[0].distList[2].components[1].distList[0], {"mu": 0.786817074047, "sigma": 5.65990534226})
        self.assertEqual(m.components[1].distList[0], {"mu": 4.93684928834, "sigma": 1.35601386675})
        self.assertEqual(m.components[1].distList[1], {"M": 4, "p": 3, "phi": [0.59906915, 0.20013298, 0.06715426, 0.13364362]})
        self.assertEqual(m.components[1].distList[2].pi, [0.46666667, 0.53333333])
        self.assertEqual(m.components[1].distList[2].components[0].distList[0], {"mu": 2.23150044795, "sigma": 5.40884624383})
        self.assertEqual(m.components[1].distList[2].components[1].distList[0], {"mu": -0.206838765738, "sigma": 5.39509340632})


class ModelSelectionTests(BaseTest):
    def setUp(self):
        random.seed(3586662)

        self.DIAG = Alphabet(['.', '0', '8', '1'])
        n1 = NormalDistribution(2.5, 0.5)
        n2 = NormalDistribution(6.0, 0.8)
        n3 = NormalDistribution(2.0, 1.8)

        mult1 = MultinomialDistribution(3, 4, [0.23, 0.26, 0.26, 0.25], alphabet=self.DIAG)
        mult2 = MultinomialDistribution(3, 4, [0.7, 0.1, 0.1, 0.1], alphabet=self.DIAG)
        mult3 = MultinomialDistribution(3, 4, [0.3, 0.3, 0.2, 0.2], alphabet=self.DIAG)

        c1 = ProductDistribution([n1, mult1])
        c2 = ProductDistribution([n2, mult2])
        c3 = ProductDistribution([n3, mult3])

        self.m1 = MixtureModel(1, [1.0], [c1])
        self.m2 = MixtureModel(2, [0.4, 0.6], [c1, c2])
        self.m3 = MixtureModel(3, [0.2, 0.3, 0.5], [c1, c2, c3])

        self.data = self.m2.sampleDataSet(100)

    def testmodelselection(self):
        mlist = [self.m1, self.m2, self.m3]
        NEC, BIC, AIC = modelSelection(self.data, mlist, silent=1)

        tNEC = [1.0, 0.00019979785365307126, 0.024609799053949243]
        tBIC = [4172.0691607116069, 1057.3561873187775, 1156.2216432806276]
        tAIC = [4205.095011641547, 1028.6993152729087, 1111.9337501188302]
        for i in range(3):
            self.assertAlmostEqual(NEC[i], tNEC[i], 16)
            self.assertAlmostEqual(BIC[i], tBIC[i], 16)
            self.assertAlmostEqual(AIC[i], tAIC[i], 16)

        self.assertEqual(mlist[numpy.argmin(NEC)].G, 2)
        self.assertEqual(mlist[numpy.argmin(BIC)].G, 2)
        self.assertEqual(mlist[numpy.argmin(AIC)].G, 2)

## Run ALL tests (comment out to deactivate)
if __name__ == '__main__':
    unittest.main()



# Individual test suites for each of the different classes
suiteDataSetTests = unittest.makeSuite(DataSetTests, 'test')
suiteMixtureModelTests = unittest.makeSuite(MixtureModelTests, 'test')
suiteBayesMixtureModelTests = unittest.makeSuite(BayesMixtureModelTests, 'test')
suiteMixtureModelPriorTests = unittest.makeSuite(MixtureModelPriorTests, 'test')
suitePartialLearningTests = unittest.makeSuite(MixtureModelPartialLearningTests, 'test')
suiteModelInitTests = unittest.makeSuite(ModelInitTests, 'test')
suiteformatDataTests = unittest.makeSuite(FormatDataTests, 'test')
suiteDirichletPriorTests = unittest.makeSuite(DirichletPriorTests, 'test')
suiteDirichletMixturePriorTests = unittest.makeSuite(DirichletMixturePriorTests, 'test')
suiteProductDistributionTests = unittest.makeSuite(ProductDistributionTests, 'test')
suiteFormatDataTests = unittest.makeSuite(FormatDataTests, 'test')


# Call to individual test suites, uncomment to activate as needed.
#runner = unittest.TextTestRunner(verbosity=2)
#runner.run(suiteFormatDataTests)
#runner.run(suiteDataSetTests)
#runner.run(suiteMixtureModelTests)
#runner.run(suiteBayesMixtureModelTests)
#runner.run(suiteMixtureModelPriorTests)
#runner.run(suitePartialLearningTests)
#runner.run(suiteModelInitTests)
#runner.run(suiteDirichletPriorTests)
#runner.run(suiteDirichletMixturePriorTests)
#runner.run(suiteProductDistributionTests)
