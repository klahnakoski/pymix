# From http://sourceforge.net/projects/ghmm/files/latest/download?source=typ_redirect
#

################################################################################
#
#       This file is part of the General Hidden Markov Model Library,
#       GHMM version __VERSION__, see http://ghmm.org
#
#       file:    ghmm.py
#       authors: Benjamin Georgi, Wasinee Rungsarityotin, Alexander Schliep,
#                Janne Grunau
#
#       Copyright (C) 1998-2004 Alexander Schliep
#       Copyright (C) 1998-2001 ZAIK/ZPR, Universitaet zu Koeln
#       Copyright (C) 2002-2004 Max-Planck-Institut fuer Molekulare Genetik,
#                               Berlin
#
#       Contact: schliep@ghmm.org
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

"""@mainpage GHMM - an open source library for Hidden Markov Models (HMM)

HMMs are stochastic models which encode a probability density over
sequences of symbols. These symbols can be discrete letters (A,C,G and
T for DNA; 1,2,3,4,5,6 for dice), real numbers (weather measurement
over time: temperature) or vectors of either or the combination
thereof (weather again: temperature, pressure, percipitation).

@note
We will always talk about emissions, emission sequence and so
forth when we refer to the sequence of symbols. Another name
for the same object is observation resp. observation sequence.

A simple model with a fair and one unfair coin can be created as follows

>> fair = [0.5, 0.5]
>> loaded = [0.9, 0.1]
>> A = [[0.9, 0.1], [0.3, 0.7]]
>> pi = [0.9, 0.1]
>> B = [fair, loaded]
>> sigma = ghmm.IntegerRange(0,2)
>> m = ghmm.HMMFromMatrices(sigma, ghmm.DiscreteDistribution(sigma), A, B, pi)

The objects one has to deal with in HMM modelling are the following

-# The domain the emissions come from: the EmissionDomain. Domain
   is to be understood mathematically and to encompass both discrete,
   finite alphabets and fields such as the real numbers or intervals
   of the reals.\n
   For technical reasons there can be two representations of an
   emission symbol: an external and an internal. The external
   representation is the view of the application using ghmm.py. The
   internal one is what is used in both ghmm.py and the ghmm
   C-library. Representations can coincide, but this is not
   guaranteed. Discrete alphabets of size k are represented as
   [0,1,2,...,k-1] internally.  It is the domain objects job to
   provide a mapping between representations in both directions.
   @note
   Do not make assumptions about the internal
   representations. It might change.

-# Every domain has to afford a distribution, which is usually
   parameterized. A distribution associated with a domain
   should allow us to compute \f$Prob[x| distribution parameters]\f$
   efficiently.\n
   The distribution defines the \b type of distribution which
   we will use to model emissions in <b>every state</b> of the HMM.
   The \b type of distribution will be identical for all states,
   their \b parameterizations will differ from state to state.

-# We will consider a Sequence of emissions from the same emission
   domain and very often sets of such sequences: SequenceSet

-# The HMM: The HMM consists of two major components: A Markov chain
   over states (implemented as a weighted directed graph with
   adjacency and inverse-adjacency lists) and the emission
   distributions per-state. For reasons of efficiency the HMM itself
   is *static*, as far as the topology of the underlying Markov chain
   (and obviously the EmissionDomain) are concerned. You cannot add or
   delete transitions in an HMM.\n
   Transition probabilities and the parameters of the per-state
   emission distributions can be easily modified. Particularly,
   Baum-Welch reestimation is supported.  While a transition cannot be
   deleted from the graph, you can set the transition probability to
   zero, which has the same effect from the theoretical point of
   view. However, the corresponding edge in the graph is still
   traversed in the computation.\n
   States in HMMs are referred to by their integer index. State sequences
   are simply list of integers.\n
   If you want to store application specific data for each state you
   have to do it yourself.\n
   Subclasses of HMM implement specific types of HMM. The type depends
   on the EmissionDomain, the Distribution used, the specific
   extensions to the 'standard' HMMs and so forth
"""
import re
import StringIO
import math
import os
from string import join
from pyLibrary.maths import Math
from pymix.distributions.multinormal import MultiNormalDistribution
from pymix.distributions.normal import NormalDistribution
from pymix.distributions.normal_left import NormalLeft
from pymix.distributions.normal_right import NormalRight
from pymix.distributions.uniform import UniformDistribution

from pymix.util.ghmm import types
from pymix.util.ghmm import wrapper
from pymix.util.ghmm.cmodel import ghmm_cmodel
from pymix.util.ghmm.dbackground import dbackground
from pymix.util.ghmm.dmodel import ghmm_dmodel
from pymix.util.ghmm.dstate import ghmm_dstate
from pymix.util.ghmm.gradescent import ghmm_dmodel_label_gradient_descent
from pymix.util.ghmm.kbest import ghmm_dmodel_label_kbest
from pymix.util.ghmm.sreestimate import ghmm_cmodel_baum_welch
from pymix.util.ghmm.sviterbi import ghmm_cmodel_viterbi
from pymix.util.ghmm.types import kSilentStates, kHigherOrderEmissions, kTiedEmissions, kBackgroundDistributions, kLabeledStates, kNotSpecified, kMultivariate, kContinuousHMM, kDiscreteHMM, \
    kTransitionClasses, kPairHMM
from pymix.util.ghmm.viterbi import ghmm_dmodel_viterbi
from pymix.vendor.ghmm import ghmmhelper
import modhmmer
from pymix.vendor.ghmm.distribution import MultivariateGaussianDistribution, ContinuousMixtureDistribution, DiscreteDistribution, GaussianMixtureDistribution, GaussianDistribution
from pymix.vendor.ghmm.emission_domain import LabelDomain, Float, Alphabet, IntegerRange, AminoAcids, DNA
from pymix.vendor.ghmm.sequence_set import SequenceSet, EmissionSequence
from pymix.util.logs import Log


class HMMFactory(object):
    """
    A HMMFactory is the base class of HMM factories.
    A HMMFactory has just a constructor and a call method
    """


GHMM_FILETYPE_SMO = 'smo' #obsolete
GHMM_FILETYPE_XML = 'xml'
GHMM_FILETYPE_HMMER = 'hmm'


class HMMOpenFactory(HMMFactory):
    """ Opens a HMM from a file.

    Currently four formats are supported:
    HMMer, our smo file format, and two xml formats.
    @note the support for smo files and the old xml format will phase out
    """

    def __init__(self, defaultFileType=None):
        self.defaultFileType = defaultFileType

    def guessFileType(self, filename):
        """ guesses the file format from the filename """
        if filename.endswith('.' + GHMM_FILETYPE_XML):
            return GHMM_FILETYPE_XML
        elif filename.endswith('.' + GHMM_FILETYPE_SMO):#obsolete
            return GHMM_FILETYPE_SMO
        elif filename.endswith('.' + GHMM_FILETYPE_HMMER):#obsolete
            return GHMM_FILETYPE_HMMER
        else:
            return None

    def __call__(self, fileName, modelIndex=None, filetype=None):

        if not isinstance(fileName, StringIO.StringIO):
            if not os.path.exists(fileName):
                Log.error('File ' + str(fileName) + ' not found.')

        if not filetype:
            if self.defaultFileType:
                Log.warning("HMMOpenHMMER, HMMOpenSMO and HMMOpenXML are deprecated. "
                            + "Use HMMOpen and the filetype parameter if needed.")
                filetype = self.defaultFileType
            else:
                filetype = self.guessFileType(fileName)
            if not filetype:
                Log.error("Could not guess the type of file " + str(fileName)
                          + " and no filetype specified")

        # XML file: both new and old format
        if filetype == GHMM_FILETYPE_XML:
            # try to validate against ghmm.dtd
            if wrapper.ghmm_xmlfile_validate(fileName):
                return self.openNewXML(fileName, modelIndex)
            else:
                return self.openOldXML(fileName)
        elif filetype == GHMM_FILETYPE_HMMER:
            return self.openHMMER(fileName)
        else:
            Log.error("Invalid file type " + str(filetype))


    def openNewXML(self, fileName, modelIndex):
        """ Open one ore more HMM in the new xml format """
        # opens and parses the file
        file = wrapper.ghmm_xmlfile_parse(fileName)
        if file == None:
            Log.note("XML has file format problems!")
            Log.error("file is not in GHMM xml format")

        nrModels = file.noModels
        modelType = file.modelType

        # we have a continuous HMM, prepare for hmm creation
        if modelType & kContinuousHMM:
            emission_domain = Float()
            if modelType & kMultivariate:
                distribution = MultivariateGaussianDistribution
                hmmClass = MultivariateGaussianMixtureHMM
            else:
                distribution = ContinuousMixtureDistribution
                hmmClass = ContinuousMixtureHMM
            getModel = file.get_cmodel

        # we have a discrete HMM, prepare for hmm creation
        elif ((modelType & kDiscreteHMM) and not (modelType & kTransitionClasses) and not (modelType & kPairHMM)):
            emission_domain = 'd'
            distribution = DiscreteDistribution
            getModel = file.get_dmodel
            if modelType & kLabeledStates:
                hmmClass = StateLabelHMM
            else:
                hmmClass = DiscreteEmissionHMM

        # currently not supported
        else:
            raise Log.error("Non-supported model type")


        # read all models to list at first
        result = []
        for i in range(nrModels):
            cmodel = getModel(i)
            if emission_domain is 'd':
                emission_domain = Alphabet(cmodel.alphabet)
            if modelType & kLabeledStates:
                labelDomain = LabelDomain(cmodel.label_alphabet)
                m = hmmClass(emission_domain, distribution(emission_domain), labelDomain, cmodel)
            else:
                m = hmmClass(emission_domain, distribution(emission_domain), cmodel)

            result.append(m)

        # for a single
        if modelIndex != None:
            if modelIndex < nrModels:
                result = result[modelIndex]
            else:
                Log.error("the file %s has only %s models" % fileName, str(nrModels))
        elif nrModels == 1:
            result = result[0]

        return result

        #obsolete

    def openOldXML(self, fileName):
        # from ghmm_gato import xmlutil

        hmm_dom = None #xmlutil.HMM(fileName)
        emission_domain = hmm_dom.AlphabetType()

        if emission_domain == int:
            [alphabets, A, B, pi, state_orders] = hmm_dom.buildMatrices()

            emission_domain = Alphabet(alphabets)
            distribution = DiscreteDistribution(emission_domain)
            # build adjacency list

            # check for background distributions
            (background_dist, orders, code2name) = hmm_dom.getBackgroundDist()
            # (background_dist, orders) = hmm_dom.getBackgroundDist()
            bg_list = []
            # if background distribution exists, set background distribution here
            if background_dist != {}:
                # transformation to list for input into BackgroundDistribution,
                # ensure the rigth order
                for i in range(len(code2name.keys()) - 1):
                    bg_list.append(background_dist[code2name[i]])

                bg = BackgroundDistribution(emission_domain, bg_list)

            # check for state labels
            (label_list, labels) = hmm_dom.getLabels()
            if labels == ['None']:
                labeldom = None
                label_list = None
            else:
                labeldom = LabelDomain(labels)

            m = HMMFromMatrices(emission_domain, distribution, A, B, pi, None, labeldom, label_list)

            # old xml is discrete, set appropiate flag
            m.cmodel.model_type |= kDiscreteHMM

            if background_dist != {}:
                ids = [-1] * m.N
                for s in hmm_dom.state.values():
                    ids[s.index - 1] = s.background # s.index ranges from [1, m.N]

                m.setBackground(bg, ids)
                Log.note("model_type %x" % m.cmodel.model_type)
                Log.note("background_id" + str(wrapper.int_array2list(m.cmodel.background_id, m.N)))
            else:
                m.cmodel.bp = None
                m.cmodel.background_id = None

            # check for tied states
            tied = hmm_dom.getTiedStates()
            if len(tied) > 0:
                m.setFlags(kTiedEmissions)
                m.cmodel.tied_to = wrapper.list2int_array(tied)

            durations = hmm_dom.getStateDurations()
            if len(durations) == m.N:
                Log.note("durations: " + str(durations))
                m.extendDurations(durations)

            return m


    def openSingleHMMER(self, fileName):
        # HMMER format models
        h = modhmmer.hmmer(fileName)

        if h.m == 4:  # DNA model
            emission_domain = DNA
        elif h.m == 20:   # Peptide model
            emission_domain = AminoAcids
        else:   # some other model
            emission_domain = IntegerRange(0, h.m)
        distribution = DiscreteDistribution(emission_domain)

        # XXX TODO: Probably slow for large matrices (Rewrite for 0.9)
        [A, B, pi, modelName] = h.getGHMMmatrices()
        return HMMFromMatrices(emission_domain, distribution, A, B, pi, hmmName=modelName)


    def openHMMER(self, fileName):
        """
        Reads a file containing multiple HMMs in HMMER format, returns list of
        HMM objects or a single HMM object.
        """
        if not os.path.exists(fileName):
            Log.error('File ' + str(fileName) + ' not found.')

        modelList = []
        string = ""
        f = open(fileName, "r")

        res = re.compile("^//")
        stat = re.compile("^ACC\s+(\w+)")
        for line in f.readlines():
            string = string + line
            m = stat.match(line)
            if m:
                name = m.group(1)
                Log.info("Reading model " + str(name) + ".")

            match = res.match(line)
            if match:
                fileLike = StringIO.StringIO(string)
                modelList.append(self.openSingleHMMER(fileLike))
                string = ""
                match = None

        if len(modelList) == 1:
            return modelList[0]
        return modelList


    def determineHMMClass(self, fileName):
        #
        # smo files. Obsolete
        #
        file = open(fileName, 'r')

        hmmRe = re.compile("^HMM\s*=")
        shmmRe = re.compile("^SHMM\s*=")
        mvalueRe = re.compile("M\s*=\s*([0-9]+)")
        densityvalueRe = re.compile("density\s*=\s*([0-9]+)")
        cosvalueRe = re.compile("cos\s*=\s*([0-9]+)")
        emission_domain = None

        while 1:
            l = file.readline()
            if not l:
                break
            l = l.strip()
            if len(l) > 0 and l[0] != '#': # Not a comment line
                hmm = hmmRe.search(l)
                shmm = shmmRe.search(l)
                mvalue = mvalueRe.search(l)
                densityvalue = densityvalueRe.search(l)
                cosvalue = cosvalueRe.search(l)

                if hmm != None:
                    if emission_domain != None and emission_domain != 'int':
                        Log.error("HMMOpenFactory:determineHMMClass: both HMM and SHMM? " + str(emission_domain))
                    else:
                        emission_domain = 'int'

                if shmm != None:
                    if emission_domain != None and emission_domain != 'double':
                        Log.error("HMMOpenFactory:determineHMMClass: both HMM and SHMM? " + str(emission_domain))
                    else:
                        emission_domain = 'double'

                if mvalue != None:
                    M = int(mvalue.group(1))

                if densityvalue != None:
                    density = int(densityvalue.group(1))

                if cosvalue != None:
                    cos = int(cosvalue.group(1))

        file.close()
        if emission_domain == 'int':
            # only integer alphabet
            emission_domain = IntegerRange(0, M)
            distribution = DiscreteDistribution
            hmm_class = DiscreteEmissionHMM
            return (hmm_class, emission_domain, distribution)

        elif emission_domain == 'double':
            # M        number of mixture components
            # density  component type
            # cos      number of state transition classes
            if M == 1 and density == 0:
                emission_domain = Float()
                distribution = GaussianDistribution
                hmm_class = GaussianEmissionHMM
                return (hmm_class, emission_domain, distribution)

            elif M > 1 and density == 0:
                emission_domain = Float()
                distribution = GaussianMixtureDistribution
                hmm_class = GaussianMixtureHMM
                return (hmm_class, emission_domain, distribution)

            else:
                Log.error("Model type can not be determined.")

        return (None, None, None)


# use only HMMOpen and specify the filetype if it can't guessed from the extension
HMMOpen = HMMOpenFactory()


class HMMFromMatricesFactory(HMMFactory):
    """ @todo Document matrix formats """

    # XXX TODO: this should use the editing context
    def __call__(self, emissionDomain, distribution, A, B, pi, hmmName=None, labelDomain=None, labelList=None, densities=None):
        if isinstance(emissionDomain, Alphabet):

            if not emissionDomain == distribution.alphabet:
                Log.error("emissionDomain and distribution must be compatible")

            # checking matrix dimensions and argument validation, only some obvious errors are checked
            if not len(A) == len(A[0]):
                Log.error("A is not quadratic.")
            if not len(pi) == len(A):
                Log.error("Length of pi does not match length of A.")
            if not len(A) == len(B):
                Log.error("Different number of entries in A and B.")

            if (labelDomain is None and labelList is not None) or (labelList is None and labelList is not None):
                Log.error("Specify either both labelDomain and labelInput or neither.")

            if isinstance(distribution, DiscreteDistribution):
                # HMM has discrete emissions over finite alphabet: DiscreteEmissionHMM
                cmodel = ghmm_dmodel(len(A), len(emissionDomain))
                cmodel.model_type |= kDiscreteHMM

                # assign model identifier (if specified)
                if hmmName != None:
                    cmodel.name = hmmName
                else:
                    cmodel.name = ''

                states = [None] * cmodel.N
                silent_states = []
                tmpOrder = []

                #initialize states
                for i in range(cmodel.N):
                    state = ghmm_dstate()
                    states[i] = state
                    # compute state order
                    if cmodel.M > 1:
                        order = Math.log(len(B[i]), cmodel.M) - 1
                    else:
                        order = len(B[i]) - 1

                    Log.note("order in state %d = %d", i, order)
                    # check or valid number of emission parameters
                    order = int(order)
                    if cmodel.M ** (order + 1) == len(B[i]):
                        tmpOrder.append(order)
                    else:
                        Log.error("The number of " + str(len(B[i])) + " emission parameters for state " + str(i) + " is invalid. State order can not be determined.")

                    state.b = list(B[i])
                    state.pi = pi[i]

                    if sum(B[i]) == 0.0:
                        silent_states.append(1)
                    else:
                        silent_states.append(0)

                    #set out probabilities
                    _, _, state.out_a = ghmmhelper.extract_out(A[i])

                    #set "in" probabilities
                    A_col_i = map(lambda x: x[i], A)
                    # Numarray use A[,:i]
                    _, _, state.in_a = ghmmhelper.extract_out(A_col_i)
                    #fix probabilities in reestimation, else 0
                    state.fix = 0

                cmodel.s = states
                if sum(silent_states) > 0:
                    cmodel.model_type |= kSilentStates
                    cmodel.silent = list(silent_states)

                cmodel.maxorder = max(tmpOrder)
                if cmodel.maxorder > 0:
                    Log.note("Set kHigherOrderEmissions.")
                    cmodel.model_type |= kHigherOrderEmissions
                    cmodel.order = wrapper.list2int_array(tmpOrder)

                # check for state labels
                if labelDomain is not None and labelList is not None:
                    if not isinstance(labelDomain, LabelDomain):
                        Log.error("LabelDomain object required.")

                    cmodel.model_type |= kLabeledStates
                    m = StateLabelHMM(emissionDomain, distribution, labelDomain, cmodel)
                    m.setLabels(labelList)
                    return m
                else:
                    return DiscreteEmissionHMM(emissionDomain, distribution, cmodel)
            else:
                Log.error(type(distribution), "Not a valid distribution for Alphabet")

        elif isinstance(emissionDomain, Float):
            # determining number of transition classes
            cos = ghmmhelper.classNumber(A)
            if cos == 1:
                A = [A]

            cmodel = ghmm_cmodel(len(A[0]), kContinuousHMM)
            Log.note("cmodel.cos = " + str(cmodel.cos))

            self.constructSwitchingTransitions(cmodel, pi, A)

            if isinstance(distribution, GaussianDistribution):
                #initialize emissions
                for i in range(cmodel.N):
                    state = cmodel.s[i]
                    state.M = 1

                    # set up emission(s), density type is normal
                    (mu, sigma) = B[i]
                    emissions = [NormalDistribution(mu, sigma)]

                    # append emission to state
                    state.e = emissions
                    state.c = wrapper.list2double_array([1.0])

                return GaussianEmissionHMM(emissionDomain, distribution, cmodel)

            elif isinstance(distribution, GaussianMixtureDistribution):
                # Interpretation of B matrix for the mixture case
                # (Example with three states and two components each):
                #  B = [
                #      [ ["mu11","mu12"],["sig11","sig12"],["w11","w12"]   ],
                #      [  ["mu21","mu22"],["sig21","sig22"],["w21","w22"]  ],
                #      [  ["mu31","mu32"],["sig31","sig32"],["w31","w32"]  ],
                #      ]

                Log.note("*** mixture model")

                cmodel.M = len(B[0][0])

                #initialize states
                for i in range(cmodel.N):
                    state = cmodel.s[i]
                    state.M = len(B[0][0])

                    # allocate arrays of emmission parameters
                    mu_list = B[i][0]
                    sigma_list = B[i][1]
                    weight_list = B[i][2]

                    state.c = weight_list

                    # set up emission(s), density type is normal
                    emissions = [None] * state.M

                    for n in range(state.M):
                        emissions[n] = NormalDistribution(mu_list[n], sigma_list[n])

                    # append emissions to state
                    state.e = emissions

                return GaussianMixtureHMM(emissionDomain, distribution, cmodel)

            elif isinstance(distribution, ContinuousMixtureDistribution):
                # Interpretation of B matrix for the mixture case
                # (Example with three states and two components each):
                #  B = [
                #      [["mu11","mu12"], ["sig11","sig12"], ["a11","a12"], ["w11","w12"]],
                #      [["mu21","mu22"], ["sig21","sig22"], ["a21","a22"], ["w21","w22"]],
                #      [["mu31","mu32"], ["sig31","sig32"], ["a31","a32"], ["w31","w32"]],
                #      ]
                #
                # wrapper.uniform: mu = min, sig = max
                # wrapper.normal_right or wrapper.normal_left: a = cutoff

                Log.note("*** general mixture model")

                cmodel.M = len(B[0][0])

                #initialize states
                for i in range(cmodel.N):
                    state = cmodel.s[i]
                    state.M = len(B[i][0])

                    # set up emission(s), density type is normal
                    emissions = [None] * state.M
                    weight_list = B[i][3]

                    for n, density in enumerate(densities[i]):
                        parameters = (B[i][0][n], B[i][1][n], B[i][2][n])
                        emissions[n] = density(*parameters)

                    # append emissions to state
                    state.e = emissions
                    state.c = wrapper.list2double_array(weight_list)

                return ContinuousMixtureHMM(emissionDomain, distribution, cmodel)

            elif isinstance(distribution, MultivariateGaussianDistribution):
                Log.note("*** multivariate gaussian distribution model")

                # this is being extended to also support mixtures of multivariate gaussians
                # Interpretation of B matrix for the multivariate gaussian case
                # (Example with three states and two mixture components with two dimensions):
                #  B = [
                #       [["mu111","mu112"],["sig1111","sig1112","sig1121","sig1122"],
                #        ["mu121","mu122"],["sig1211","sig1212","sig1221","sig1222"],
                #        ["w11","w12"] ],
                #       [["mu211","mu212"],["sig2111","sig2112","sig2121","sig2122"],
                #        ["mu221","mu222"],["sig2211","sig2212","sig2221","sig2222"],
                #        ["w21","w22"] ],
                #       [["mu311","mu312"],["sig3111","sig3112","sig3121","sig3122"],
                #        ["mu321","mu322"],["sig3211","sig3212","sig3221","sig3222"],
                #        ["w31","w32"] ],
                #      ]
                #
                # ["mu311","mu312"] is the mean vector of the two dimensional
                # gaussian in state 3, mixture component 1
                # ["sig1211","sig1212","sig1221","sig1222"] is the covariance
                # matrix of the two dimensional gaussian in state 1, mixture component 2
                # ["w21","w22"] are the weights of the mixture components
                # in state 2
                # For states with only one mixture component, a implicit weight
                # of 1.0 is assumed

                cmodel.model_type |= kMultivariate
                cmodel.dim = len(B[0][0]) # all states must have same dimension

                #initialize states
                for i in range(cmodel.N):
                    # set up state parameterss
                    state = cmodel.s[i]
                    state.M = len(B[i]) / 2
                    if state.M > cmodel.M:
                        cmodel.M = state.M

                    # multiple mixture components
                    if state.M > 1:
                        state.c = wrapper.list2double_array(B[i][state.M * 2]) # Mixture weights.
                    else:
                        state.c = wrapper.list2double_array([1.0])

                    # set up emission(s), density type is normal
                    emissions = [None] * state.M # M emission components in this state

                    for em in range(state.M):
                        emissions[em] = MultiNormalDistribution(B[i][em * 2], B[i][em * 2 + 1])

                    # append emissions to state
                    state.e = emissions

                return MultivariateGaussianMixtureHMM(emissionDomain, distribution, cmodel)

            else:
                Log.error(type(distribution),
                    "Cannot construct model for this domain/distribution combination")
        else:
            Log.error("Unknown emission doamin" + str(emissionDomain))

    def constructSwitchingTransitions(self, cmodel, pi, A):
        """ @internal function: creates switching transitions """

        #initialize states
        for i in range(cmodel.N):

            state = cmodel.s[i]
            state.pi = pi[i]

            #set out probabilities
            trans = ghmmhelper.extract_out_cos(A, cmodel.cos, i)
            state.out_a = trans[2]

            #set "in" probabilities
            trans = ghmmhelper.extract_in_cos(A, cmodel.cos, i)
            state.in_a = trans[2]


HMMFromMatrices = HMMFromMatricesFactory()

#-------------------------------------------------------------------------------
#- Background distribution

class BackgroundDistribution(object):
    """ Background distributions object

        holds discrete distributions used as background while training
        discrete HMMs to avoid overfitting.
        Input is a discrete EmissionDomain and a list of list. Each list is
        a distinct distribution. The distributions can be of higher order.
        The length of a single distribution is a power of len(EmissionDomain)
    """

    def __init__(self, emissionDomain, bgInput):

        if type(bgInput) == list:
            self.emissionDomain = emissionDomain
            distNum = len(bgInput)

            order = wrapper.int_array_alloc(distNum)
            b = wrapper.double_matrix_alloc_row(distNum)
            for i in range(distNum):
                if len(emissionDomain) > 1:
                    o = Math.log(len(bgInput[i]), len(emissionDomain)) - 1
                else:
                    o = len(bgInput[i]) - 1

                assert (o % 1) == 0, "Invalid order of distribution " + str(i) + ": " + str(o)

                wrapper.int_array_setitem(order, i, int(o))
                # dynamic allocation, rows have different lenghts
                b_i = bgInput[i]
                wrapper.double_matrix_set_col(b, i, b_i)

            self.cbackground = dbackground(distNum, len(emissionDomain), order, b)
            self.name2id = dict()
        elif isinstance(bgInput, dbackground):
            self.cbackground = bgInput
            self.emissionDomain = emissionDomain
            self.name2id = dict()
            self.updateName2id()
        else:
            Log.error("Input type " + str(type(bgInput)) + " not recognized.")

    def __del__(self):
        Log.note("__del__ BackgroundDistribution " + str(self.cbackground))
        del self.cbackground
        self.cbackground = None

    def __str__(self):
        outstr = 'BackgroundDistribution (N= ' + str(self.cbackground.n) + '):\n'
        outstr += str(self.emissionDomain) + "\n"
        d = ghmmhelper.double_matrix2list(self.cbackground.b, self.cbackground.n, len(self.emissionDomain))
        outstr += "Distributions:\n"
        f = lambda x: "%.2f" % (x,)  # float rounding function

        for i in range(self.cbackground.n):
            if self.cbackground.getName(i) is not None:
                outstr += '  ' + str(i + 1) + ", name = " + self.cbackground.getName(i);
            else:
                outstr += '  ' + str(i + 1)
            outstr += " :(order= " + str(self.cbackground.getOrder(i)) + "): "
            outstr += " " + join(map(f, d[i]), ', ') + "\n"
        return outstr


    def verboseStr(self):
        outstr = "BackgroundDistribution instance:\n"
        outstr += "Number of distributions: " + str(self.cbackground.n) + "\n\n"
        outstr += str(self.emissionDomain) + "\n"
        d = ghmmhelper.double_matrix2list(self.cbackground.b, self.cbackground.n, len(self.emissionDomain))
        outstr += "Distributions:\n"
        for i in range(self.cbackground.n):
            outstr += "  Order: " + str(self.cbackground.getOrder(i)) + "\n"
            outstr += "  " + str(i + 1) + ": " + str(d[i]) + "\n"
        return outstr

    def getCopy(self):
        return BackgroundDistribution(self.emissionDomain, self.cbackground.copy())

    def toLists(self):
        dim = self.cbackground.m
        distNum = self.cbackground.n
        orders = wrapper.int_array2list(self.cbackground.order, distNum)
        B = []
        for i in xrange(distNum):
            order = orders[i]
            size = int(pow(self.m, (order + 1)))
            b = [0.0] * size
            for j in xrange(size):
                b[j] = wrapper.double_matrix_getitem(self.cbackground.b, i, j)
            B.append(b)
        return (distNum, orders, B)

    def getName(self, i):
        """return the name of the ith backgound distrubution"""
        if i < self.cbackground.n:
            return self.cbackground.getName(i)

    def setName(self, i, name):
        """sets the name of the ith background distrubution to name"""
        if i < self.cbackground.n:
            self.cbackground.setName(i, name)
            self.name2id[name] = i

    def updateName2id(self):
        """adds all background names to the dictionary name2id"""
        for i in xrange(self.cbackground.n):
            tmp = self.cbackground.name[i]
            if tmp is not None:
                self.name2id[tmp] = i


#-------------------------------------------------------------------------------
#- HMM and derived
class HMM(object):
    """ The HMM base class.

    All functions where the C signatures allows it will be defined in here.
    Unfortunately there stil is a lot of overloading going on in derived classes.

    Generic features (these apply to all derived classes):
    - Forward algorithm
    - Viterbi algorithm
    - Baum-Welch training
    - HMM distance metric
    - ...

    """

    def __init__(self, emissionDomain, distribution, cmodel):
        self.emissionDomain = emissionDomain
        self.distribution = distribution
        self.cmodel = cmodel

        self.N = self.cmodel.N  # number of states
        self.M = self.cmodel.M  # number of symbols / mixture components
        self.name2id = dict()
        self.updateName2id()

    @property
    def maxorder(self):
        Log.error("Not allowed")

    @maxorder.setter
    def maxorder(self, value):
        Log.error("Not allowed")

    @property
    def model_type(self):
        Log.error("Not allowed")

    @model_type.setter
    def model_type(self, value):
        Log.error("Not allowed")


    def __del__(self):
        """ Deallocation routine for the underlying C data structures. """
        Log.note("__del__ HMM" + str(self.cmodel))

    def loglikelihood(self, emissionSequences):
        """ Compute Math.log( P[emissionSequences| model]) using the forward algorithm
        assuming independence of the sequences in emissionSequences

        @param emissionSequences can either be a SequenceSet or a EmissionSequence

        @returns Math.log( P[emissionSequences| model]) of type float which is
        computed as \f$\sum_{s} Math.log( P[s| model])\f$ when emissionSequences
        is a SequenceSet

        @note The implementation does not compute the full forward matrix since
        we are only interested in the likelihoods in this case.
        """
        return sum(self.loglikelihoods(emissionSequences))


    def loglikelihoods(self, emissionSequences):
        """ Compute a vector ( Math.log( P[s| model]) )_{s} of log-likelihoods of the
        individual emission_sequences using the forward algorithm

        @param emissionSequences is of type SequenceSet

        @returns Math.log( P[emissionSequences| model]) of type float
        (numarray) vector of floats

        """
        # Log.note("HMM.loglikelihoods() -- begin")
        emissionSequences = emissionSequences.asSequenceSet()
        seqNumber = len(emissionSequences)

        likelihoodList = []

        for i in range(seqNumber):
            # Log.note("getting likelihood for sequence %i\n" % i)
            seq = emissionSequences.cseq.getSequence(i)
            tmp = emissionSequences.cseq.getLength(i)

            try:
                likelihood = self.cmodel.logp(seq, tmp)
                likelihoodList.append(likelihood)
            except Exception, e:
                self.cmodel.logp(seq, tmp)
                Log.warning("forward returned -1: Sequence " + str(i) + " cannot be build.", e)
                # XXX TODO Eventually this should trickle down to C-level
                # Returning -DBL_MIN instead of infinity is stupid, since the latter allows
                # to continue further computations with that inf, which causes
                # things to blow up later.
                # cmodel.logp() could do without a return value if -Inf is returned
                # What should be the semantics in case of computing the likelihood of
                # a set of sequences
                likelihoodList.append(-float('Inf'))

        del emissionSequences
        # Log.note("HMM.loglikelihoods() -- end")
        return likelihoodList

    # Further Marginals ...
    def pathPosterior(self, sequence, path):
        """
        @returns the log posterior probability for 'path' having generated
        'sequence'.

        @attention pathPosterior needs to calculate the complete forward and
        backward matrices. If you are interested in multiple paths it would
        be more efficient to use the 'posterior' function directly and not
        multiple calls to pathPosterior

        @todo for silent states things are more complicated -> to be done
        """
        # XXX TODO for silent states things are more complicated -> to be done
        if self.hasFlags(kSilentStates):
            raise NotImplementedError("Models with silent states not yet supported.")

        # calculate complete posterior matrix
        post = self.posterior(sequence)
        path_posterior = []

        if not self.hasFlags(kSilentStates):
            # if there are no silent states things are straightforward
            assert len(path) == len(sequence), "Path and sequence have different lengths"

            # appending posteriors for each element of path
            for p, state in enumerate(path):
                try:
                    path_posterior.append(post[p][state])
                except IndexError:
                    Log.error("Invalid state index " + str(state) + ". Model and path are incompatible")
            return path_posterior
            #        # XXX TODO silent states are yet to be done
            #        else:
            #            # for silent state models we have to propagate the silent states in each column of the
            #            # posterior matrix
            #
            #            assert not self.isSilent(path[0]), "First state in path must not be silent."
            #
            #            j = 0   # path index
            #            for i in range(len(sequence)):
            #                pp = post[i][path[j]]
            #
            #                print pp
            #
            #                if pp == 0:
            #                    return float('-inf')
            #                else:
            #                    path_log_lik += Math.log(post[p][path[p]])
            #                    j+=1
            #
            #
            #                # propagate path up until the next emitting state
            #                while self.isSilent(path[j]):
            #
            #                    print "** silent state ",path[j]
            #
            #                    pp =  post[i][path[j]]
            #                    if pp == 0:
            #                        return float('-inf')
            #                    else:
            #                        path_log_lik += Math.log(post[p][path[p]])
            #                        j+=1
            #
            #            return path_log_lik

    def statePosterior(self, sequence, state, time):
        """
        @returns the log posterior probability for being at 'state'
        at time 'time' in 'sequence'.

        @attention: statePosterior needs to calculate the complete forward
        and backward matrices. If you are interested in multiple states
        it would be more efficient to use the posterior function directly
        and not multiple calls to statePosterior

        @todo for silent states things are more complicated -> to be done
        """
        # XXX TODO for silent states things are more complicated -> to be done
        if self.hasFlags(kSilentStates):
            raise NotImplementedError("Models with silent states not yet supported.")

        # checking function arguments
        if not 0 <= time < len(sequence):
            Log.error("Invalid sequence index: " + str(time) + " (sequence has length " + str(len(sequence)) + " ).")
        if not 0 <= state < self.N:
            Log.error("Invalid state index: " + str(state) + " (models has " + str(self.N) + " states ).")

        post = self.posterior(sequence)
        return post[time][state]


    def posterior(self, sequence):
        """ Posterior distribution matrix for 'sequence'.

        @todo for silent states things are more complicated -> to be done
        """
        # XXX TODO for silent states things are more complicated -> to be done
        if self.hasFlags(kSilentStates):
            raise NotImplementedError("Models with silent states not yet supported.")

        if not isinstance(sequence, EmissionSequence):
            Log.error("Input to posterior must be EmissionSequence object")

        (alpha, scale) = self.forward(sequence)
        beta = self.backward(sequence, scale)

        return map(lambda v, w: map(lambda x, y: x * y, v, w), alpha, beta)


    def joined(self, emissionSequence, stateSequence):
        """ log P[ emissionSequence, stateSequence| m] """

        if not isinstance(emissionSequence, EmissionSequence):
            Log.error("EmissionSequence required, got " + str(emissionSequence.__class__.__name__))

        seqdim = emissionSequence.emissionDomain.dimension
        if seqdim < 1:
            Log.error("not expected")

        t = len(emissionSequence)
        s = len(stateSequence)

        if t / seqdim != s and not self.hasFlags(kSilentStates):
            Log.error("sequence and state sequence have different lengths " +
                      "but the model has no silent states.")

        seq = emissionSequence.cseq.getSequence(0)
        states = stateSequence

        logp = self.cmodel.logp_joint(seq, t, states, s)
        return logp

    # The functions for model training are defined in the derived classes.
    def baumWelch(self, trainingSequences, nrSteps=wrapper.MAX_ITER_BW, loglikelihoodCutoff=wrapper.EPS_ITER_BW):
        raise NotImplementedError("to be defined in derived classes")

    def baumWelchSetup(self, trainingSequences, nrSteps):
        raise NotImplementedError("to be defined in derived classes")

    def baumWelchStep(self, nrSteps, loglikelihoodCutoff):
        raise NotImplementedError("to be defined in derived classes")

    def baumWelchDelete(self):
        raise NotImplementedError("to be defined in derived classes")

    # extern double ghmm_c_prob_distance(smodel *cm0, smodel *cm, int maxT, int symmetric, int verbose);
    def distance(self, model, seqLength):
        """
        @returns the distance between 'self.cmodel' and 'model'.
        """
        return self.cmodel.prob_distance(model.cmodel, seqLength, 0, 0)


    def forward(self, emissionSequence):
        """
        @returns the (N x T)-matrix containing the forward-variables
        and the scaling vector
        """
        Log.note("HMM.forward -- begin")
        # XXX Allocations should be in try, except, finally blocks
        # to assure deallocation even in the case of errrors.
        # This will leak otherwise.
        seq = emissionSequence.cseq.getSequence(0)

        t = len(emissionSequence)
        calpha = wrapper.double_matrix_alloc(t, self.N)
        cscale = wrapper.double_array_alloc(t)

        self.cmodel.forward(seq, t, calpha, cscale)

        Log.note("HMM.forward -- end")
        return calpha, cscale


    def backward(self, emissionSequence, scalingVector):
        """
        @returns the (N x T)-matrix containing the backward-variables
        """
        Log.note("HMM.backward -- begin")
        seq = emissionSequence.cseq.getSequence(0)

        # alllocating beta matrix
        t = len(emissionSequence)
        beta = wrapper.double_matrix_alloc(t, self.N)
        self.cmodel.backward(seq, t, beta, scalingVector)

        Log.note("HMM.backward -- end")
        return beta


    def viterbi(self, eseqs):
        """ Compute the Viterbi-path for each sequence in emissionSequences

        @param eseqs can either be a SequenceSet or an EmissionSequence

        @returns [q_0, ..., q_T] the viterbi-path of \p eseqs is an
        EmmissionSequence object,
        [[q_0^0, ..., q_T^0], ..., [q_0^k, ..., q_T^k]} for a k-sequence
        SequenceSet
        """
        Log.note("HMM.viterbi() -- begin")
        emissionSequences = eseqs.asSequenceSet()

        seqNumber = len(emissionSequences)

        allLogs = []
        allPaths = []
        for i in range(seqNumber):
            seq = emissionSequences.cseq.getSequence(i)
            seq_len = emissionSequences.cseq.getLength(i)

            if seq_len > 0:
                viterbiPath, log_p = ghmm_dmodel_viterbi(self.cmodel, seq, seq_len)
            else:
                viterbiPath, log_p = ([], 0)

            allPaths.append(viterbiPath)
            allLogs.append(log_p)

        Log.note("HMM.viterbi() -- end")
        if seqNumber > 1:
            return allPaths, allLogs
        else:
            return allPaths[0], allLogs[0]


    def sample(self, seqNr, T, seed=0):
        """ Sample emission sequences.

        @param seqNr number of sequences to be sampled
        @param T maximal length of each sequence
        @param seed initialization value for rng, default 0 leaves the state
        of the rng alone
        @returns a SequenceSet object.
        """
        seqPtr = self.cmodel.generate_sequences(seed, T, seqNr, -1)
        return SequenceSet(self.emissionDomain, seqPtr)


    def sampleSingle(self, T, seed=0, native=False):
        """ Sample a single emission sequence of length at most T.

        @param T maximal length of the sequence
        @param seed initialization value for rng, default 0 leaves the state
        of the rng alone
        @returns a EmissionSequence object.
        """
        Log.note("HMM.sampleSingle() -- begin")
        seqPtr = self.cmodel.generate_sequences(seed, T, 1, -1, native=native)
        Log.note("HMM.sampleSingle() -- end")
        return EmissionSequence(self.emissionDomain, seqPtr)

    def getStateFix(self, state):
        state = self.state(state)
        s = self.cmodel.s[state]
        return s.fix

    def setStateFix(self, state, flag):
        state = self.state(state)
        s = self.cmodel.s[state]
        s.fix = flag

    def clearFlags(self, flags):
        """ Clears one or more model type flags.
        @attention Use with care.
        """
        Log.note("clearFlags: " + self.printtypes(flags))
        self.cmodel.model_type &= ~flags

    def hasFlags(self, flags):
        """ Checks if the model has one or more model type flags set
        """
        return self.cmodel.model_type & flags

    def setFlags(self, flags):
        """ Sets one or more model type flags.
        @attention Use with care.
        """
        Log.note("setFlags: " + self.printtypes(flags))
        self.cmodel.model_type |= flags

    def state(self, stateLabel):
        """ Given a stateLabel return the integer index to the state

        """
        return self.name2id[stateLabel]

    def getInitial(self, i):
        """ Accessor function for the initial probability \f$\pi_i\f$ """
        state = self.cmodel.s[i]
        return state.pi

    def setInitial(self, i, prob, fixProb=False):
        """ Accessor function for the initial probability \f$\pi_i\f$.

        If 'fixProb' = True \f$\pi\f$ will be rescaled to 1 with 'pi[i]'
        fixed to the arguement value of 'prob'.

        """
        state = self.cmodel.s[i]
        old_pi = state.pi
        state.pi = prob

        # renormalizing pi, pi(i) is fixed on value 'prob'
        if fixProb:
            coeff = (1.0 - old_pi) / prob
            for j in range(self.N):
                if i != j:
                    state = self.cmodel.s[j]
                    p = state.pi
                    state.pi = p / coeff

    def getTransition(self, i, j):
        """ Accessor function for the transition a_ij """
        i = self.state(i)
        j = self.state(j)

        transition = self.cmodel.get_transition(i, j)
        if transition < 0.0:
            transition = 0.0
        return transition

    def setTransition(self, i, j, prob):
        """ Accessor function for the transition a_ij. """
        i = self.state(i)
        j = self.state(j)

        if not 0.0 <= prob <= 1.0:
            Log.error("Transition " + str(prob) + " is not a probability.")

        self.cmodel.set_transition(i, j, prob)


    def getEmission(self, i):
        """
        Accessor function for the emission distribution parameters of state 'i'.

        For discrete models the distribution over the symbols is returned,
        for continuous models a matrix of the form
        [ [mu_1, sigma_1, weight_1] ... [mu_M, sigma_M, weight_M]  ] is returned.

        """
        raise NotImplementedError

    def setEmission(self, i, distributionParemters):
        """ Set the emission distribution parameters

        Defined in derived classes.
         """
        raise NotImplementedError

    def asMatrices(self):
        "To be defined in derived classes."
        raise NotImplementedError


    def normalize(self):
        """ Normalize transition probs, emission probs (if applicable)
        """
        Log.note("Normalizing now.")

        self.cmodel.normalize()

    def randomize(self, noiseLevel):
        """ to be defined in derived class """
        raise NotImplementedError

    def write(self, fileName):
        """ Writes HMM to file 'fileName'.

        """
        self.cmodel.write_xml(fileName)


    def printtypes(self, model_type):
        strout = []
        if model_type == kNotSpecified:
            return 'kNotSpecified'
        for k, v in types.__dict__.items():
            if v == -1:
                continue
            if not isinstance(v, int):
                continue
            if model_type & v:
                strout.append(k)
        return ' '.join(strout)

    def updateName2id(self):
        """adds all state names to the dictionary name2id"""
        for i in xrange(self.cmodel.N):
            self.name2id[i] = i
            if self.cmodel.getStateName(i) is not None:
                self.name2id[self.cmodel.getStateName(i)] = i

    def setStateName(self, index, name):
        """sets the state name of state index to name"""
        self.cmodel.setStateName(index, name)
        self.name2id[name] = index

    def getStateName(self, index):
        """returns the name of the state index"""
        return self.cmodel.getStateName(index)


class DiscreteEmissionHMM(HMM):
    """ HMMs with discrete emissions.

    Optional features:
    - silent states
    - higher order states
    - parameter tying in training
    - background probabilities in training
    """

    def __init__(self, emissionDomain, distribution, cmodel):
        HMM.__init__(self, emissionDomain, distribution, cmodel)

        # self.model_type = self.cmodel.model_type  # model type
        # self.maxorder = self.cmodel.maxorder
        self.background = None

    def __str__(self):
        hmm = self.cmodel
        strout = [str(self.__class__.__name__)]
        if self.cmodel.name:
            strout.append(" " + str(self.cmodel.name))
        strout.append("(N=" + str(hmm.N))
        strout.append(", M=" + str(hmm.M) + ')\n')

        f = lambda x: "%.2f" % (x,) # float rounding function

        if self.hasFlags(kHigherOrderEmissions):
            order = wrapper.int_array2list(self.cmodel.order, self.N)
        else:
            order = [0] * hmm.N

        if hmm.N <= 4:
            iter_list = range(self.N)
        else:
            iter_list = [0, 1, 'X', hmm.N - 2, hmm.N - 1]

        for k in iter_list:
            if k == 'X':
                strout.append('\n  ...\n\n')
                continue

            state = hmm.s[k]
            strout.append("  state " + str(k) + ' (')
            if order[k] > 0:
                strout.append('order=' + str(order[k]) + ',')

            strout.append("initial=" + f(state.pi) + ')\n')
            strout.append("    Emissions: ")
            for outp in range(hmm.M ** (order[k] + 1)):
                strout.append(f(state.b[outp]))
                if outp < hmm.M ** (order[k] + 1) - 1:
                    strout.append(', ')
                else:
                    strout.append('\n')

            strout.append("    Transitions:")
            #trans = [0.0] * hmm.N
            strout.append(','.join([" ->" + str(state.getOutState(i)) + ' (' + f(a) + ')' for i, a in enumerate(state.out_a)]))
            strout.append('\n')

        return join(strout, '')


    def verboseStr(self):
        hmm = self.cmodel
        strout = ["\nGHMM Model\n"]
        strout.append("Name: " + str(self.cmodel.name))
        strout.append("\nModelflags: " + self.printtypes(self.cmodel.model_type))
        strout.append("\nNumber of states: " + str(hmm.N))
        strout.append("\nSize of Alphabet: " + str(hmm.M))
        if self.hasFlags(kHigherOrderEmissions):
            order = wrapper.int_array2list(self.cmodel.order, self.N)
        else:
            order = [0] * hmm.N

        for k in range(hmm.N):
            state = hmm.s[k]
            strout.append("\n\nState number " + str(k) + ":")
            if state.desc is not None:
                strout.append("\nState Name: " + state.desc)
            strout.append("\nState order: " + str(order[k]))
            strout.append("\nInitial probability: " + str(state.pi))
            #strout.append("\nsilent state: " + str(self.cmodel.silent[k]))
            strout.append("\nOutput probabilites: ")
            for outp in range(hmm.M ** (order[k] + 1)):
                strout.append(str(state.b[ outp]))
                if outp % hmm.M == hmm.M - 1:
                    strout.append("\n")
                else:
                    strout.append(", ")

            strout.append("\nOutgoing transitions:")
            for i, a in enumerate(state.out_a):
                strout.append("\ntransition to state " + str(i))
                strout.append(" with probability " + str(a))
            strout.append("\nIngoing transitions:")
            for i, _ in enumerate(state.in_a):
                strout.append("\ntransition from state " + str(i))
                strout.append(" with probability " + str(state.in_a[ i]))
            strout.append("\nint fix:" + str(state.fix) + "\n")

        if self.hasFlags(kSilentStates):
            strout.append("\nSilent states: \n")
            for k in range(hmm.N):
                strout.append(str(self.cmodel.getSilent(k)) + ", ")
        strout.append("\n")
        return join(strout, '')


    def extendDurations(self, durationlist):
        """ extend states with durations larger than one.

        @note this done by explicit state copying in C
        """

        for i in range(len(durationlist)):
            if durationlist[i] > 1:
                error = self.cmodel.duration_apply(i, durationlist[i])
                if error:
                    Log.error("durations not applied")
                else:
                    self.N = self.cmodel.N

    def getEmission(self, i):
        i = self.state(i)
        state = self.cmodel.s[i]
        if self.hasFlags(kHigherOrderEmissions):
            order = wrapper.int_array_getitem(self.cmodel.order, i)
            emissions = wrapper.double_array2list(state.b, self.M ** (order + 1))
        else:
            emissions = wrapper.double_array2list(state.b, self.M)
        return emissions

    def setEmission(self, i, distributionParameters):
        """ Set the emission distribution parameters for a discrete model."""
        i = self.state(i)
        if not len(distributionParameters) == self.M:
            Log.error("Can not handle more than zero-order emmisions at this time")

        state = self.cmodel.s[i]

        # updating silent flag and/or model type if necessary
        if self.hasFlags(kSilentStates):
            if sum(distributionParameters) == 0.0:
                self.cmodel.setSilent(i, 1)
            else:
                self.cmodel.setSilent(i, 0)
                #change model_type and free array if no silent state is left
                if 0 == sum(wrapper.int_array2list(self.cmodel.silent, self.N)):
                    self.clearFlags(kSilentStates)
                    self.cmodel.silent = None
        #if the state becomes the first silent state allocate memory and set the silen flag
        elif sum(distributionParameters) == 0.0:
            self.setFlags(kSilentStates)
            slist = [0] * self.N
            slist[i] = 1
            self.cmodel.silent = slist

        #CODE ASSUMES THE ORDER IS ZERO
        self.clearFlags(kHigherOrderEmissions)
        if self.cmodel.order is None:
            self.cmodel.order = [0] * self.M

        self.cmodel.order[i] = int(Math.log(len(distributionParameters), self.M)) - 1
        if self.M ** (self.cmodel.order[i] + 1) != len(distributionParameters):
            Log.error("distributionParameters has wrong length")
        self.cmodel.maxorder = max(self.cmodel.order)
        if self.cmodel.maxorder > 0:
            self.setFlags(kHigherOrderEmissions)

        #set the emission probabilities
        state.b = distributionParameters


    # XXX Change name?
    def backwardTermination(self, emissionSequence, beta, scalingVector):
        """
        Result: the backward log probability of emissionSequence
        """
        seq = emissionSequence.cseq.getSequence(0)
        t = len(emissionSequence)

        logp = self.cmodel.backward_termination(seq, t, beta, scalingVector)
        return logp

    def baumWelch(self, trainingSequences, nrSteps=wrapper.MAX_ITER_BW, loglikelihoodCutoff=wrapper.EPS_ITER_BW):
        """ Reestimates the model with the sequence in 'trainingSequences'.

        @note that training for models including silent states is not yet
        supported.

        @param trainingSequences EmissionSequence or SequenceSet object
        @param nrSteps the maximal number of BW-steps
        @param loglikelihoodCutoff the least relative improvement in likelihood
        with respect to the last iteration required to continue.

        """
        if not isinstance(trainingSequences, EmissionSequence) and not isinstance(trainingSequences, SequenceSet):
            Log.error("EmissionSequence or SequenceSet required, got " + str(trainingSequences.__class__.__name__))

        if self.hasFlags(kSilentStates):
            raise NotImplementedError("Sorry, training of models containing silent states not yet supported.")

        self.cmodel.baum_welch_nstep(trainingSequences.cseq, nrSteps, loglikelihoodCutoff)

    def fbGibbs(self, trainingSequences, pA, pB, pPi, burnIn=100, seed=0):
        """Reestimates the model and returns a sampled state sequence

        @note uses gsl, silent states not supported

        @param seed int for random seed, 0 default
        @param trainingSequences EmissionSequence
        @param pA prior count for transitions
        @param pB prior count for emissions
        @param pPI prior count for initial state
        @param burnin number of iterations
        @return set of sampled paths for each training sequence
        @warning work in progress
        """
        if not isinstance(trainingSequences, EmissionSequence) and not isinstance(trainingSequences, SequenceSet):
            Log.error("EmissionSequence or SequenceSet required, got " + str(trainingSequences.__class__.__name__))
        if self.hasFlags(kSilentStates):
            Log.error("Sorry, training of models containing silent states not yet supported.")
        A, i = ghmmhelper.list2double_matrix(pA)
        if self.hasFlags(kHigherOrderEmissions):
            B = wrapper.double_matrix_alloc_row(len(pB))
            for i in range(len(pB)):
                wrapper.double_matrix_set_col(B, i, wrapper.list2double_array(pB[i]))
        else:
            B, j = ghmmhelper.list2double_matrix(pB)
        Pi = wrapper.list2double_array(pPi)

        return ghmmhelper.int_matrix2list(self.cmodel.fbgibbs(trainingSequences.cseq, A, B, Pi, burnIn, seed), trainingSequences.cseq.seq_number, len(trainingSequences))

    def cfbGibbs(self, trainingSequences, pA, pB, pPi, R=-1, burnIn=100, seed=0):
        """Reestimates the model and returns a sampled state sequence

        @note uses gsl, silent states not supported

        @param seed int for random seed, 0 default
        @param trainingSequences EmissionSequence or SequenceSet
        @param pA prior count for transitions
        @param pB prior count for emissions
        @param pPI prior count for initial state
        @param R length of uniform compression >0, works best for .5Math.log(sqrt(T)) where T is length of seq
        @param burnin number of iterations
        @return set of sampled paths for each training sequence
        @warning work in progress
        """
        if not isinstance(trainingSequences, EmissionSequence) and not isinstance(trainingSequences, SequenceSet):
            Log.error("EmissionSequence or SequenceSet required, got " + str(trainingSequences.__class__.__name__))

        if self.hasFlags(kSilentStates):
            Log.error("Sorry, training of models containing silent states not yet supported.")
        if R is -1:
            R = int(math.ceil(.5 * Math.log(math.sqrt(len(trainingSequences)))))
            #print R
        if R <= 1:
            R = 2
        A, i = ghmmhelper.list2double_matrix(pA)
        if self.hasFlags(kHigherOrderEmissions):
            B = wrapper.double_matrix_alloc_row(len(pB))
            for i in range(len(pB)):
                wrapper.double_matrix_set_col(B, i, wrapper.list2double_array(pB[i]))
        else:
            B, j = ghmmhelper.list2double_matrix(pB)
        Pi = wrapper.list2double_array(pPi)
        return ghmmhelper.int_matrix2list(self.cmodel.cfbgibbs(trainingSequences.cseq, A, B, Pi, R, burnIn, seed), trainingSequences.cseq.seq_number, len(trainingSequences))

    def applyBackgrounds(self, backgroundWeight):
        """
        Apply the background distribution to the emission probabilities of states
        which have been assigned one (usually in the editor and coded in the XML).

        applyBackground computes a convex combination of the emission probability
        and the background

        @param backgroundWeight (within [0,1]) controls the background's
        contribution for each state.
        """
        if not len(backgroundWeight) == self.N:
            Log.error("Argument 'backgroundWeight' does not match number of states.")

        cweights = backgroundWeight
        result = self.cmodel.background_apply(cweights)

        if result:
            Log.error("applyBackground failed.")


    def setBackgrounds(self, backgroundObject, stateBackground):
        """
        Configure model to use the background distributions in 'backgroundObject'.

        @param backgroundObject BackgroundDistribution
        @param 'stateBackground' a list of indixes (one for each state) refering
        to distributions in 'backgroundObject'.

        @note values in backgroundObject are deep copied into the model
        """

        if not isinstance(backgroundObject, BackgroundDistribution):
            Log.error("BackgroundDistribution required, got " + str(backgroundObject.__class__.__name__))

        if not type(stateBackground) == list:
            Log.error("list required got " + str(type(stateBackground)))

        if not len(stateBackground) == self.N:
            Log.error("Argument 'stateBackground' does not match number of states.")

        self.background = backgroundObject.getCopy()
        self.cmodel.bp = self.background.cbackground
        self.cmodel.background_id = wrapper.list2int_array(stateBackground)

        # updating model type
        self.setFlags(kBackgroundDistributions)

    def setBackgroundAssignments(self, stateBackground):
        """ Change all the assignments of background distributions to states.

        Input is a list of background ids or '-1' for no background, or list of background names
        """
        if not type(stateBackground) == list:
            Log.error("list required got " + str(type(stateBackground)))

        assert self.cmodel.background_id is not None, "Error: No backgrounds defined in model."
        assert len(stateBackground) == self.N, "Error: Number of weigths does not match number of states."
        # check for valid background id
        for d in stateBackground:
            if type(d) == str:
                assert self.background.name2id.has_key(d), "Error:  Invalid background distribution name."
                d = self.background.name2id[d]
            assert d in range(self.background.cbackground.n), "Error: Invalid background distribution id."

        for i, b_id in enumerate(stateBackground):
            if type(b_id) == str:
                b_id = self.background.name2id[b_id]
            wrapper.int_array_setitem(self.cmodel.background_id, i, b_id)


    def getBackgroundAssignments(self):
        """ Get the background assignments of all states

        '-1' -> no background
        """
        if self.hasFlags(kBackgroundDistributions):
            return wrapper.int_array2list(self.cmodel.background_id, self.N)


    def updateTiedEmissions(self):
        """ Averages emission probabilities of tied states. """
        assert self.hasFlags(kTiedEmissions) and self.cmodel.tied_to is not None, "cmodel.tied_to is undefined."
        self.cmodel.update_tie_groups()


    def setTieGroups(self, tieList):
        """ Sets the tied emission groups

        @param tieList contains for every state either '-1' or the index
        of the tied emission group leader.

        @note The tied emission group leader is tied to itself
        """
        if len(tieList) != self.N:
            Log.error("Number of entries in tieList is different from number of states.")

        if self.cmodel.tied_to is None:
            Log.note("allocating tied_to")
            self.cmodel.tied_to = list(tieList)
            self.setFlags(kTiedEmissions)
        else:
            Log.note("tied_to already initialized")
            for i in range(self.N):
                wrapper.int_array_setitem(self.cmodel.tied_to, i, tieList[i])


    def removeTieGroups(self):
        """ Removes all tied emission information. """
        if self.hasFlags(kTiedEmissions) and self.cmodel.tied_to != None:
            self.cmodel.tied_to = None
            self.clearFlags(kTiedEmissions)

    def getTieGroups(self):
        """ Gets tied emission group structure. """
        if not self.hasFlags(kTiedEmissions) or self.cmodel.tied_to is None:
            Log.error("HMM has no tied emissions or self.cmodel.tied_to is undefined.")

        return wrapper.int_array2list(self.cmodel.tied_to, self.N)


    def getSilentFlag(self, state):
        state = self.state(state)
        if self.hasFlags(kSilentStates):
            return self.cmodel.getSilent(state)
        else:
            return 0

    def asMatrices(self):
        "Return the parameters in matrix form."
        A = []
        B = []
        pi = []
        if self.hasFlags(kHigherOrderEmissions):
            order = wrapper.int_array2list(self.cmodel.order, self.N)
        else:
            order = [0] * self.N

        for i in range(self.cmodel.N):
            A.append([0.0] * self.N)
            state = self.cmodel.s[i]
            pi.append(state.pi)
            B.append(wrapper.double_array2list(state.b, self.M ** (order[i] + 1)))
            for j, a in enumerate(state.out_a):
                A[i][j] = state.out_a[j]

        return [A, B, pi]


    def isSilent(self, state):
        """
        @returns True if 'state' is silent, False otherwise
        """
        state = self.state(state)
        if not 0 <= state <= self.N - 1:
            Log.error("Invalid state index")

        if self.hasFlags(kSilentStates) and self.cmodel.silent[state]:
            return True
        else:
            return False

    def write(self, fileName):
        """
        Writes HMM to file 'fileName'.
        """
        if self.cmodel.alphabet is None:
            self.cmodel.alphabet = self.emissionDomain

        self.cmodel.write_xml(fileName)


######################################################
class StateLabelHMM(DiscreteEmissionHMM):
    """ Labelled HMMs with discrete emissions.

        Same feature list as in DiscreteEmissionHMM models.
    """

    def __init__(self, emissionDomain, distribution, labelDomain, cmodel):
        DiscreteEmissionHMM.__init__(self, emissionDomain, distribution, cmodel)

        if not isinstance(labelDomain, LabelDomain):
            Log.error("Invalid labelDomain")

        self.labelDomain = labelDomain


    def __str__(self):
        hmm = self.cmodel
        strout = [str(self.__class__.__name__)]
        if self.cmodel.name:
            strout.append(" " + str(self.cmodel.name))
        strout.append("(N= " + str(hmm.N))
        strout.append(", M= " + str(hmm.M) + ')\n')

        f = lambda x: "%.2f" % (x,) # float rounding function

        if self.hasFlags(kHigherOrderEmissions):
            order = wrapper.int_array2list(self.cmodel.order, self.N)
        else:
            order = [0] * hmm.N
        label = wrapper.int_array2list(hmm.label, self.N)

        if hmm.N <= 4:
            iter_list = range(self.N)
        else:
            iter_list = [0, 1, 'X', hmm.N - 2, hmm.N - 1]

        for k in iter_list:
            if k == 'X':
                strout.append('\n  ...\n\n')
                continue

            state = hmm.s[k]
            strout.append("  state " + str(k) + ' (')
            if order[k] > 0:
                strout.append('order= ' + str(order[k]) + ',')

            strout.append("initial= " + f(state.pi) + ', label= ' + str(self.labelDomain.external(label[k])) + ')\n')
            strout.append("    Emissions: ")
            for outp in range(hmm.M ** (order[k] + 1)):
                strout.append(f(state.b[ outp]))
                if outp < hmm.M ** (order[k] + 1) - 1:
                    strout.append(', ')
                else:
                    strout.append('\n')

            strout.append("    Transitions:")
            #trans = [0.0] * hmm.N
            strout.append(','.join([" ->" + str(i) + ' (' + f(a) + ')' for i, a in enumerate(state.out_a)]))
            strout.append('\n')

        return join(strout, '')


    def verboseStr(self):
        hmm = self.cmodel
        strout = ["\nGHMM Model\n"]
        strout.append("Name: " + str(self.cmodel.name))
        strout.append("\nModelflags: " + self.printtypes(self.cmodel.model_type))
        strout.append("\nNumber of states: " + str(hmm.N))
        strout.append("\nSize of Alphabet: " + str(hmm.M))

        if hmm.model_type & kHigherOrderEmissions:
            order = wrapper.int_array2list(hmm.order, self.N)
        else:
            order = [0] * hmm.N
        label = wrapper.int_array2list(hmm.label, self.N)
        for k in range(hmm.N):
            state = hmm.s[k]
            strout.append("\n\nState number " + str(k) + ":")
            if state.desc is not None:
                strout.append("\nState Name: " + state.desc)
            strout.append("\nState label: " + str(self.labelDomain.external(label[k])))

            strout.append("\nState order: " + str(order[k]))
            strout.append("\nInitial probability: " + str(state.pi))
            strout.append("\nOutput probabilites:\n")
            for outp in range(hmm.M ** (order[k] + 1)):
                strout += str(state.b[ outp])
                if outp % hmm.M == hmm.M - 1:
                    strout.append("\n")
                else:
                    strout.append(", ")

            strout.append("Outgoing transitions:")
            for i, a in enumerate(state.out_a):
                strout.append("\ntransition to state " + str(i) + " with probability " + str(state.getOutProb(i)))
            strout.append("\nIngoing transitions:")
            for i, _ in enumerate(state.in_a):
                strout.append("\ntransition from state " + str(i) + " with probability " + str(state.getInProb(i)))
            strout.append("\nint fix:" + str(state.fix) + "\n")

        if hmm.model_type & kSilentStates:
            strout.append("\nSilent states: \n")
            for k in range(hmm.N):
                strout.append(str(hmm.silent[k]) + ", ")
            strout.append("\n")

        return join(strout, '')

    def setLabels(self, labelList):
        """  Set the state labels to the values given in labelList.

        LabelList is in external representation.
        """

        assert len(labelList) == self.N, "Invalid number of labels."

        # set state label to to the appropiate index
        for i in range(self.N):
            if not self.labelDomain.isAdmissable(labelList[i]):
                Log.error("Label " + str(labelList[i]) + " not included in labelDomain.")

        self.cmodel.label = wrapper.list2int_array([self.labelDomain.internal(l) for l in labelList])

    def getLabels(self):
        labels = wrapper.int_array2list(self.cmodel.label, self.N)
        return [self.labelDomain.external(l) for l in labels]

    def getLabel(self, stateIndex):
        """
        @returns label of the state 'stateIndex'.
        """
        return self.cmodel.label[stateIndex]

    def externalLabel(self, internal):
        """
        @returns label representation of an int or list of ints
        """

        if type(internal) is int:
            return self.labelDomain.external[internal] # return Label
        elif type(internal) is list:
            return self.labelDomain.externalSequence(internal)
        else:
            Log.error('int or list needed')

    def internalLabel(self, external):
        """
        @returns int representation of an label or list of labels
        """

        if type(external) is list:
            return self.labelDomain.internalSequence(external)
        else:
            return self.labelDomain.internal(external)

    def sampleSingle(self, seqLength, seed=0, native=False):
        seqPtr = self.cmodel.label_generate_sequences(seed, seqLength, 1, seqLength, native=native)
        return EmissionSequence(self.emissionDomain, seqPtr, labelDomain=self.labelDomain)

    def sample(self, seqNr, seqLength, seed=0):
        seqPtr = self.cmodel.label_generate_sequences(seed, seqLength, seqNr, seqLength)
        return SequenceSet(self.emissionDomain, seqPtr, labelDomain=self.labelDomain)


    def labeledViterbi(self, emissionSequences):
        """
        @returns the labeling of the input sequence(s) as given by the viterbi
        path.

        For one EmissionSequence a list of labels is returned; for an SequenceSet
        a list of lists of labels.

        """
        emissionSequences = emissionSequences.asSequenceSet()
        seqNumber = len(emissionSequences)

        if not emissionSequences.emissionDomain == self.emissionDomain:
            Log.error("Sequence and model emissionDomains are incompatible.")

        vPath, log_p = self.viterbi(emissionSequences)

        f = lambda i: self.labelDomain.external(self.getLabel(i))
        if seqNumber == 1:
            labels = map(f, vPath)
        else:
            labels = [map(f, vp) for vp in vPath]

        return (labels, log_p)


    def kbest(self, emissionSequences, k=1):
        """ Compute the k probable labeling for each sequence in emissionSequences

        @param emissionSequences can either be a SequenceSet or an
        EmissionSequence
        @param k the number of labelings to produce

        Result: [l_0, ..., l_T] the labeling of emissionSequences is an
        EmmissionSequence object,
        [[l_0^0, ..., l_T^0], ..., [l_0^j, ..., l_T^j]} for a j-sequence
        SequenceSet
        """
        if self.hasFlags(kSilentStates):
            Log.error("Sorry, k-best decoding on models containing silent states not yet supported.")

        emissionSequences = emissionSequences.asSequenceSet()
        seqNumber = len(emissionSequences)

        allLogs = []
        allLabels = []

        for i in range(seqNumber):
            seq = emissionSequences.cseq.getSequence(i)
            seq_len = emissionSequences.cseq.getLength(i)

            labeling, log_p = ghmm_dmodel_label_kbest(self.cmodel, seq, seq_len, k)
            oneLabel = wrapper.int_array2list(labeling, seq_len)

            allLabels.append(oneLabel)
            allLogs.append(log_p)

        if emissionSequences.cseq.seq_number > 1:
            return (map(self.externalLabel, allLabels), allLogs)
        else:
            return (self.externalLabel(allLabels[0]), allLogs[0])


    def gradientSearch(self, emissionSequences, eta=.1, steps=20):
        """ trains a model with given sequences using a gradient descent algorithm

        @param emissionSequences can either be a SequenceSet or an
        EmissionSequence
        @param eta algortihm terminates if the descent is smaller than eta
        @param steps number of iterations
        """

        # check for labels
        if not self.hasFlags(kLabeledStates):
            Log.error("Error: Model is no labeled states.")

        emissionSequences = emissionSequences.asSequenceSet()
        seqNumber = len(emissionSequences)

        try:
            self.cmodel = ghmm_dmodel_label_gradient_descent(self.cmodel, emissionSequences.cseq, eta, steps)
        except Exception, e:
            Log.error("Gradient descent finished not successfully.", e)


    def labeledlogikelihoods(self, emissionSequences):
        """ Compute a vector ( Math.log( P[s,l| model]) )_{s} of log-likelihoods of the
        individual \p emissionSequences using the forward algorithm

        @param emissionSequences SequenceSet

        Result: Math.log( P[emissionSequences,labels| model]) of type float
        (numarray) vector of floats
        """
        emissionSequences = emissionSequences.asSequenceSet()
        seqNumber = len(emissionSequences)

        if emissionSequences.cseq.state_labels is None:
            Log.error("Sequence needs to be labeled.")

        likelihoodList = []

        for i in range(seqNumber):
            seq = emissionSequences.cseq.getSequence(i)
            labels = wrapper.int_matrix_get_col(emissionSequences.cseq.state_labels, i)
            tmp = emissionSequences.cseq.getLength(i)

            try:
                likelihood = self.cmodel.label_logp(seq, labels, tmp)
                likelihoodList.append(likelihood)
            except Exception, e:
                Log.warning("forward returned -1: Sequence" + str(i) + "cannot be build.", e)
                likelihoodList.append(-float('Inf'))

        return likelihoodList

    def labeledForward(self, emissionSequence, labelSequence):
        """

        Result: the (N x T)-matrix containing the forward-variables
        and the scaling vector
        """
        if not isinstance(emissionSequence, EmissionSequence):
            Log.error("EmissionSequence required, got " + str(emissionSequence.__class__.__name__))

        n_states = self.cmodel.N

        t = emissionSequence.cseq.getLength(0)
        if t != len(labelSequence):
            Log.error("emissionSequence and labelSequence must have same length")

        calpha = wrapper.double_matrix_alloc(t, n_states)
        cscale = wrapper.double_array_alloc(t)

        seq = emissionSequence.cseq.getSequence(0)
        label = wrapper.list2int_array(self.internalLabel(labelSequence))

        logp = self.cmodel.label_forward(seq, label, t, calpha, cscale)

        # translate alpha / scale to python lists
        pyscale = wrapper.double_array2list(cscale, t)
        pyalpha = ghmmhelper.double_matrix2list(calpha, t, n_states)

        return (logp, pyalpha, pyscale)

    def labeledBackward(self, emissionSequence, labelSequence, scalingVector):
        """

            Result: the (N x T)-matrix containing the backward-variables
        """
        if not isinstance(emissionSequence, EmissionSequence):
            Log.error("EmissionSequence required, got " + str(emissionSequence.__class__.__name__))

        t = emissionSequence.cseq.getLength(0)
        if t != len(labelSequence):
            Log.error("emissionSequence and labelSequence must have same length")

        seq = emissionSequence.cseq.getSequence(0)
        label = wrapper.list2int_array(self.internalLabel(labelSequence))

        # parsing 'scalingVector' to C double array.
        cscale = wrapper.list2double_array(scalingVector)

        # alllocating beta matrix
        cbeta = wrapper.double_matrix_alloc(t, self.cmodel.N)
        logp = self.cmodel.label_backward(seq, label, t, cbeta, cscale)

        pybeta = ghmmhelper.double_matrix2list(cbeta, t, self.cmodel.N)

        return logp, pybeta

    def labeledBaumWelch(self, trainingSequences, nrSteps=wrapper.MAX_ITER_BW,
        loglikelihoodCutoff=wrapper.EPS_ITER_BW):
        """ Reestimates the model with the sequence in 'trainingSequences'.

        @note that training for models including silent states is not yet
        supported.

        @param trainingSequences EmissionSequence or SequenceSet object
        @param nrSteps the maximal number of BW-steps
        @param loglikelihoodCutoff the least relative improvement in likelihood
        with respect to the last iteration required to continue.

        """
        if not isinstance(trainingSequences, EmissionSequence) and not isinstance(trainingSequences, SequenceSet):
            Log.error("EmissionSequence or SequenceSet required, got " + str(trainingSequences.__class__.__name__))

        if self.hasFlags(kSilentStates):
            Log.error("Sorry, training of models containing silent states not yet supported.")

        self.cmodel.label_baum_welch_nstep(trainingSequences.cseq, nrSteps, loglikelihoodCutoff)


    def write(self, fileName):
        """ Writes HMM to file 'fileName'.

        """
        if self.cmodel.alphabet is None:
            self.cmodel.alphabet = self.emissionDomain

        if self.cmodel.label_alphabet is None:
            self.cmodel.label_alphabet = self.labelDomain

        self.cmodel.write_xml(fileName)


class GaussianEmissionHMM(HMM):
    """ HMMs with Gaussian distribution as emissions.

    """

    def __init__(self, emissionDomain, distribution, cmodel):
        HMM.__init__(self, emissionDomain, distribution, cmodel)

        # Baum Welch context, call baumWelchSetup to initalize
        self.BWcontext = None

    def getTransition(self, i, j):
        """ @returns the probability of the transition from state i to state j.
        Raises IndexError if the transition is not allowed
        """
        i = self.state(i)
        j = self.state(j)

        transition = self.cmodel.get_transition(i, j, 0)
        if transition < 0.0: # Tried to access non-existing edge:
            transition = 0.0
        return transition

    def setTransition(self, i, j, prob):
        """ Accessor function for the transition a_ij """

        i = self.state(i)
        j = self.state(j)

        if not self.cmodel.check_transition(i, j, 0):
            Log.error("No transition between state " + str(i) + " and " + str(j))

        self.cmodel.set_transition(i, j, 0, float(prob))

    def getEmission(self, i):
        """ @returns (mu, sigma^2)  """
        i = self.state(i)
        if not 0 <= i < self.N:
            Log.error("Index " + str(i) + " out of bounds.")

        state = self.cmodel.s[i]
        mu = state.getMean(0)
        sigma = state.getStdDev(0)
        return (mu, sigma)

    def setEmission(self, i, values):
        """ Set the emission distributionParameters for state i

        @param i index of a state
        @param values tuple of mu, sigma
        """
        mu, sigma = values
        i = self.state(i)

        state = self.cmodel.s[i]
        state.setMean(0, float(mu))
        state.setStdDev(0, float(sigma))

    def getEmissionProbability(self, value, i):
        """ @returns probability of emitting value in state i  """
        i = self.state(i)
        state = self.cmodel.s[i]
        p = state.calc_b(value)
        return p


    def __str__(self):
        hmm = self.cmodel
        strout = [str(self.__class__.__name__)]
        if self.cmodel.name:
            strout.append(" " + str(self.cmodel.name))
        strout.append("(N=" + str(hmm.N) + ')\n')

        f = lambda x: "%.2f" % (x,)  # float rounding function

        if hmm.N <= 4:
            iter_list = range(self.N)
        else:
            iter_list = [0, 1, 'X', hmm.N - 2, hmm.N - 1]

        for k in iter_list:
            if k == 'X':
                strout.append('\n  ...\n\n')
                continue

            state = hmm.s[k]
            strout.append("  state " + str(k) + " (")
            strout.append("initial=" + f(state.pi))
            if self.cmodel.cos > 1:
                strout.append(', cos=' + str(self.cmodel.cos))
            strout.append(", mu=" + f(state.getMean(0)) + ', ')
            strout.append("sigma=" + f(state.getStdDev(0)))
            strout.append(')\n')

            strout.append("    Transitions: ")
            if self.cmodel.cos > 1:
                strout.append("\n")

            for c in range(self.cmodel.cos):
                if self.cmodel.cos > 1:
                    strout.append('      class: ' + str(c) + ':')

                strout.append(','.join(['->' + str(i) + ' (' + f(state.getOutProb(i, c)) + ')' for i, a in enumerate(state.out_a)]))
                strout.append('\n')

        return join(strout, '')


    def verboseStr(self):
        hmm = self.cmodel
        strout = ["\nHMM Overview:"]
        strout.append("\nNumber of states: " + str(hmm.N))
        strout.append("\nNumber of mixture components: " + str(hmm.M))

        for k in range(hmm.N):
            state = hmm.s[k]
            strout.append("\n\nState number " + str(k) + ":")
            if state.desc is not None:
                strout.append("\nState Name: " + state.desc)
            strout.append("\nInitial probability: " + str(state.pi) + "\n")

            weight = ""
            mue = ""
            u = ""

            weight += str(state.c[ 0])
            mue += str(state.getMean(0))
            u += str(state.getStdDev(0))

            strout.append("  mean: " + str(mue) + "\n")
            strout.append("  variance: " + str(u) + "\n")
            strout.append("  fix: " + str(state.fix) + "\n")

            for c in range(self.cmodel.cos):
                strout.append("\n  Class : " + str(c))
                strout.append("\n    Outgoing transitions:")
                for i, a in enumerate(state.out_a):
                    strout.append("\n      transition to state " + str(i) + " with probability = " + str(state.getOutProb(i, c)))
                strout.append("\n    Ingoing transitions:")
                for i, _ in enumerate(state.in_a):
                    strout.append("\n      transition from state " + str(i) + " with probability = " + str(state.getInProb(i, c)))

        return join(strout, '')

    def forward(self, emissionSequence):
        """

        Result: the (N x T)-matrix containing the forward-variables
        and the scaling vector
        """
        if not isinstance(emissionSequence, EmissionSequence):
            Log.error("EmissionSequence required, got " + str(emissionSequence.__class__.__name__))

        i = self.cmodel.N

        t = emissionSequence.cseq.getLength(0)
        calpha = wrapper.double_matrix_alloc(t, i)
        cscale = wrapper.double_array_alloc(t)

        seq = emissionSequence.cseq.getSequence(0)

        logp = self.cmodel.forward(seq, t, None, calpha, cscale)

        # translate alpha / scale to python lists
        pyscale = wrapper.double_array2list(cscale, t) # XXX return Python2.5 arrays???
        pyalpha = ghmmhelper.double_matrix2list(calpha, t, i) # XXX return Python2.5 arrays? Also
        # XXX Check Matrix-valued input.
        return (pyalpha, pyscale)

    def backward(self, emissionSequence, scalingVector):
        """

        Result: the (N x T)-matrix containing the backward-variables
        """
        if not isinstance(emissionSequence, EmissionSequence):
            Log.error("EmissionSequence required, got " + str(emissionSequence.__class__.__name__))

        seq = emissionSequence.cseq.getSequence(0)

        # parsing 'scalingVector' to C double array.
        cscale = wrapper.list2double_array(scalingVector)

        # alllocating beta matrix
        t = emissionSequence.cseq.getLength(0)
        cbeta = wrapper.double_matrix_alloc(t, self.cmodel.N)

        error = self.cmodel.backward(seq, t, None, cbeta, cscale)
        if error == -1:
            Log.error("backward finished with -1: EmissionSequence cannot be build.")

        pybeta = ghmmhelper.double_matrix2list(cbeta, t, self.cmodel.N)
        return pybeta

    def loglikelihoods(self, emissionSequences):
        """ Compute a vector ( Math.log( P[s| model]) )_{s} of log-likelihoods of the
        individual emissionSequences using the forward algorithm.

        @param emissionSequences SequenceSet

        Result: Math.log( P[emissionSequences| model]) of type float
        (numarray) vector of floats

        """
        emissionSequences = emissionSequences.asSequenceSet()
        seqNumber = len(emissionSequences)

        if self.cmodel.cos > 1:
            Log.note("self.cmodel.cos = " + str(self.cmodel.cos))
            assert self.cmodel.class_change is not None, "Error: class_change not initialized."

        likelihoodList = []

        for i in range(seqNumber):
            seq = emissionSequences.cseq.getSequence(i)
            tmp = emissionSequences.cseq.getLength(i)

            if self.cmodel.cos > 1:
                self.cmodel.class_change.k = i

            try:
                likelihood = self.cmodel.logp(seq, tmp)
                likelihoodList.append(likelihood)
            except Exception, e:
                Log.warning("forward returned -1: Sequence " + str(i) + " cannot be build.", e)
                # XXX TODO: Eventually this should trickle down to C-level
                # Returning -DBL_MIN instead of infinity is stupid, since the latter allows
                # to continue further computations with that inf, which causes
                # things to blow up later.
                # cmodel.logp() could do without a return value if -Inf is returned
                # What should be the semantics in case of computing the likelihood of
                # a set of sequences?
                likelihoodList.append(-float('Inf'))

        # resetting class_change->k to default
        if self.cmodel.cos > 1:
            self.cmodel.class_change.k = -1

        return likelihoodList


    def viterbi(self, emissionSequences):
        """ Compute the Viterbi-path for each sequence in emissionSequences

        @param emissionSequences can either be a SequenceSet or an
        EmissionSequence

        Result: [q_0, ..., q_T] the viterbi-path of emission_sequences is an
        EmmissionSequence object,
        [[q_0^0, ..., q_T^0], ..., [q_0^k, ..., q_T^k]} for a k-sequence
        SequenceSet
        """
        emissionSequences = emissionSequences.asSequenceSet()
        seqNumber = len(emissionSequences)

        if self.cmodel.cos > 1:
            Log.note("self.cmodel.cos = " + str(self.cmodel.cos))
            assert self.cmodel.class_change is not None, "Error: class_change not initialized."

        allLogs = []
        allPaths = []
        for i in range(seqNumber):
            if self.cmodel.cos > 1:
                # if emissionSequence is a sequenceSet with multiple sequences,
                # use sequence index as class_change.k
                self.cmodel.class_change.k = i

            seq = emissionSequences.cseq.getSequence(i)
            seq_len = emissionSequences.cseq.getLength(i)

            viterbiPath, log_p = ghmm_cmodel_viterbi(self.cmodel, seq, seq_len)

            if viterbiPath != None:
                onePath = wrapper.int_array2list(viterbiPath, seq_len)
            else:
                onePath = []

            allPaths.append(onePath)
            allLogs.append(log_p)

        # resetting class_change->k to default
        if self.cmodel.cos > 1:
            self.cmodel.class_change.k = -1

        if emissionSequences.cseq.seq_number > 1:
            return (allPaths, allLogs)
        else:
            return (allPaths[0], allLogs[0])

    def baumWelch(self, trainingSequences, nrSteps=wrapper.MAX_ITER_BW, loglikelihoodCutoff=wrapper.EPS_ITER_BW):
        """ Reestimate the model parameters given the training_sequences.

        Perform at most nr_steps until the improvement in likelihood
        is below likelihood_cutoff

        @param trainingSequences can either be a SequenceSet or a Sequence
        @param nrSteps the maximal number of BW-steps
        @param loglikelihoodCutoff the least relative improvement in likelihood
        with respect to the last iteration required to continue.

        Result: Final loglikelihood
        """

        if not isinstance(trainingSequences, SequenceSet) and not isinstance(trainingSequences, EmissionSequence):
            Log.error("baumWelch requires a SequenceSet or EmissionSequence object.")

        if not self.emissionDomain.CDataType == "double":
            Log.error("Continuous sequence needed.")

        self.baumWelchSetup(trainingSequences, nrSteps, loglikelihoodCutoff)
        ghmm_cmodel_baum_welch(self.BWcontext)
        likelihood = self.BWcontext.logp
        #(steps_made, loglikelihood_array, scale_array) = self.baumWelchStep(nrSteps,
        #                                                                    loglikelihoodCutoff)
        self.baumWelchDelete()

        return likelihood

    def baumWelchSetup(self, trainingSequences, nrSteps, loglikelihoodCutoff=wrapper.EPS_ITER_BW):
        """ Setup necessary temporary variables for Baum-Welch-reestimation.

        Use with baumWelchStep for more control over the training, computing
        diagnostics or doing noise-insertion

        @param trainingSequences can either be a SequenceSet or a Sequence
        @param nrSteps the maximal number of BW-steps
        @param loglikelihoodCutoff the least relative improvement in likelihood
        with respect to the last iteration required to continue.
        """
        self.BWcontext = wrapper.ghmm_cmodel_baum_welch_context(self.cmodel, trainingSequences.cseq)
        self.BWcontext.eps = loglikelihoodCutoff
        self.BWcontext.max_iter = nrSteps


    def baumWelchStep(self, nrSteps, loglikelihoodCutoff):
        """
        Compute one iteration of Baum Welch estimation.

        Use with baumWelchSetup for more control over the training, computing
        diagnostics or doing noise-insertion
        """
        # XXX Implement me
        raise NotImplementedError

    def baumWelchDelete(self):
        """
        Delete the necessary temporary variables for Baum-Welch-reestimation
        """
        self.BWcontext = None

    def asMatrices(self):
        "Return the parameters in matrix form."
        A = []
        B = []
        pi = []
        for i in range(self.cmodel.N):
            A.append([0.0] * self.N)
            B.append([0.0] * 2)
            state = self.cmodel.s[i]
            pi.append(state.pi)

            B[i][0] = state.getMean(0)
            B[i][1] = state.getStdDev(0)

            for j, _ in enumerate(state.out_a[0]):
                A[i][j] = state.out_a[0][j]

        return [A, B, pi]


# XXX - this class will taken over by ContinuousMixtureHMM
class GaussianMixtureHMM(GaussianEmissionHMM):
    """ HMMs with mixtures of Gaussians as emissions.

    Optional features:
    - fixing mixture components in training

    """

    def getEmission(self, i, comp):
        """
        @returns (mu, sigma^2, weight) of component 'comp' in state 'i'
        """
        i = self.state(i)
        state = self.cmodel.s[i]
        mu = state.getMean(comp)
        sigma = state.getStdDev(comp)
        weigth = state.getWeight(comp)
        return (mu, sigma, weigth)

    def setEmission(self, i, comp, values):
        """ Set the emission distribution parameters for a single component in a single state.

        @param i index of a state
        @param comp index of a mixture component
        @param values tuple of mu, sigma, weight
        """
        mu, sigma, weight = values
        i = self.state(i)

        state = self.cmodel.s[i]
        state.setMean(comp, float(mu))  # GHMM C is german: mue instead of mu
        state.setStdDev(comp, float(sigma))
        state.setWeight(comp, float(weight))

    def getMixtureFix(self, state):
        state = self.state(state)
        s = self.cmodel.s[state]
        mixfix = []
        for i in range(s.M):
            emission = s.getEmission(i)
            mixfix.append(emission.fixed)
        return mixfix

    def setMixtureFix(self, state, flags):
        state = self.state(state)
        s = self.cmodel.s[state]
        for i in range(s.M):
            emission = s.getEmission(i)
            emission.fixed = flags[i]

    def __str__(self):
        hmm = self.cmodel
        strout = [str(self.__class__.__name__)]
        if self.cmodel.name:
            strout.append(" " + str(self.cmodel.name))
        strout.append("(N=" + str(hmm.N) + ')\n')

        f = lambda x: "%.2f" % (x,)  # float rounding function

        if hmm.N <= 4:
            iter_list = range(self.N)
        else:
            iter_list = [0, 1, 'X', hmm.N - 2, hmm.N - 1]

        for k in iter_list:
            if k == 'X':
                strout.append('\n  ...\n\n')
                continue

            state = hmm.s[k]
            strout.append("  state " + str(k) + " (")
            strout.append("initial=" + f(state.pi))
            if self.cmodel.cos > 1:
                strout.append(', cos=' + str(self.cmodel.cos))
            strout.append(')\n')

            weight = ""
            mue = ""
            u = ""

            for outp in range(state.M):
                emission = state.getEmission(outp)
                weight += str(state.c[ outp]) + ", "
                mue += str(emission.mean) + ", "
                u += str(emission.variance) + ", "

            strout.append("    Emissions (")
            strout.append("weights=" + str(weight) + ", ")
            strout.append("mu=" + str(mue) + ", ")
            strout.append("sigma=" + str(u) + ")\n")

            strout.append("    Transitions: ")
            if self.cmodel.cos > 1:
                strout.append("\n")

            for c in range(self.cmodel.cos):
                if self.cmodel.cos > 1:
                    strout.append('      class: ' + str(c) + ':')
                strout.append(', '.join(['->' + str(i) + ' (' + str(state.getOutProb(i, c)) + ')' for i, _ in enumerate(state.out_a)]))
                strout.append('\n')

        return join(strout, '')


    def verboseStr(self):
        "defines string representation"
        hmm = self.cmodel

        strout = ["\nOverview of HMM:"]
        strout.append("\nNumber of states: " + str(hmm.N))
        strout.append("\nNumber of mixture components: " + str(hmm.M))

        for k in range(hmm.N):
            state = hmm.s[k]
            strout.append("\n\nState number " + str(k) + ":")
            if state.desc is not None:
                strout.append("\nState Name: " + state.desc)
            strout.append("\nInitial probability: " + str(state.pi))
            strout.append("\n" + str(state.M) + " mixture component(s):\n")

            weight = ""
            mue = ""
            u = ""

            for outp in range(state.M):
                emission = state.getEmission(outp)
                weight += str(state.c[ outp]) + ", "
                mue += str(emission.mean) + ", "
                u += str(emission.variance) + ", "

            strout.append("  pdf component weights : " + str(weight) + "\n")
            strout.append("  mean vector: " + str(mue) + "\n")
            strout.append("  variance vector: " + str(u) + "\n")

            for c in range(self.cmodel.cos):
                strout.append("\n  Class : " + str(c))
                strout.append("\n    Outgoing transitions:")
                for i, _ in enumerate(state.out_a):
                    strout.append("\n      transition to state " + str(i) + " with probability = " + str(state.getOutProb(i, c)))
                strout.append("\n    Ingoing transitions:")
                for i, _ in enumerate(state.in_a):
                    strout.append("\n      transition from state " + str(i) + " with probability = " + str(state.getInProb(i, c)))

            strout.append("\nint fix:" + str(state.fix) + "\n")
        return join(strout, '')


    def asMatrices(self):
        "Return the parameters in matrix form."
        A = []
        B = []
        pi = []
        for i in range(self.cmodel.N):
            A.append([0.0] * self.N)
            B.append([])
            state = self.cmodel.s[i]
            pi.append(state.pi)

            mulist = []
            siglist = []
            for j in range(state.M):
                emission = state.getEmission(j)
                mulist.append(emission.mean)
                siglist.append(emission.variance)

            B[i].append(mulist)
            B[i].append(siglist)
            B[i].append(wrapper.double_array2list(state.c, state.M))

            for j, _ in enumerate(state.out_a[0]):
                A[i][j] = state.out_a[0][j]

        return [A, B, pi]


class ContinuousMixtureHMM(GaussianMixtureHMM):
    """ HMMs with mixtures of any univariate (one dimensional) Continuous
    Distributions as emissions.

    Optional features:
    - fixing mixture components in training
    """

    def getEmission(self, i, comp):
        """
        @returns the paramenters of component 'comp' in state 'i'
        - (type, mu,  sigma^2, weight)        - for a gaussian component
        - (type, mu,  sigma^2, min,   weight) - for a right tail gaussian
        - (type, mu,  sigma^2, max,   weight) - for a left  tail gaussian
        - (type, max, mix,     weight)        - for a uniform
        """
        i = self.state(i)
        state = self.cmodel.s[i]
        emission = state.e[comp]
        if isinstance(emission, NormalRight):
            return NormalRight, emission.mean, emission.variance, emission.minimum, state.getWeight(comp)
        elif isinstance(emission, UniformDistribution):
            return UniformDistribution, emission.start, emission.end, state.getWeight(comp)
        elif isinstance(emission, NormalLeft):
            return NormalLeft, emission.mean, emission.variance, emission.maximum, state.getWeight(comp)
        elif isinstance(emission, NormalDistribution):
            return NormalDistribution, emission.mean, emission.variance, state.getWeight(comp)

    def setEmission(self, i, comp, distType, values):
        """ Set the emission distribution parameters for a mixture component
        of a single state.

        @param i index of a state
        @param comp index of a mixture component
        @param distType type of the distribution
        @param values tuple (mu, sigma, a , weight) and is interpreted depending
        on distType
        - mu     - mean for normal, normal_approx, normal_right, normal_left
        - mu     - max for uniform
        - sigma  - standard deviation for normal, normal_approx, normal_right,
          normal_left
        - sigma  - min for uniform
        - a      - cut-off normal_right and normal_left
        - weight - always component weight
        """

        mu, sigma, a, weight = values
        i = self.state(i)

        state = self.cmodel.s[i]
        state.setWeight(comp, weight)
        state.e[comp] = distType(*values)

    def __str__(self):
        """ defines string representation """
        return "<ContinuousMixtureHMM with " + str(self.cmodel.N) + " states>"

    def verboseStr(self):
        """ Human readable model description """
        hmm = self.cmodel

        strout = ["\nOverview of HMM:"]
        strout.append("\nNumber of states: " + str(hmm.N))
        strout.append("\nMaximum number of output distributions per state: " + str(hmm.M))

        for k in range(hmm.N):
            state = hmm.s[k]
            strout.append("\n\nState number " + str(k) + ":")
            if state.desc is not None:
                strout.append("\nState Name: " + state.desc)
            strout.append("\n  Initial probability: " + str(state.pi))
            strout.append("\n  " + str(state.M) + " density function(s):")

            for outp in range(state.M):
                comp_str = "\n    " + str(state.getWeight(outp)) + " * "
                emission = state.getEmission(outp)
                if isinstance(emission, NormalRight):
                    comp_str += "normal right tail(mean = " + str(emission.mean)
                    comp_str += ", variance = " + str(emission.variance)
                    comp_str += ", minimum = " + str(emission.minimum) + ")"
                elif isinstance(emission, NormalLeft):
                    comp_str += "normal left tail(mean = " + str(emission.mean)
                    comp_str += ", variance = " + str(emission.variance)
                    comp_str += ", maximum = " + str(emission.maximum) + ")"
                elif isinstance(emission, UniformDistribution):
                    comp_str += "uniform(minimum = " + str(emission.start)
                    comp_str += ", maximum = " + str(emission.end) + ")"
                elif isinstance(emission, NormalDistribution):
                    comp_str += "normal(mean = " + str(emission.mean)
                    comp_str += ", variance = " + str(emission.variance) + ")"

                strout.append(comp_str)

            for c in range(self.cmodel.cos):
                strout.append("\n  Class : " + str(c))
                strout.append("\n    Outgoing transitions:")
                for i, _ in enumerate(state.out_a):
                    strout.append("\n      transition to state " + str(i) + " with probability = " + str(state.getOutProb(i, c)))

                strout.append("\n    Ingoing transitions:")
                for i, _ in enumerate(state.in_a):
                    strout.append("\n      transition from state " + str(i) + " with probability = " + str(state.getInProb(i, c)))

            strout.append("\n  int fix:" + str(state.fix))

        strout.append("\n")
        return join(strout, '')

    def asMatrices(self):
        """Return the parameters in matrix form.
           It also returns the density type"""
        # XXX inherit transitions ????

        A = []
        B = []
        pi = []
        d = []
        for i in range(self.cmodel.N):
            A.append([0.0] * self.N)
            B.append([])
            state = self.cmodel.s[i]
            pi.append(state.pi)
            denList = []

            parlist = []
            for j in range(state.M):
                emission = state.getEmission(j)
                denList.append(emission.__class__)
                if isinstance(emission, NormalRight):
                    parlist.append([emission.mean, emission.variance, emission.minimum, state.getWeight(j)])
                elif isinstance(emission, NormalLeft):
                    parlist.append([emission.mean, emission.variance, emission.maximum, state.getWeight(j)])
                elif isinstance(emission, NormalDistribution):
                    parlist.append([emission.mean, emission.variance, 0, state.getWeight(j)])
                elif isinstance(emission, UniformDistribution):
                    parlist.append([emission.start, emission.end, 0, state.getWeight(j)])
                else:
                    Log.error("Unsupported distribution" + str(emission.type))

            for j in range(4):
                B[i].append([l[j] for l in parlist])

            d.append(denList)

            for j, _ in enumerate(state.out_a[0]):
                A[i][j] = state.out_a[0][j]

        return [A, B, pi, d]


class MultivariateGaussianMixtureHMM(GaussianEmissionHMM):
    """ HMMs with Multivariate Gaussian distribution as emissions.

    States can have multiple mixture components.
    """

    def __init__(self, emissionDomain, distribution, cmodel):
        HMM.__init__(self, emissionDomain, distribution, cmodel)

        # Baum Welch context, call baumWelchSetup to initalize
        self.BWcontext = ""

    def getEmission(self, i, m):
        """
        @returns mean and covariance matrix of component m in state i
        """
        i = self.state(i)
        state = self.cmodel.s[i]
        assert 0 <= m < state.M, "Index " + str(m) + " out of bounds."

        emission = state.e[m]
        mu = wrapper.double_array2list(emission.mean, emission.dimension)
        sigma = wrapper.double_array2list(emission.variance, emission.dimension * emission.dimension)
        return (mu, sigma)

    def setEmission(self, i, m, values):
        """ Set the emission distributionParameters for mixture component m in
        state i

        @param i index of a state
        @param m index of a mixture component
        @param values tuple of mu, sigma
        """

        mu, sigma = values
        i = self.state(i)

        self.cmodel.s[i].e[m] = MultiNormalDistribution(mu, sigma)


    def __str__(self):
        hmm = self.cmodel
        strout = ["\nHMM Overview:"]
        strout.append("\nNumber of states: " + str(hmm.N))
        strout.append("\nmaximum Number of mixture components: " + str(hmm.M))
        strout.append("\nNumber of dimensions: " + str(hmm.dim))

        for k in range(hmm.N):
            state = hmm.s[k]
            strout.append("\n\nState number " + str(k) + ":")
            strout.append("\nInitial probability: " + str(state.pi))
            strout.append("\nNumber of mixture components: " + str(state.M))

            for m in range(state.M):
                strout.append("\n\n  Emission number " + str(m) + ":")
                emission = state.getEmission(m)

                strout.append("\n    emission type: " + emission.__class__.__name__)
                strout.append("\n    emission weight: " + str(state.c[m]))
                strout.append("\n    mean: " + str(emission.mean))
                strout.append("\n    covariance matrix: " + str(emission.variance))
                strout.append("\n    inverse of covariance matrix: " + str(emission.variance_inv))
                strout.append("\n    determinant of covariance matrix: " + str(emission.variance_det))
                strout.append("\n    cholesky decomposition of covariance matrix: " + str(emission.sigmacd))
                strout.append("\n    fix: " + str(state.fix))

            for c in range(self.cmodel.cos):
                strout.append("\n\n  Class : " + str(c))
                strout.append("\n    Outgoing transitions:")
                for i, _ in enumerate(state.out_a):
                    strout.append("\n      transition to state " + str(i) + " with probability = " + str(state.getOutProb(i, c)))
                strout.append("\n    Ingoing transitions:")
                for i, a in enumerate(state.in_a):
                    strout.append("\n      transition from state " + str(i) + " with probability = " + str(state.getInProb(i, c)))

        return join(strout, '')

    def asMatrices(self):
        "Return the parameters in matrix form."
        A = []
        B = []
        pi = []
        for i in range(self.cmodel.N):
            A.append([0.0] * self.N)
            emissionparams = []
            state = self.cmodel.s[i]
            pi.append(state.pi)
            for m in range(state.M):
                emission = state.getEmission(m)
                mu = wrapper.double_array2list(emission.mean, emission.dimension)
                sigma = wrapper.double_array2list(emission.variance, (emission.dimension * emission.dimension))
                emissionparams.append(mu)
                emissionparams.append(sigma)

            if state.M > 1:
                weights = wrapper.double_array2list(state.c, state.M)
                emissionparams.append(weights)

            B.append(emissionparams)

            for j, _ in enumerate(state.out_a[0]):
                A[i][j] = state.out_a[0][j]

        return [A, B, pi]


def HMMDiscriminativeTraining(HMMList, SeqList, nrSteps=50, gradient=0):
    """ Trains a couple of HMMs to increase the probablistic distance
    if the the HMMs are used as classifier.

    @param HMMList List of labeled HMMs
    @param SeqList List of labeled sequences, one for each HMM
    @param nrSteps maximal number of iterations
    @param gradient @todo document me

    @note this method does a initial expectation maximization training
    """

    if len(HMMList) != len(SeqList):
        Log.error('Input list are not equally long')

    if not isinstance(HMMList[0], StateLabelHMM):
        Log.error('Input is not a StateLabelHMM')

    if not SeqList[0].hasStateLabels:
        Log.error('Input sequence has no labels')

    inplen = len(HMMList)
    if gradient not in [0, 1]:
        Log.error("TrainingType " + gradient + " not supported.")

    for i in range(inplen):
        if HMMList[i].emissionDomain.CDataType == "double":
            Log.error('discriminative training is at the moment only implemented on discrete HMMs')
            #initial training with Baum-Welch
        HMMList[i].baumWelch(SeqList[i], 3, 1e-9)

    HMMArray = wrapper.dmodel_ptr_array_alloc(inplen)
    SeqArray = wrapper.sequences_ptr_array_alloc(inplen)

    for i in range(inplen):
        HMMArray[i] = HMMList[i].cmodel
        SeqArray[i] = SeqList[i].cseq

    wrapper.ghmm_dmodel_label_discriminative(HMMArray, SeqArray, inplen, nrSteps, gradient)

    for i in range(inplen):
        HMMList[i].cmodel = HMMArray[i]
        SeqList[i].cseq = SeqArray[i]

    return HMMDiscriminativePerformance(HMMList, SeqList)


def HMMDiscriminativePerformance(HMMList, SeqList):
    """ Computes the discriminative performce of the HMMs in HMMList
    under the sequences in SeqList
    """

    if len(HMMList) != len(SeqList):
        Log.error('Input list are not equally long')

    if not isinstance(HMMList[0], StateLabelHMM):
        Log.error('Input is not a StateLabelHMM')

    if not SeqList[0].hasStateLabels:
        Log.error('Input sequence has no labels')

    inplen = len(HMMList)

    single = [0.0] * inplen

    HMMArray = wrapper.dmodel_ptr_array_alloc(inplen)
    SeqArray = wrapper.dseq_ptr_array_alloc(inplen)

    for i in range(inplen):
        wrapper.dmodel_ptr_array_setitem(HMMArray, i, HMMList[i].cmodel)
        wrapper.dseq_ptr_array_setitem(SeqArray, i, SeqList[i].cseq)

    retval = wrapper.ghmm_dmodel_label_discrim_perf(HMMArray, SeqArray, inplen)
    return retval

########## Here comes all the Pair HMM stuff ##########
class DiscretePairDistribution(DiscreteDistribution):
    """
    A DiscreteDistribution over TWO Alphabets: The discrete distribution
    is parameterized by the vector of probabilities.
    To get the index of the vector that corresponds to a pair of characters
    use the getPairIndex method.

    """

    def __init__(self, alphabetX, alphabetY, offsetX, offsetY):
        """
        construct a new DiscretePairDistribution
        @param alphabetX Alphabet object for sequence X
        @param alphabetY Alphabet object for sequence Y
        @param offsetX number of characters the alphabet of sequence X
        consumes at a time
        @param offsetY number of characters the alphabet of sequence Y
        consumes at a time
        """
        self.alphabetX = alphabetX
        self.alphabetY = alphabetY
        self.offsetX = offsetX
        self.offsetY = offsetY
        self.prob_vector = None
        self.pairIndexFunction = wrapper.ghmm_dpmodel_pair

    def getPairIndex(self, charX, charY):
        """
        get the index of a pair of two characters in the probability vector
        (if you use the int representation both values must be ints)
        @param charX character chain or int representation
        @param charY character chain or int representation
        @return the index of the pair in the probability vector
        """
        if not (type(charX) == type(1) and type(charY) == type(1)):
            if charX == "-":
                intX = 0 # check this!
            else:
                intX = self.alphabetX.internal(charX)
            if charY == "-":
                intY = 0 # check this!
            else:
                intY = self.alphabetY.internal(charY)
        else:
            intX = charX
            intY = charY
        return self.pairIndexFunction(intX, intY,
            len(self.alphabetX),
            self.offsetX, self.offsetY)

    def setPairProbability(self, charX, charY, probability):
        """
        set the probability of the [air charX and charY to probability
        @param charX character chain or int representation
        @param charY character chain or int representation
        @param probability probability (0<=float<=1)
        """
        self.prob_vector[self.getPairIndex(charX, charY)] = probability

    def getEmptyProbabilityVector(self):
        """
        get an empty probability vector for this distribution (filled with 0.0)
        @return list of floats
        """
        length = self.pairIndexFunction(len(self.alphabetX) - 1,
            len(self.alphabetY) - 1,
            len(self.alphabetX),
            self.offsetX, self.offsetY) + 1
        return [0.0 for i in range(length)]

    def getCounts(self, sequenceX, sequenceY):
        """
        extract the pair counts for aligned sequences sequenceX and sequenceY
        @param sequenceX string for sequence X
        @param sequenceY strinf for sequence Y
        @return a list of counts
        """
        counts = self.getEmptyProbabilityVector()
        if self.offsetX != 0 and self.offsetY != 0:
            assert len(sequenceX) / self.offsetX == len(sequenceY) / self.offsetY
            for i in range(len(sequenceX) / self.offsetX):
                charX = sequenceX[i * self.offsetX:(i + 1) * self.offsetX]
                charY = sequenceY[i * self.offsetY:(i + 1) * self.offsetY]
                counts[self.getPairIndex(charX, charY)] += 1
            return counts
        elif self.offsetX == 0 and self.offsetY == 0:
            Log.error("Silent states (offsetX==0 and offsetY==0) not supported")
            return counts
        elif self.offsetX == 0:
            charX = "-"
            for i in range(len(sequenceY) / self.offsetY):
                charY = sequenceY[i * self.offsetY:(i + 1) * self.offsetY]
                counts[self.getPairIndex(charX, charY)] += 1
            return counts
        elif self.offsetY == 0:
            charY = "-"
            for i in range(len(sequenceX) / self.offsetX):
                charX = sequenceX[i * self.offsetX:(i + 1) * self.offsetX]
                counts[self.getPairIndex(charX, charY)] += 1
            return counts


class PairHMM(HMM):
    """
    Pair HMMs with discrete emissions over multiple alphabets.
    Optional features: continuous values for transition classes
    """

    def __init__(self, emissionDomains, distribution, cmodel):
        """
        create a new PairHMM object (this should only be done using the
        factory: e.g model = PairHMMOpenXML(modelfile) )
        @param emissionDomains list of EmissionDomain objects
        @param distribution (not used) inherited from HMM
        @param cmodel a swig pointer on the underlying C structure
        """
        HMM.__init__(self, emissionDomains[0], distribution, cmodel)
        self.emissionDomains = emissionDomains
        self.alphabetSizes = []
        for domain in self.emissionDomains:
            if isinstance(domain, Alphabet):
                self.alphabetSizes.append(len(domain))

        self.maxSize = 10000
        self.model_type = self.cmodel.model_type  # model type
        self.background = None

        self.states = {}

    def __str__(self):
        """
        string representation (more for debuging) shows the contents of the C
        structure ghmm_dpmodel
        @return string representation
        """
        return "<PairHMM with " + str(self.cmodel.N) + " states>"

    def verboseStr(self):
        """
        string representation (more for debuging) shows the contents of the C
        structure ghmm_dpmodel
        @return string representation
        """
        hmm = self.cmodel
        strout = ["\nGHMM Model\n"]
        strout.append("Name: " + str(self.cmodel.name))
        strout.append("\nModelflags: " + self.printtypes(self.cmodel.model_type))
        strout.append("\nNumber of states: " + str(hmm.N))
        strout.append("\nSize of Alphabet: " + str(hmm.M))
        for k in range(hmm.N):
            state = hmm.s[k]
            strout.append("\n\nState number " + str(k) + ":")
            if state.desc is not None:
                strout.append("\nState Name: " + state.desc)
            strout.append("\nInitial probability: " + str(state.pi))
            strout.append("\nOutput probabilites: ")
            #strout.append(str(state.b[outp]))
            strout.append("\n")

            strout.append("\nOutgoing transitions:")
            for i, a in enumerate(state.out_a):
                strout.append("\ntransition to state " + str(i) + " with probability " + str(a))
            strout.append("\nIngoing transitions:")
            for i, a in enumerate(state.a):
                strout.append("\ntransition from state " + str(i) + " with probability " + str(a))
                strout.append("\nint fix:" + str(state.fix) + "\n")

        if hmm.model_type & kSilentStates:
            strout.append("\nSilent states: \n")
            for k in range(hmm.N):
                strout.append(str(hmm.silent[k]) + ", ")
            strout.append("\n")

        return join(strout, '')

    def viterbi(self, complexEmissionSequenceX, complexEmissionSequenceY):
        """
        run the naive implementation of the Viterbi algorithm and
        return the viterbi path and the log probability of the path
        @param complexEmissionSequenceX sequence X encoded as ComplexEmissionSequence
        @param complexEmissionSequenceY sequence Y encoded as ComplexEmissionSequence
        @return (path, log_p)
        """
        # get a pointer on a double and a int to get return values by reference
        log_p_ptr = wrapper.double_array_alloc(1)
        length_ptr = wrapper.int_array_alloc(1)
        # call log_p and length will be passed by reference
        cpath = self.cmodel.viterbi(complexEmissionSequenceX.cseq,
            complexEmissionSequenceY.cseq,
            log_p_ptr, length_ptr)
        # get the values from the pointers
        log_p = log_p_ptr[ 0]
        length = length_ptr[0]
        path = [cpath[x] for x in range(length)]
        return path, log_p

    def viterbiPropagate(self, complexEmissionSequenceX, complexEmissionSequenceY, startX=None, startY=None, stopX=None, stopY=None, startState=None, startLogp=None, stopState=None, stopLogp=None):
        """
        run the linear space implementation of the Viterbi algorithm and
        return the viterbi path and the log probability of the path
        @param complexEmissionSequenceX sequence X encoded as ComplexEmissionSequence
        @param complexEmissionSequenceY sequence Y encoded as ComplexEmissionSequence
        Optional parameters to run the algorithm only on a segment:
        @param startX start index in X
        @param startY start index in Y
        @param stopX stop index in X
        @param stopY stop index in Y
        @param startState start the path in this state
        @param stopState path ends in this state
        @param startLogp initialize the start state with this log probability
        @param stopLogp if known this is the logp of the partial path
        @return (path, log_p)
        """
        # get a pointer on a double and a int to get return values by reference
        log_p_ptr = wrapper.double_array_alloc(1)
        length_ptr = wrapper.int_array_alloc(1)
        # call log_p and length will be passed by reference
        if not (startX and startY and stopX and stopY and startState and stopState and startLogp):
            cpath = self.cmodel.viterbi_propagate(
                complexEmissionSequenceX.cseq,
                complexEmissionSequenceY.cseq,
                log_p_ptr, length_ptr,
                self.maxSize)
        else:
            if stopLogp == None:
                stopLogp = 0
            cpath = self.cmodel.viterbi_propagate_segment(
                complexEmissionSequenceX.cseq,
                complexEmissionSequenceY.cseq,
                log_p_ptr, length_ptr, self.maxSize,
                startX, startY, stopX, stopY, startState, stopState,
                startLogp, stopLogp)

        # get the values from the pointers
        log_p = log_p_ptr[ 0]
        length = length_ptr[0]
        path = [cpath[x] for x in range(length)]
        return (path, log_p)

    def logP(self, complexEmissionSequenceX, complexEmissionSequenceY, path):
        """
        compute the log probability of two sequences X and Y and a path
        @param complexEmissionSequenceX sequence X encoded as
        ComplexEmissionSequence
        @param complexEmissionSequenceY sequence Y encoded as
        ComplexEmissionSequence
        @param path the state path
        @return log probability
        """
        cpath = wrapper.list2int_array(path)
        logP = self.cmodel.viterbi_logP(complexEmissionSequenceX.cseq,
            complexEmissionSequenceY.cseq,
            cpath, len(path))
        return logP

    def addEmissionDomains(self, emissionDomains):
        """
        add additional EmissionDomains that are not specified in the XML file.
        This is used to add information for the transition classes.
        @param emissionDomains a list of EmissionDomain objects
        """
        self.emissionDomains.extend(emissionDomains)
        discreteDomains = []
        continuousDomains = []
        for i in range(len(emissionDomains)):
            if emissionDomains[i].CDataType == "int":
                discreteDomains.append(emissionDomains[i])
                self.alphabetSizes.append(len(emissionDomains[i]))
            if emissionDomains[i].CDataType == "double":
                continuousDomains.append(emissionDomains[i])

        self.cmodel.number_of_alphabets += len(discreteDomains)
        self.cmodel.size_of_alphabet = wrapper.list2int_array(self.alphabetSizes)

        self.cmodel.number_of_d_seqs += len(continuousDomains)

    def checkEmissions(self, eps=0.0000000000001):
        """
        checks the sum of emission probabilities in all states
        @param eps precision (if the sum is > 1 - eps it passes)
        @return 1 if the emission of all states sum to one, 0 otherwise
        """
        allok = 1
        for state in self.states:
            emissionSum = sum(state.emissions)
            if abs(1 - emissionSum) > eps:
                Log.note(("Emissions in state %s (%s) do not sum to 1 (%s)" % (state.id, state.label, emissionSum)))
                allok = 0
        return allok

    def checkTransitions(self, eps=0.0000000000001):
        """
        checks the sum of outgoing transition probabilities for all states
        @param eps precision (if the sum is > 1 - eps it passes)
        @return 1 if the transitions of all states sum to one, 0 otherwise
        """
        allok = 1
        # from build matrices in xmlutil:
        orders = {}
        k = 0 # C style index
        for s in self.states: # ordering from XML
            orders[s.index] = k
            k = k + 1

        for state in self.states:
            for tclass in range(state.kclasses):
                outSum = 0.0
                c_state = self.cmodel.getState(orders[state.index])
                for out, a in enumerate(c_state.out_a):
                    outSum += wrapper.double_matrix_getitem(c_state.out_a, out, tclass)

                if abs(1 - outSum) > eps:
                    Log.note("Outgoing transitions in state %s (%s) do not sum to 1 (%s) for class %s" % (state.id, state.label, outSum, tclass))
                    allok = 0
        return allok
