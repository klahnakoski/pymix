import copy
import math
from pymix import _C_mixextend
import random
import numpy
import sys
from core.pymix_util import mixextend
from ..distributions.prob import ProbDistribution
from ..distributions.product import ProductDistribution
from ..pymix_util.errors import InvalidPosteriorDistribution, ConvergenceFailureEM, InvalidDistributionInput
from ..pymix_util.dataset import DataSet
from ..pymix_util.maths import sumlogs, dict_intersection
from ..pymix_util.stats import entropy, sym_kl_dist, get_posterior


class MixtureModel(ProbDistribution):
    """
    Class for a context-specific independence (CSI) mixture models.
    The components are naive Bayes models (i.e. ProductDistribution objects).
    """

    def __init__(self, G, pi, components, compFix=None, struct=0, identifiable=1):
        """
        Constructor

        @param G: number of components
        @param pi: mixture weights
        @param components: list of ProductDistribution objects, each entry is one component
        @param compFix: list of optional flags for fixing components in the reestimation
                         the following values are supported: 1 distribution parameters are fixed, 2 distribution
                         parameters and mixture coefficients are fixed
        @param struct: Flag for CSI structure, 0 = no CSI structure, 1 = CSI structure
        """
        assert len(pi) == len(components) == G, str(len(pi)) + ', ' + str(len(components)) + ', ' + str(G)
        assert abs((1.0 - sum(pi))) < 1e-12, "sum(pi) = " + str(sum(pi)) + ", " + str(abs((1.0 - sum(pi))))

        self.freeParams = 0
        self.p = components[0].p

        # Internally components must be a list of ProductDistribution objects. In case the input is a list of
        # ProbDistributions we convert components accordingly.
        if isinstance(components[0], ProbDistribution) and not isinstance(components[0], ProductDistribution):
            # make sure all elements of components are ProbDistribution of the same dimension
            for c in components:
                assert isinstance(c, ProbDistribution)
                assert c.p == self.p
            for i, c in enumerate(components):
                components[i] = ProductDistribution([c])

        self.dist_nr = components[0].dist_nr

        # some checks to ensure model validity
        for c in components:
            # components have to be either ProductDistribution objects or a list of univariate ProbDistribution objects
            assert isinstance(c, ProductDistribution), "Got " + str(c.__class__) + " as component."
            assert self.p == c.p, str(self.p) + " != " + str(c.p)
            assert self.dist_nr == c.dist_nr
            self.freeParams += c.freeParams

        self.freeParams += G - 1  # free parameters of the mixture coefficients

        self.G = G  # Number of components
        self.pi = numpy.array(pi, dtype='Float64')  # vector of mixture weights

        # XXX  check numpy capabilities for arrays of objects
        self.components = components   # list of distribution objects

        self.suff_p = None  # dimension of sufficient statistic data

        self.nr_tilt_steps = 10 # number of steps for deterministic annealing in the EM, 10 is default
        self.heat = 0.5  # initial heat parameter for deterministic annealing

        # compFix contains flags for each component, which determine whether the distribution parameter in a
        # component will be skipped in the reestimation,
        if compFix:
            assert len(compFix) == self.G, str(len(compFix)) + " != " + str(self.G)
            self.compFix = compFix
        else:
            self.compFix = [0] * self.G

        # initializing dimensions for sufficient statistics data
        self.update_suff_p()

        self.struct = struct
        if self.struct:
            self.initStructure()

        # flag that determines whether identifiability is enforced in training
        self.identFlag = identifiable

        # error tolerance for the objective function in the parameter training
        self.err_tol = 1e-6

        # minmal mixture coefficient value
        self.min_pi = 0.05 # TEST

    def __eq__(self, other):
        res = False
        if isinstance(other, MixtureModel):
            if numpy.allclose(other.pi, self.pi) and other.G == self.G:
                res = True
                for i in range(self.G):
                    if not (other.components[i] == self.components[i]):
                        #print other.components[i] ,"!=", self.components[i]
                        return False

        return res

    def __copy__(self):
        copy_components = []
        copy_pi = copy.deepcopy(self.pi)
        copy_compFix = copy.deepcopy(self.compFix)
        for i in range(self.G):
            copy_components.append(copy.deepcopy(self.components[i]))

        copy_model = MixtureModel(self.G, copy_pi, copy_components, compFix=copy_compFix)
        copy_model.nr_tilt_steps = self.nr_tilt_steps
        copy_model.suff_p = self.suff_p
        copy.identFlag = self.identFlag

        if self.struct:
            copy_model.initStructure()

            copy_leaders = copy.deepcopy(self.leaders)
            copy_groups = copy.deepcopy(self.groups)

            copy_model.leaders = copy_leaders
            copy_model.groups = copy_groups

        return copy_model

    def __str__(self):
        s = "G = " + str(self.G)
        s += "\np = " + str(self.p)
        s += "\npi =" + str(self.pi) + "\n"
        s += "compFix = " + str(self.compFix) + "\n"
        for i in range(self.G):
            s += "Component " + str(i) + ":\n"
            s += "  " + str(self.components[i]) + "\n"

        if self.struct:
            s += "\nCSI structure:\n"
            s += "leaders:" + str(self.leaders) + "\n"
            s += "groups:" + str(self.groups) + "\n"

        return s

    def initStructure(self):
        """
        Initializes the CSI structure.
        """

        self.struct = 1
        # for a model with group structure we have to check for valid model topology
        if self.struct:
            # ensuring identical number of distributions in all components
            nr = self.components[0].dist_nr
            for i in range(1, self.G):
                assert isinstance(self.components[i], ProductDistribution)
                assert self.components[i].dist_nr == nr
                # checking for consistent dimensionality of elementar distributions among components
                for j in range(nr):
                    assert self.components[i][j].p == self.components[0][j].p
                    assert self.components[i][j].freeParams == self.components[0][j].freeParams

            # if there is already a CSI structure in the model, components within the same group
            # share a reference to the same distribution object. For the new structure we need to make copies.
            if hasattr(self, 'groups'):
                for j in range(nr):
                    for i in self.groups[j]:
                        for r in self.groups[j][i]:
                            self.components[r][j] = copy.copy(self.components[r][j])

            # Variables for model structure.
            # leaders holds for each dimension the indixes of the group representing components. Initially each
            # component is the representative of itself, e.g. there are no groups.
            self.leaders = []

            # groups is a list of dictionaries, one for each dimension p. The group members of a leader are hashed with the
            # leaders index as key. Initially all components are inserted with an empty list.
            self.groups = []

            for i in range(nr):
                self.leaders.append(range(self.G))
                d = {}
                for j in range(self.G):
                    d[j] = []
                self.groups.append(d)

    def modelInitialization(self, data, rtype=1, missing_value=None):
        """
        Perform model initialization given a random assigment of the
        data to the components.

        @param data: DataSet object
        @param rtype: type of random assignments.
        0 = fuzzy assingment
        1 = hard assingment
        @param missing_value: missing symbol to be ignored in parameter estimation (if applicable)

        @return: posterior assigments
        """
        if not isinstance(data, DataSet):
            raise TypeError, "DataSet object required, got" + str(data.__class__)
        else:
            if data.internalData is None:
                data.internalInit(self)

        # reset structure if applicable
        if self.struct:
            self.initStructure()

        l = numpy.zeros((self.G, len(data)), dtype='Float64')
        for i in range(len(data)):
            if rtype == 0:
                for j in range(self.G):
                    l[j, i] = random.uniform(0.1, 1)
                s = sum(l[:, i])
                for j in range(self.G):
                    l[j, i] /= s
            else:
                l[random.randint(0, self.G - 1), i] = 1


        # do one M Step
        fix_pi = 1.0
        unfix_pi = 0.0
        fix_flag = 0   # flag for fixed mixture components
        for i in range(self.G):
            # setting values for pi
            if self.compFix[i] == 2:
                fix_pi -= self.pi[i]
                fix_flag = 1
            else:
                self.pi[i] = l[i, :].sum() / len(data)
                unfix_pi += self.pi[i]

            if self.compFix[i] == 1 or self.compFix[i] == 2:
                # fixed component
                continue
            else:
                # components are product distributions that may contain mixtures
                last_index = 0

                for j in range(self.components[i].dist_nr):
                    if isinstance(self.components[i][j], MixtureModel):
                        dat_j = data.singleFeatureSubset(j)
                        self.components[i][j].modelInitialization(dat_j, rtype=rtype, missing_value=missing_value)
                    else:
                        loc_l = l[i, :]
                        # masking missing values from parameter estimation
                        if data.missingSymbols.has_key(j):
                            ind_miss = data.getMissingIndices(j)
                            for k in ind_miss:
                                loc_l[k] = 0.0

                        self.components[i][j].MStep(loc_l, data.getInternalFeature(j))

        # renormalizing mixing proportions in case of fixed components
        if fix_flag:
            if unfix_pi == 0.0:
            #print "----\n",self,"----\n"
                print "unfix_pi = ", unfix_pi
                print "fix_pi = ", fix_pi
                print "pi = ", self.pi
                raise RuntimeError, "unfix_pi = 0.0"

            for i in range(self.G):
                if self.compFix[i] == 0:
                    self.pi[i] = (self.pi[i] * fix_pi) / unfix_pi
        return l


    def pdf(self, x):
        logp_list = numpy.zeros((self.G, len(x)), dtype='Float64')
        for i in range(self.G):
            if self.pi[i] == 0.0:
                log_pi = float('-inf')
            else:
                log_pi = numpy.log(self.pi[i])
            logp_list[i] = log_pi + self.components[i].pdf(x)

        p = numpy.zeros(len(x), dtype='Float64')
        for j in range(len(x)):
            p[j] = sumlogs(logp_list[:, j])
        return p

    def sample(self):
        sum = 0.0
        p = random.random()
        for k in range(self.G):
            sum += self.pi[k]
            if sum >= p:
                break
        return self.components[k].sample()

    def sampleSet(self, nr):
        ls = []
        for i in range(nr):
            sum = 0.0
            p = random.random()
            for k in range(self.G):
                sum += self.pi[k]
                if sum >= p:
                    break
            ls.append(self.components[k].sample())
        return ls


    def sampleDataSet(self, nr):
        """
        Returns a DataSet object of size 'nr'.

        @param nr: size of DataSet to be sampled

        @return: DataSet object
        """
        ls = self.sampleSet(nr)
        data = DataSet()
        data.dataMatrix = ls
        data.N = nr
        data.p = self.p
        data.sampleIDs = []

        for i in range(data.N):
            data.sampleIDs.append("sample" + str(i))

        for h in range(data.p):
            data.headers.append("X_" + str(h))

        data.internalInit(self)
        return data

    def sampleDataSetLabels(self, nr):
        """
        Samples a DataSet of size 'nr' and returns the DataSet and the true
        component labels

        @param nr: size of DataSet to be sampled

        @return: tuple of DataSet object and list of labels
        """
        [c, ls] = self.sampleSetLabels(nr)

        data = DataSet()
        data.dataMatrix = ls
        data.N = nr
        data.p = self.p
        data.sampleIDs = []

        for i in range(data.N):
            data.sampleIDs.append("sample" + str(i))

        for h in range(data.p):
            data.headers.append("X_" + str(h))

        data.internalInit(self)
        return [data, c]

    def sampleSetLabels(self, nr):
        """ Same as sample but the component labels are returned as well.
            Useful for testing purposes mostly.

        """
        ls = []
        label = []
        for i in range(nr):
            sum = 0.0
            p = random.random()
            for k in range(self.G):
                sum += self.pi[k]
                if sum >= p:
                    break
            label.append(k)
            ls.append(self.components[k].sample())

        return [numpy.array(label), ls]

    def EM(self, data, max_iter, delta, silent=False, mix_pi=None, mix_posterior=None, tilt=0, EStep=None, EStepParam=None):
        """
        Reestimation of mixture parameters using the EM algorithm.

        @param data: DataSet object
        @param max_iter: maximum number of iterations
        @param delta: minimal difference in likelihood between two iterations before
        convergence is assumed.
        @param silent: 0/1 flag, toggles verbose output
        @param mix_pi: [internal use only] necessary for the reestimation of
        mixtures as components
        @param mix_posterior:[internal use only] necessary for the reestimation of
        mixtures as components
        @param tilt: 0/1 flag, toggles the use of a deterministic annealing in the training
        @param EStep: function implementing the EStep, by default self.EStep
        @param EStepParam: additional paramenters for more complex EStep implementations

        @return: tuple of posterior matrix and log-likelihood from the last iteration
        """
        if isinstance(data, numpy.ndarray):
            raise TypeError, "DataSet object required."
        elif isinstance(data, DataSet):
            if data.internalData is None:
                if not silent:
                    sys.stdout.write("Parsing data set...")
                    sys.stdout.flush()
                data.internalInit(self)
                if not silent:
                    sys.stdout.write("done\n")
                    sys.stdout.flush()
        else:
            raise ValueError, "Unknown input type format: " + str(data.__class__)

        log_p_old = -1.0
        step = 0

        if EStep == None:
            EStep = self.EStep

        # if deterministic annealing is activated, increase number of steps by self.nr_tilt_steps
        if tilt:
            if not silent:
                sys.stdout.write("Running EM with " + str(self.nr_tilt_steps) + " steps of deterministic annealing.\n")
            max_iter += self.nr_tilt_steps

        log_p = 0.0
        while 1:
            [log_l, log_p] = EStep(data, mix_posterior, mix_pi, EStepParam)
            if log_p_old != -1.0 and not silent and step != 0:
                if tilt and step <= self.nr_tilt_steps:
                    sys.stdout.write("TILT Step " + str(step) + ": log likelihood: " + str(log_p) + "\n")
                else:
                    sys.stdout.write("Step " + str(step) + ": log likelihood: " + str(log_p_old) + "   (diff=" + str(diff) + ")\n")

            # checking for convergence
            diff = (log_p - log_p_old)

            if diff < 0.0 and step > 1 and abs(diff / log_p_old) > self.err_tol:
                print log_p, log_p_old, diff, step, abs(diff / log_p_old)
                print "WARNING: EM divergent."
                raise ConvergenceFailureEM, "Convergence failed."

            if numpy.isnan(log_p):
                print "WARNING: One sample was not assigned."
                raise ConvergenceFailureEM, "Non assigned element."

            if (not tilt or (tilt and step + 1 >= self.nr_tilt_steps)) and diff >= 0.0 and delta >= abs(diff) and max_iter != 1:
                if not silent:
                    sys.stdout.write("Step " + str(step) + ": log likelihood: " + str(log_p) + "   (diff=" + str(diff) + ")\n")
                    sys.stdout.write("Convergence reached with log_p " + str(log_p) + " after " + str(step) + " steps.\n")
                if self.identFlag:
                    self.identifiable()
                return (log_l, log_p)

            if step == max_iter:
                if not silent:
                    sys.stdout.write("Max_iter " + str(max_iter) + " reached -> stopping\n")
                if self.identFlag:
                    self.identifiable()
                return (log_l, log_p)

            log_p_old = log_p

            # compute posterior likelihood matrix from log posterior
            l = numpy.exp(log_l)

            # deterministic annealing, shifting posterior toward uniform distribution.
            if tilt and step + 1 <= self.nr_tilt_steps and mix_posterior is None:
                h = self.heat - (step * (self.heat / (self.nr_tilt_steps))  )
                for j in range(data.N):
                    uni = 1.0 / self.G
                    tilt_l = (uni - l[:, j]) * h

                    l[:, j] += tilt_l
                    #print l[:,j]

            # variables for component fixing
            fix_pi = 1.0
            unfix_pi = 0.0
            fix_flag = 0   # flag for fixed mixture components

            # update component parameters and mixture weights
            for i in range(self.G):
                if self.compFix[i] == 2:   # pi[i] is fixed
                    fix_pi -= self.pi[i]
                    fix_flag = 1
                    continue

                else:
                    # for mixtures of mixtures we need to multiply in the mix_pi[i]s
                    if mix_pi is not None:
                        self.pi[i] = ( l[i, :].sum() / (data.N * mix_pi))
                    else:
                        self.pi[i] = ( l[i, :].sum() / (data.N ) )

                    unfix_pi += self.pi[i]

                if self.compFix[i] == 1 or self.compFix[i] == 2:
                    #print "  Component ",i," is skipped from reestimation."
                    continue
                else:
                    # Check for model structure
                    if not self.struct:
                        # there might be mixtures down in the hierarchy, so pi[i] is passed to MStep
                        self.components[i].MStep(l[i, :], data, self.pi[i])

            # if there is a model structure we update the leader distributions only
            if self.struct:
                datRange = self.components[0].suff_dataRange

                for j in range(self.dist_nr):
                    for k in self.leaders[j]:
                        if j == 0:
                            prev = 0
                        else:
                            prev = datRange[j - 1]

                        # compute group posterior
                        g_post = numpy.array(l[k, :].tolist(), dtype='Float64')

                        for memb in self.groups[j][k]:
                            g_post += l[memb, :]

                        if isinstance(self.components[k][j], MixtureModel):
                            self.components[k][j].MStep(g_post, data.singleFeatureSubset(j), self.pi[k])
                        else:
                            self.components[k][j].MStep(g_post, data.getInternalFeature(j))

            # renormalizing mixing proportions in case of fixed components
            if fix_flag:
                if unfix_pi == 0.0:
                    #print "----\n",self,"----\n"
                    print "unfix_pi = ", unfix_pi
                    print "fix_pi = ", fix_pi
                    print "pi = ", self.pi
                    raise RuntimeError, "unfix_pi = 0.0"

                for i in range(self.G):
                    if self.compFix[i] == 0:
                        self.pi[i] = (self.pi[i] * fix_pi) / unfix_pi

            sys.stdout.flush()
            step += 1

    # XXX all old code should be collected together (-> remove ?)
    def EStep_old(self, data, mix_posterior=None, mix_pi=None, EStepParam=None):
        """ [Old implementation, kept around for regression testing]

        Reestimation of mixture parameters using the EM algorithm.

        @param data: DataSet object
        @param mix_posterior:[internal use only] necessary for the reestimation of
        mixtures as components
        @param mix_pi: [internal use only] necessary for the reestimation of
        mixtures as components
        @param EStepParam: additional paramenters for more complex EStep implementations, in
        this implementaion it is ignored

        @return: tuple of log likelihood matrices and sum of log-likelihood of components

        """
        log_l = numpy.zeros((self.G, data.N), dtype='Float64')
        log_col_sum = numpy.zeros(data.N, dtype='Float64')  # array of column sums of log_l
        log_pi = numpy.log(self.pi)  # array of log mixture coefficients

        # compute log of mix_posterior (if present)
        if mix_posterior is not None:
            log_mix_posterior = numpy.log(mix_posterior)

        # computing log posterior distribution
        for i in range(self.G):
            #print i,self.components[i].pdf(data).tolist()

            # XXX cache redundant pdfs for models with CSI structure
            log_l[i] = log_pi[i] + self.components[i].pdf(data)

        for j in range(data.N):
            log_col_sum[j] = sumlogs(log_l[:, j]) # sum over jth column of log_l

            # if posterior is invalid, check for model validity
            if log_col_sum[j] == float('-inf'):

                # if self is at the top of hierarchy, the model is unable to produce the
                # sequence and an exception is raised.
                if mix_posterior is None and not mix_pi:
                    #print "\n---- Invalid -----\n",self,"\n----------"
                    #print "\n---------- Invalid ---------------"
                    #print "mix_pi = ", mix_pi
                    #print "x[",j,"] = ", data.getInternalFeature(j)
                    #print "l[:,",j,"] = ", log_l[:,j]
                    #print 'data[',j,'] = ',data.dataMatrix[j]

                    raise InvalidPosteriorDistribution, "Invalid posterior distribution."

            # for valid posterior, normalize and go on
            else:
                # normalizing log posterior
                log_l[:, j] = log_l[:, j] - log_col_sum[j]
                # adjusting posterior for lower hierarchy mixtures
                if mix_posterior is not None:
                    # multiplying in the posterior of upper hierarchy mixture
                    log_l[:, j] = log_l[:, j] + log_mix_posterior[j]

        # computing data log likelihood as criteria of convergence
        log_p = numpy.sum(log_col_sum)

        return log_l, log_p


    def EStep(self, data, mix_posterior=None, mix_pi=None, EStepParam=None):
        """Reestimation of mixture parameters using the EM algorithm.

        @param data: DataSet object
        @param mix_posterior:[internal use only] necessary for the reestimation of
        mixtures as components
        @param mix_pi: [internal use only] necessary for the reestimation of
        mixtures as components
        @param EStepParam: additional paramenters for more complex EStep implementations, in
        this implementaion it is ignored

        @return: tuple of log likelihood matrices and sum of log-likelihood of components

        """
        log_l = numpy.zeros((self.G, data.N), dtype='Float64')
        log_col_sum = numpy.zeros(data.N, dtype='Float64')  # array of column sums of log_l
        log_pi = numpy.log(self.pi)  # array of log mixture coefficients

        # compute log of mix_posterior (if present)
        if mix_posterior is not None:
            log_mix_posterior = numpy.log(mix_posterior)
        else:
            log_mix_posterior = None

        # computing log posterior distribution
        for i in range(self.G):
            #print i,self.components[i].pdf(data).tolist()

            # XXX cache redundant pdfs for models with CSI structure
            pdf = self.components[i].pdf(data)
            log_l[i] = log_pi[i] + pdf

        # log_l is normalized in-place
        #print sum(numpy.exp(log_l)==float('-inf'))
        (log_l, log_p) = mixextend.get_normalized_posterior_matrix(log_l)

        if log_p == float('-inf'):
            raise InvalidPosteriorDistribution, "Invalid posterior distribution."

        if mix_posterior is not None:
            # multiplying in the posterior of upper hierarchy mixture
            log_l = log_l + log_mix_posterior

        return log_l, log_p


    def randMaxEM(self, data, nr_runs, nr_steps, delta, tilt=0, silent=False):
        """
        Performs `nr_runs` normal EM runs with random initial parameters
        and returns the model which yields the maximum likelihood.

        @param data: DataSet object
        @param nr_runs: number of repeated random initializations
        @param nr_steps: maximum number of steps in each run
        @param delta: minimim difference in log-likelihood before convergence
        @param tilt: 0/1 flag, toggles the use of a deterministic annealing in the training
        @param silent:0/1 flag, toggles verbose output

        @return: log-likelihood of winning model
        """
        if isinstance(data, numpy.ndarray):
            raise TypeError, "DataSet object required."
        elif isinstance(data, DataSet):
            if data.internalData is None:
                if not silent:
                    sys.stdout.write("Parsing data set...")
                    sys.stdout.flush()
                data.internalInit(self)
                if not silent:
                    sys.stdout.write("done\n")
                    sys.stdout.flush()
        else:
            raise ValueError, "Unknown input type format: " + str(data.__class__)

        logp_list = []
        best_logp = float('-inf')
        best_model = None
        candidate_model = copy.copy(self)  # copying the model parameters

        for i in range(nr_runs):
            # we do repeated random intializations until we get a model with valid posteriors in all components
            init = 0
            while not init:
                try:
                    candidate_model.modelInitialization(data)    # randomizing parameters of the model copy
                except InvalidPosteriorDistribution:
                    pass
                else:
                    init = 1

            try:
                (l, log_p) = candidate_model.EM(data, nr_steps, delta, silent=silent, tilt=tilt)  # running EM
            except ConvergenceFailureEM:
                sys.stdout.write("Run " + str(i) + " produced invalid model, omitted.\n")
            except InvalidPosteriorDistribution:
                sys.stdout.write("Run " + str(i) + " produced invalid model, omitted.\n")
            else:
                logp_list.append(log_p)

                # check whether current model is better than previous optimum
                if log_p > best_logp:
                    best_model = copy.copy(candidate_model)
                    best_logp = log_p

        if not silent:
            print "\nBest model likelihood over ", nr_runs, "random initializations:"
            print "Model likelihoods:", logp_list
            print "Average logp: ", sum(logp_list) / float(nr_runs), " SD:", numpy.array(logp_list).std()
            print "Best logp:", best_logp

        # check whether at least one run was sucessfully completed
        if best_model == None:
            raise ConvergenceFailureEM, 'All ' + str(nr_runs) + ' runs have failed.'

        self.components = best_model.components  # assign best parameter set to model 'self'
        self.pi = best_model.pi

        return best_logp  # return final data log likelihood


    def structureEM(self, data, nr_repeats, nr_runs, nr_steps, delta, tilt=0, silent=False):
        """
        EM training for models with CSI structure.
        First a candidate model is generated by using the randMaxEM procedure,
        then the structure is trained.

        @param data: DataSet object
        @param nr_repeats: number of candidate models to be generated
        @param nr_runs: number of repeated random initializations
        @param nr_steps: maximum number of steps for the long training run
        @param delta: minimim difference in log-likelihood before convergence
        @param tilt: 0/1 flag, toggles the use of deterministic annealing in the training
        @param silent:0/1 flag, toggles verbose output

        @return: log-likelihood of winning model
        """
        if isinstance(data, numpy.ndarray):
            raise TypeError, "DataSet object required."
        elif isinstance(data, DataSet):
            if data.internalData is None:
                sys.stdout.write("Parsing data set...")
                sys.stdout.flush()
                data.internalInit(self)
                sys.stdout.write("done\n")
                sys.stdout.flush()
        else:
            raise ValueError, "Unknown input type format: " + str(data.__class__)

        assert self.struct
        best_logp = None
        best_candidate = None
        candidate_model = copy.copy(self)  # copying the model parameters
        for r in range(nr_repeats):
            error = 0
            candidate_model.modelInitialization(data)    # randomizing parameters of the model copy
            log_p = candidate_model.randMaxEM(data, nr_runs, nr_steps, delta, tilt=tilt, silent=silent)
            ch = candidate_model.updateStructureGlobal(data, silent=silent)
            if not silent:
                print "Changes = ", ch
            while ch != 0:
                try:
                    candidate_model.EM(data, 30, 0.01, silent=1, tilt=0)
                    ch = candidate_model.updateStructureGlobal(data, silent=silent)
                    if not silent:
                        print "Changes = ", ch
                except ConvergenceFailureEM:
                    error = 1
                    break

            if not error:
                (l, log_p) = candidate_model.EM(data, 30, 0.01, silent=1, tilt=0)
                if r == 0 or log_p > best_logp:
                    best_logp = log_p
                    best_candidate = copy.copy(candidate_model)
            else:
                continue

        self.components = best_candidate.components  # assign best parameter set to model 'self'
        self.pi = best_candidate.pi
        self.groups = best_candidate.groups
        self.leaders = best_candidate.leaders
        self.freeParams = best_candidate.freeParams
        return best_logp

    # MStep is used for parameter estimation of lower hierarchy mixtures
    def MStep(self, posterior, data, mix_pi=None):
        self.EM(data, 1, 0.1, silent=True, mix_pi=mix_pi, mix_posterior=posterior)

    def mapEM(self, data, prior, max_iter, delta, silent=False, mix_pi=None, mix_posterior=None, tilt=0):
        """
        Reestimation of maximum a posteriori (MAP) mixture parameters using the EM algorithm.

        @param data: DataSet object
        @param max_iter: maximum number of iterations
        @param prior: an appropriate MixtureModelPrior object
        @param delta: minimal difference in likelihood between two iterations before
        convergence is assumed.
        @param silent: 0/1 flag, toggles verbose output
        @param mix_pi: [internal use only] necessary for the reestimation of
        mixtures as components
        @param mix_posterior:[internal use only] necessary for the reestimation of
        mixtures as components
        @param tilt: 0/1 flag, toggles the use of a deterministic annealing in the training

        @return: tuple of posterior matrix and log-likelihood from the last iteration
        """

        if isinstance(data, numpy.ndarray):
            raise TypeError, "DataSet object required."
        elif isinstance(data, DataSet):
            if data.internalData is None:
                if not silent:
                    sys.stdout.write("Parsing data set...")
                    sys.stdout.flush()
                data.internalInit(self)
                if not silent:
                    sys.stdout.write("done\n")
                    sys.stdout.flush()
        else:
            raise ValueError, "Unknown input type format: " + str(data.__class__)

        log_p_old = float('-inf')
        step = 0

        # if deterministic annealing is activated, increase number of steps by self.nr_tilt_steps
        if tilt:
            if not silent:
                sys.stdout.write("Running EM with " + str(self.nr_tilt_steps) + " steps of deterministic annealing.\n")
            max_iter += self.nr_tilt_steps

        # for lower hierarchy mixture we need the log of mix_posterior
        if mix_posterior is not None:
            log_mix_posterior = numpy.log(mix_posterior)
        else:
            log_mix_posterior = None

        while 1:
            log_p = 0.0
            # matrix of log posterior probs: components# * (sequence positions)
            log_l = numpy.zeros((self.G, data.N), dtype='Float64')
            #log_col_sum = numpy.zeros(data.N,dtype='Float64')  # array of column sums of log_l
            log_pi = numpy.log(self.pi)  # array of log mixture coefficients

            # computing log posterior distribution
            for i in range(self.G):
                log_l[i] = log_pi[i] + self.components[i].pdf(data)

            # computing data log likelihood as criteria of convergence
            # log_l is normalized in-place and likelihood is returned as log_p
            (log_l, log_p) = mixextend.get_normalized_posterior_matrix(log_l)

            # adjusting posterior for lower hierarchy mixtures
            if mix_posterior is not None:
                # multiplying in the posterior of upper hierarchy mixture
                log_l = log_l + log_mix_posterior

            # compute posterior likelihood matrix from log posterior
            l = numpy.exp(log_l)

            # update prior hyper parametes in an empirical Bayes fashion, if appropriate
            if prior.constant_hyperparams == 0:
                prior.updateHyperparameters(self, l, data)

            # we have to take the parameter prior into account to form the objective function
            # Since we assume independence between parameters in different components, the prior
            # contribution is given by a product over the individual component and structure priors
            log_prior = prior.pdf(self)

            # calculate objective function
            log_p += log_prior

            # checking for convergence
            # XXX special case for the parameter udpate of lower hierarchy mixtures
            if max_iter == 1 and mix_posterior != None and log_p == float('-inf'):
                diff = -1.0  # dummy value for diff
            else:
                diff = (log_p - log_p_old)

            if log_p_old != -1.0 and not silent and step > 0:
                if tilt and step <= self.nr_tilt_steps:
                    sys.stdout.write("TILT Step " + str(step) + ": log posterior: " + str(log_p) + "\n")
                else:
                    sys.stdout.write("Step " + str(step) + ": log posterior: " + str(log_p) + "   (diff=" + str(diff) + ")\n")

            if diff < 0.0 and step > 0 and abs(diff / log_p_old) > self.err_tol:
                #print log_p,log_p_old, diff,step,abs(diff / log_p_old)
                print "WARNING: EM divergent."
                diff = - diff
                #raise ConvergenceFailureEM,"Convergence failed, EM divergent: "

            if (not tilt or (tilt and step + 1 >= self.nr_tilt_steps)) and delta >= diff and max_iter != 1:
                if not silent:
                    sys.stdout.write("Convergence reached with log_p " + str(log_p) + " after " + str(step) + " steps.\n")
                if self.identFlag:
                    self.identifiable()
                return (log_l, log_p)

            log_p_old = log_p
            if step == max_iter:
                if not silent:
                    sys.stdout.write("Max_iter " + str(max_iter) + " reached -> stopping\n")

                if self.identFlag:
                    self.identifiable()
                return (log_l, log_p)

            # deterministic annealing, shifting posterior toward uniform distribution.
            if tilt and step + 1 <= self.nr_tilt_steps and mix_posterior is None:
                h = self.heat - (step * (self.heat / (self.nr_tilt_steps))  )
                for j in range(data.N):
                    uni = 1.0 / self.G
                    tilt_l = (uni - l[:, j]) * h
                    l[:, j] += tilt_l

            # variables for component fixing
            fix_pi = 1.0
            unfix_pi = 0.0
            fix_flag = 0   # flag for fixed mixture components

            # update component parameters and mixture weights
            for i in range(self.G):
                if self.compFix[i] & 2:   # pi[i] is fixed
                    fix_pi -= self.pi[i]
                    fix_flag = 1
                    continue
                else:
                    # for mixtures of mixtures we need to multiply in the mix_pi[i]s
                    if mix_pi is not None:
                        self.pi[i] = ( l[i, :].sum() + prior.piPrior.alpha[i] - 1.0 ) / ((data.N * mix_pi) + prior.piPrior.alpha_sum - self.G )
                        #print i, ( l[i,:].sum() + prior.piPrior.alpha[i] -1.0 ),((data.N * mix_pi) + prior.piPrior.alpha_sum - self.G )
                    else:
                        self.pi[i] = ( l[i, :].sum() + prior.piPrior.alpha[i] - 1.0 ) / (data.N + ( prior.piPrior.alpha_sum - self.G) )

                    unfix_pi += self.pi[i]

                if self.compFix[i] & 1:
                    continue
                else:
                    # Check for model structure
                    if not self.struct:
                        prior.compPrior.mapMStep(self.components[i], l[i, :], data, self.pi[i], i)

            # if there is a model structure we update the leader distributions only
            if self.struct:
                for j in range(self.dist_nr):
                    for k in self.leaders[j]:
                        # compute group posterior
                        # XXX extension function for pooled posterior ?
                        g_post = copy.deepcopy(l[k, :])
                        g_pi = self.pi[k]
                        for memb in self.groups[j][k]:
                            g_post += l[memb, :]
                            g_pi += self.pi[memb]

                        if isinstance(self.components[k][j], MixtureModel):
                            prior.compPrior[j].mapMStep(self.components[k][j], g_post, data.singleFeatureSubset(j), g_pi, k)
                        else:
                            try:
                                prior.compPrior[j].mapMStep(self.components[k][j], g_post, data.getInternalFeature(j), g_pi, k)
                            except InvalidPosteriorDistribution:
                                raise

            # renormalizing mixing proportions in case of fixed components
            if fix_flag:
                if unfix_pi == 0.0:
                    #print "----\n",self,"----\n"
                    #print "unfix_pi = ", unfix_pi
                    #print "fix_pi = ", fix_pi
                    #print "pi = ", self.pi
                    #print self
                    raise ValueError, "unfix_pi = 0.0"
                for i in range(self.G):
                    if self.compFix[i] == 0:
                        self.pi[i] = (self.pi[i] * fix_pi) / unfix_pi

            sys.stdout.flush()
            step += 1

    def classify(self, data, labels=None, entropy_cutoff=None, silent=0, EStep=None, EStepParam=None):
        """
        Classification of input 'data'.
        Assignment to mixture components by maximum likelihood over
        the component membership posterior. No parameter reestimation.

        @param data: DataSet object
        @param labels: optional sample IDs
        @param entropy_cutoff: entropy threshold for the posterior distribution. Samples which fall
        above the threshold will remain unassigned
        @param silent: 0/1 flag, toggles verbose output
        @param EStep: function implementing the EStep, by default self.EStep
        @param EStepParam: additional paramenters for more complex EStep implementations

        @return: list of class labels
        """
        if isinstance(data, DataSet):
            if data.internalData is None:
                if not silent:
                    sys.stdout.write("Parsing data set...")
                    sys.stdout.flush()
                data.internalInit(self)
                if not silent:
                    sys.stdout.write("done\n")
                    sys.stdout.flush()
        else:
            raise ValueError, "Invalid input type format: " + str(data.__class__) + ", DataSet required."

        if EStep == None:
            EStep = self.EStep

        labels = data.sampleIDs

        # compute posterior distribution of component membership for cluster assignment
        [l, log_l] = EStep(data, EStepParam=EStepParam)

        if not silent:
            print "classify loglikelihood: " + str(log_l) + ".\n"

        # cluster assingments initialised with -1
        z = numpy.ones(data.N, dtype='Int32') * -1

        entropy_list = numpy.zeros(data.N, dtype='Float64')
        max_entropy = math.log(self.G, 2)

        # compute posterior entropies
        for i in range(data.N):
            exp_l = numpy.exp(l[:, i])
            if self.G == 1:
                entropy_list[i] = entropy(exp_l)
            else:
                entropy_list[i] = entropy(exp_l) / max_entropy

            if not entropy_cutoff:
                entropy_cutoff = float('inf')

            #            if not silent:
            #                print 'sample', data.sampleIDs[i],':',entropy_list[i]

            # apply entropy cutoff
            if entropy_list[i] < entropy_cutoff:
                # cluster assignment by maximum likelihood over the component membership posterior
                z[i] = numpy.argmax(l[:, i])

        if not silent:
            # printing out the clusters
            cluster = {}
            en = {}
            for j in range(-1, self.G, 1):
                cluster[j] = []
                en[j] = []

            for i in range(data.N):
                cluster[z[i]].append(labels[i])
                en[z[i]].append(entropy_list[i])

            print "\n** Clustering **"
            for j in range(self.G):
                print "Cluster ", j, ', size', len(cluster[j])
                print cluster[j], "\n"

            print "Unassigend due to entropy cutoff:"
            print cluster[-1], "\n"

        return z


    def isValid(self, x):
        """
        Exhaustive check whether a given DataSet is compatible with the model.
        If self is a lower hierarchy mixture 'x' is a single data sample in external representation.
        """
        if isinstance(x, DataSet):
            # self is at top level of the hierarchy
            for i in range(self.G):
                for j in range(x.N):
                    try:
                        self.components[i].isValid(x.dataMatrix[j])
                    except InvalidDistributionInput, ex:
                        ex.message = "\n\tin MixtureModel.components[" + str(i) + "] for DataSet.dataMatrix[" + str(j) + "]."
                        raise
        else:
            for i in range(self.G):
                try:
                    self.components[i].isValid(x)
                except InvalidDistributionInput, ex:
                    ex.message += "\n\tMixtureModel.components[" + str(i) + "]."
                    raise

    def formatData(self, x):
        [new_p, res] = self.components[0].formatData(x)
        return [new_p, res]

    def reorderComponents(self, order):
        """
        Reorder components into a new order

        @param order: list of indices giving the new order
        """
        # reordering components
        components_order = []
        pi_order = []
        compFix_order = []

        index_map = {}   # maps old indices to new indices
        for k, i in enumerate(order):
            index_map[i] = k
            pi_order.append(self.pi[i])
            components_order.append(self.components[i])
            compFix_order.append(self.compFix[i])

        # assigning ordered parameters
        self.pi = numpy.array(pi_order, dtype='Float64')
        self.components = components_order
        self.compFix = compFix_order

        order_leaders = []
        order_groups = []
        # updating structure if necessary
        if self.struct:
            f = lambda x: index_map[x]
            for j in range(self.dist_nr):
                new_l = map(f, self.leaders[j])
                order_leaders.append(new_l)
                d = {}
                for h in self.leaders[j]:
                    new_g = map(f, self.groups[j][h])
                    d[index_map[h]] = new_g
                order_groups.append(d)

            # reformat leaders and groups such that the minimal
            # index of a group is used as the leader
            for j in range(self.dist_nr):
                for lind, lead in enumerate(order_leaders[j]):
                    tg = [lead] + order_groups[j][lead]

                    tg.sort()
                    nl = tg.pop(0)
                    order_leaders[j][lind] = nl
                    order_groups[j].pop(lead)

                    order_groups[j][nl] = tg
                order_leaders[j].sort()

        self.leaders = order_leaders
        self.groups = order_groups

    def identifiable(self):
        """ To provide identifiability the components are ordered by the mixture coefficient in
            ascending order.
        """
        indices = {}
        # storing indices for sorting
        for i in range(self.G):
            indices[i] = self.pi[i]

        # determine new order of components by ascending mixture weight
        items = [(v, k) for k, v in indices.items()]
        items.sort()

        order = []
        for it in items:
            order.append(it[1])
        self.reorderComponents(order)

    def flatStr(self, offset):
        offset += 1
        s = "\t" * offset + str(';Mix') + ";" + str(self.G) + ";" + str(self.pi.tolist()) + ";" + str(self.compFix) + "\n"
        for c in self.components:
            s += c.flatStr(offset)

        return s

    def printClusterEntropy(self, data):
        """
        Print out cluster stability measured by the entropy of the component membership posterior.

        @param data: DataSet object
        """
        if isinstance(data, numpy.ndarray):
            sequence = data
            seqLen = len(sequence)

        elif isinstance(data, DataSet):
            sequence = data.internalData

        print "-------- getClusterEntropy ------------"
        post_entropy = []
        log_l = self.getPosterior(sequence)
        l = numpy.exp(log_l)
        for i in range(data.N):
            post_entropy.append(entropy(l[:, i]))

        max_entropy = entropy([1.0 / self.G] * self.G)
        print "Max entropy for G=", self.G, ":", max_entropy

        print "--------\nPosterior distribuion: % max entropy"
        for i in range(data.N):
            print data.sampleIDs[i], ": ", post_entropy[i], " -> ", post_entropy[i] / max_entropy, "%"
        print "--------\n"

    def posteriorTraceback(self, x):
        return self.pdf(x)[0]

    def printTraceback(self, data, z, en_cut=1.01):
        """
        Prints out the posterior traceback, i.e. a detailed account of the
        contribution to the component membership posterior of each sample in each feature ordered
        by a clustering.

        @param data: DataSet object
        @param z: class labels
        @param en_cut: entropy threshold
        """
        if isinstance(data, numpy.ndarray):
            sequence = data
            seqLen = len(sequence)
        elif isinstance(data, DataSet):
            templist = []
            for i in range(data.N):
                [t, dat] = self.components[0].formatData(data.dataMatrix[i])
                templist.append(dat)
            sequence = numpy.array(templist)
            labels = data.sampleIDs
            seqLen = len(sequence)

        l, log_p = self.EStep(data)

        print "seqLen = ", data.p
        print "pi = ", self.pi
        max_en = entropy([1.0 / self.G] * self.G)
        for c in range(self.G):
            temp = numpy.where(z == c)
            c_index = temp[0]
            print "\n---------------------------------------------------"
            print "Cluster = ", c, ": ", c_index
            for j in c_index:
                t = self.pdf(numpy.array([sequence[j]]))[0]
                print "\nj = ", j, ", id =", data.sampleIDs[j], ", log_l = ", t, " -> ", numpy.exp(t)
                print "posterior = ", numpy.exp(l[:, j]).tolist(), "\n"
                tb = []
                for g in range(self.G):
                    ll = self.components[g].posteriorTraceback(data.internalData[j])
                    tb.append(ll)
                tb_arr = numpy.array(tb, dtype='Float64')
                for i in range(len(tb[0])):
                    s = sumlogs(tb_arr[:, i])
                    tb_arr[:, i] = tb_arr[:, i] - s
                exp_arr = numpy.exp(tb_arr)
                exp_tb = exp_arr.tolist()
                max_comp = numpy.zeros(len(tb[0]))
                en_percent = numpy.zeros(len(tb[0]), dtype='Float64')
                for i in range(len(tb[0])):
                    max_comp[i] = numpy.argmax(tb_arr[:, i])
                    en_percent[i] = entropy(exp_arr[:, i]) / max_en
                print "     ",
                for q in range(len(tb[0])):
                    if en_percent[q] < en_cut:
                        head_len = len(data.headers[q])
                        print " " * (16 - head_len) + str(data.headers[q]),
                print
                print "     ",
                for q in range(len(tb[0])):
                    if en_percent[q] < en_cut:
                        x_len = len(str(data.dataMatrix[j][q]))
                        print " " * (16 - x_len) + str(data.dataMatrix[j][q]),
                print
                for e in range(self.G):
                    print e, ": [",
                    if e != z[j]:
                        for k in range(len(exp_tb[e])):
                            if en_percent[k] < en_cut:
                                print "%16.10f" % (exp_tb[e][k],),
                    else:
                        for k in range(len(exp_tb[e])):
                            if en_percent[k] < en_cut:
                                print " *  %12.10f" % (exp_tb[e][k],),
                    print "]"
                print "max  ",
                for e in range(len(data.headers)):
                    if en_percent[e] < en_cut:
                        print " " * 15 + str(max_comp[e]),
                print
                print "%EN  ",
                for e in range(len(data.headers)):
                    if en_percent[e] < en_cut:
                        print "%16.4f" % (en_percent[e],),
                print


    def update_suff_p(self):
        suff_p = self.components[0].update_suff_p()
        for i in range(1, self.G):
            sp = self.components[i].update_suff_p()
            assert sp == suff_p
        self.suff_p = suff_p
        return self.suff_p

    def updateStructureGlobal(self, data, silent=1):
        """
        Updating CSI structure by chosing smallest KL distance merging, optimizing the AIC score.
        This was the first approach implemented for the CSI structure learning and using the Bayesian approach instead
        is stronly recommended.

        @param data: DataSet object
        @param silent: verbosity flag

        @return: number of structure changes
        """
        assert self.struct == 1, "No structure in model."

        datRange = self.components[0].suff_dataRange
        new_leaders = []
        new_groups = []
        change = 0

        # building posterior factor matrix for the current group structure
        l = numpy.zeros((self.G, data.N, self.dist_nr ), dtype='Float64')
        for j in range(self.dist_nr):
            if j == 0:
                prev = 0
            else:
                prev = datRange[j - 1]
            for lead_j in self.leaders[j]:
                if self.components[lead_j][j].suff_p == 1:
                    l_row = self.components[lead_j][j].pdf(data.internalData[:, datRange[j] - 1])
                else:
                    l_row = self.components[lead_j][j].pdf(data.internalData[:, prev:datRange[j]])
                l[lead_j, :, j] = l_row
                for v in self.groups[j][lead_j]:
                    l[v, :, j] = l_row

        g = numpy.sum(l, 2)
        for k in range(self.G):
            g[k, :] += numpy.log(self.pi[k])
        sum_logs = numpy.zeros(data.N, dtype='Float64')
        for n in range(data.N):
            sum_logs[n] = sumlogs(g[:, n])
        lk = sum(sum_logs)
        for j in range(self.dist_nr):
            # initialize free parameters
            full_fp_0 = self.freeParams
            if not silent:
                print "\n************* j = ", j, "*****************\n"
            term = 0
            while not term:
                nr_lead = len(self.leaders[j])
                if nr_lead == 1:
                    break
                min_dist = float('inf')
                merge_cand1 = -1
                merge_cand2 = -1
                # compute symmetric KL distances between all groups
                for i in range(nr_lead):
                    for k in range(i + 1, nr_lead):
                        d = sym_kl_dist(self.components[self.leaders[j][i]][j], self.components[self.leaders[j][k]][j])
                        if d < min_dist:
                            min_dist = d
                            merge_cand1 = self.leaders[j][i]
                            merge_cand2 = self.leaders[j][k]
                if not silent:
                    print "-------------------"
                    print merge_cand1, " -> ", merge_cand2, " = ", min_dist
                    print self.components[merge_cand1][j]
                    print self.components[merge_cand2][j]

                full_BIC_0 = -2 * lk + (full_fp_0 * numpy.log(data.N))
                # compute merged distribution of candidates with minimal KL distance
                candidate_dist = copy.copy(self.components[merge_cand1][j])
                merge_list = [self.components[merge_cand2][j]]

                # computing total weights for the two groups to be merged
                pi_list = [self.pi[merge_cand1], self.pi[merge_cand2]]
                for m in self.groups[j][merge_cand1]:
                    pi_list[0] += self.pi[m]
                for m in self.groups[j][merge_cand2]:
                    pi_list[1] += self.pi[m]

                # computing candidate leader distribution for the merged group
                candidate_dist.merge(merge_list, pi_list)

                if not silent:
                    print "candidate:", candidate_dist

                # computing the new reduced model complexity
                full_fp_1 = full_fp_0 - self.components[merge_cand1][j].freeParams

                # initialising new group structure with copies of the current structure
                new_leaders = copy.deepcopy(self.leaders)
                new_groups = copy.deepcopy(self.groups)

                # removing merged leader from self.leaders
                ind = new_leaders[j].index(merge_cand2)
                new_leaders[j].pop(ind)

                # joining merged groups and removing old group entry
                new_groups[j][merge_cand1] += [merge_cand2] + new_groups[j][merge_cand2]
                new_groups[j].pop(merge_cand2)

                if not silent:
                    print "\ncandidate model structure:"
                    print "lead = ", new_leaders[j]
                    print "groups = ", new_groups[j], "\n"

                # updating likelihood matrix
                l_1 = copy.copy(l)
                if j == 0:
                    prev = 0
                else:
                    prev = datRange[j - 1]
                if candidate_dist.suff_p == 1:
                    l_row = candidate_dist.pdf(data.internalData[:, datRange[j] - 1])
                else:
                    l_row = candidate_dist.pdf(data.internalData[:, prev:datRange[j]])
                l_1[merge_cand1, :, j] = l_row
                for v in new_groups[j][merge_cand1]:
                    l_1[v, :, j] = l_row
                g = numpy.sum(l_1, 2)
                for k in range(self.G):
                    g[k, :] += numpy.log(self.pi[k])

                sum_logs = numpy.zeros(data.N, dtype='Float64')
                for n in range(data.N):
                    sum_logs[n] = sumlogs(g[:, n])
                lk_1 = sum(sum_logs)
                full_BIC_1 = -2 * lk_1 + (full_fp_1 * numpy.log(data.N))
                AIC_0 = -2 * lk + ( 2 * full_fp_0 )
                AIC_1 = -2 * lk_1 + ( 2 * full_fp_1 )

                if not silent:
                    print "LK_0: ", lk
                    print "LK_1: ", lk_1

                    print "full_fp_0 =", full_fp_0
                    print "full_fp_1 =", full_fp_1

                    print "\nfull_BIC_0 =", full_BIC_0
                    print "full_BIC_1 =", full_BIC_1

                    print "Vorher: AIC_0 =", AIC_0
                    print "Nachher: AIC_1 =", AIC_1

                    #if  AIC_1 < AIC_0:
                    #    print "Merge accepted according to AIC"
                    #else:
                    #    print "Merge rejected according to AIC"

                if AIC_1 < AIC_0:
                    if not silent:
                        print "\n*** Merge accepted !"
                    change += 1

                    # new_model.components[merge_cand1][j]
                    # assigning leader distribution to group members
                    self.components[merge_cand1][j] = copy.copy(candidate_dist)
                    for g in new_groups[j][merge_cand1]:
                        self.components[g][j] = self.components[merge_cand1][j]

                    self.leaders = new_leaders
                    self.groups = new_groups

                    lk = lk_1
                    l = l_1

                    full_fp_0 = full_fp_1

                    self.freeParams = full_fp_1

                    # if only one group is left terminate, update free parameters and continue with next variable
                    if len(self.leaders[j]) == 1:
                        if not silent:
                            print "*** Fully merged !"
                        term = 1
                else:
                    if not silent:
                        print "\n*** Merge rejected: Abort !"
                        # finished with this variable, update free parameters and go on
                    term = 1

                    # reassinging groups and leaders
        for j in range(self.dist_nr):
            for l in self.leaders[j]:
                for g in self.groups[j][l]:
                    self.components[g][j] = self.components[l][j]
        return change

    def minimalStructure(self):
        """ Finds redundant components in the model structure and collapses the
            structure to a minimal representation.
        """
        assert self.struct == 1, "No structure in model."

        distNr = self.components[0].dist_nr
        # features with only one group can be excluded
        exclude = []
        for i in range(distNr):
            if len(self.leaders[i]) == 1:
                exclude.append(i)
            # get first feature with more than one group
        first = -1
        for i in range(distNr):
            if i not in exclude:
                first = i
                break
            # initialising group dictionaries for first non-trivial group structure
        firstgroup_dicts = []
        for j in range(len(self.leaders[first])):
            d = {}
            for k in [self.leaders[first][j]] + self.groups[first][self.leaders[first][j]]:
                d[k] = 1
            firstgroup_dicts.append(d)

        # initialising group dictionaries for remaining features
        allgroups_dicts = []
        for i in range(first + 1, distNr, 1):
            if i in exclude:
                continue
            gdicts_i = []
            for l in self.leaders[i]:
                d = {}
                for k in [l] + self.groups[i][l]:
                    d[k] = 1
                gdicts_i.append(d)
            allgroups_dicts.append(gdicts_i)

        toMerge = []
        # for each group in first non-trivial structure
        for g, fdict in enumerate(firstgroup_dicts):
            candidate_dicts = [fdict]
            # for each of the other non-trivial features
            for i, dict_list in enumerate(allgroups_dicts):
                new_candidate_dicts = []
                # for each group in the i-th feature
                for adict in dict_list:
                    # for each candidate group
                    for j, cand_dict in enumerate(candidate_dicts):
                        # find intersection
                        inter_d = dict_intersection(cand_dict, adict)
                        if len(inter_d) >= 2:
                            new_candidate_dicts.append(inter_d)
                candidate_dicts = new_candidate_dicts
                # check whether any valid candidates are left
                if len(candidate_dicts) == 0:
                    break
            if len(candidate_dicts) > 0:
                for c in candidate_dicts:
                    toMerge.append(c.keys())

        if len(toMerge) > 0:  # postprocess toMerge to remove single entry sets
            for i in range(len(toMerge) - 1, -1, -1):
                if len(toMerge[i]) == 1:
                    toMerge.pop(i)

        d_merge = None
        if len(toMerge) > 0:
            d_merge = {}  # map from indices to be merged to respective leader index
            for m in range(len(toMerge)):
                tm = toMerge[m]
                tm.sort()
                lead = tm.pop(0)
                for x in tm:
                    d_merge[x] = lead
            new_pi = self.pi.tolist()
            new_compFix = copy.copy(self.compFix)

            l = d_merge.keys()
            l.sort()
            l.reverse()
            for j in l:

                # update new_pi
                pi_j = new_pi.pop(j)
                new_pi[d_merge[j]] += pi_j
                self.components.pop(j)  # remove component

                # update compFix
                cf_j = new_compFix.pop(j)
                new_compFix[d_merge[j]] = new_compFix[d_merge[j]] or cf_j

                # update leaders
                for l1 in range(len(self.leaders)):
                    for l2 in range(len(self.leaders[l1])):
                        if self.leaders[l1][l2] == j:
                            self.leaders[l1][l2] = d_merge[j]
                        elif self.leaders[l1][l2] > j:
                            self.leaders[l1][l2] -= 1

                # update component indices in groups
                for g_j in range(len(self.groups)):
                    for g in self.groups[g_j].keys():
                        if g == j:
                            tmp = self.groups[g_j][g]
                            self.groups[g_j].pop(j)
                            self.groups[g_j][d_merge[j]] = tmp
                        elif g > j:
                            tmp = self.groups[g_j][g]
                            self.groups[g_j].pop(g)
                            self.groups[g_j][g - 1] = tmp

                # remove merged component from groups
                for g_j in range(len(self.groups)):
                    for g in self.groups[g_j].keys():
                        for gm in range(len(self.groups[g_j][g]) - 1, -1, -1):
                            if self.groups[g_j][g][gm] == j:
                                self.groups[g_j][g].pop(gm)
                            elif self.groups[g_j][g][gm] > j:
                                self.groups[g_j][g][gm] -= 1

            self.G = self.G - len(l)  # update number of components
            self.pi = numpy.array(new_pi, dtype='Float64')  # set new pi in model
            self.compFix = new_compFix

        self.updateFreeParams()
        if self.identFlag:
            self.identifiable()

        return d_merge

    def removeComponent(self, ind):
        """
        Deletes a component from the model.

        @param ind: ind of component to be removed
        """
        self.G = self.G - 1  # update number of components
        tmp = self.pi.tolist() # update pi
        tmp.pop(ind)
        tmp = map(lambda x: x / sum(tmp), tmp) # renormalize pi
        self.pi = numpy.array(tmp, dtype='Float64')  # set new pi in model
        self.components.pop(ind)  # remove component
        if self.compFix:
            self.compFix.pop(ind)

        # update CSI structure if necessary
        if self.struct:
            #update leaders
            for k, ll in enumerate(self.leaders):
                try:  # remove ind from leader lists
                    r = ll.index(ind)
                    self.leaders[k].pop(r)
                except ValueError:
                    pass
                for i, l in enumerate(self.leaders[k]): # update to new indices
                    if l > ind:
                        self.leaders[k][i] -= 1

            new_groups = []
            # update groups
            for i, dg in enumerate(self.groups):
                new_groups.append({})
                for k in dg.keys():
                    if k == ind:
                        # case ind is leader: remove ind and select new leader
                        gr = self.groups[i].pop(k)
                        if len(gr) > 0:  # need to re-enter with new leader
                            new_l = gr.pop(0)
                            if new_l > ind:
                                new_l -= 1
                            for r in range(len(gr)):
                                if gr[r] > ind:
                                    gr[r] -= 1
                            new_groups[i][new_l] = gr
                            self.leaders[i].append(new_l)
                    else:
                        # case ind is not leader but might be in the group
                        gr = self.groups[i].pop(k)
                        if ind in gr:
                            gr.remove(ind)
                        for r in range(len(gr)):
                            if gr[r] > ind:
                                gr[r] -= 1
                        if k > ind:
                            new_groups[i][k - 1] = gr  # need to change key
                        else:
                            new_groups[i][k] = gr
        self.groups = new_groups
        self.updateFreeParams()
        if self.identFlag:
            self.identifiable()

    def merge(self, dlist, weights):
        raise DeprecationWarning, 'Part of the outdated structure learning implementation.'

    def printStructure(self, data=None):
        """
        Pretty print of the model structure
        """
        assert self.struct == 1, "No model structure."
        if data:
            headers = data.headers
        else:
            headers = range(self.dist_nr)
        for i in range(self.dist_nr):
            print "Feature " + str(i) + ": " + str(headers[i])
            for j, l in enumerate(self.leaders[i]):
                if self.groups[i][l] == []:
                    print "\tGroup " + str(j) + ": " + "(" + str(l) + ")"
                else:
                    print "\tGroup " + str(j) + ": " + str(tuple([l] + self.groups[i][l]))
                print "\t  ", self.components[l][i], "\n"


    def updateFreeParams(self):
        """
        Updates the number of free parameters for the current group structure
        """
        if self.struct == 0:
            self.freeParams = self.G - 1
            for c in self.components:
                if isinstance(c, ProductDistribution):
                    for m in c.distList:
                        self.freeParams += m.freeParams
                else:
                    self.freeParams += c.freeParams
                    #print 'param models', c.freeParams
                    #self.freeParams = (self.components[0].freeParams * self.G) + self.G-1
        else:
            fp = 0
            for i in range(self.dist_nr):
                for l in self.leaders[i]:
                    fp += self.components[l][i].freeParams
            fp += self.G - 1
            self.freeParams = fp


    def validStructure(self):
        """
        Checks whether the CSI structure is syntactically correct. Mostly for debugging.
        """
        if self.struct == 0:
            return True
        r = range(self.G)
        try:
            # check valid entries in group and leader
            for j in range(self.dist_nr):
                for l in self.leaders[j]:
                    assert l in r
                    for g in self.groups[j][l]:
                        assert g in r
                # check completeness of structure
            for j in range(self.dist_nr):
                tmp = copy.copy(self.leaders[j])
                for g in self.groups[j].keys():
                    tmp += copy.copy(self.groups[j][g])
                tmp.sort()
                assert tmp == r
        except AssertionError:
            print 'Invalid structure:', j
            print self.leaders
            print self.groups
            raise

    def sufficientStatistics(self, posterior, data):
        """
        Returns sufficient statistics for a given data set and posterior.
        """
        assert self.dist_nr == 1
        sub_post = get_posterior(self, data, logreturn=True)

        suff_stat = []
        dat = data.getInternalFeature(0)
        for i in range(self.G):
            if self.compFix[i] == 2:
                suff_stat.append([float('-inf'), float('inf')])
                continue

            np = sub_post[i] + posterior
            inds = numpy.where(np != float('-inf'))
            suff_stat.append(self.components[i][0].sufficientStatistics(np[inds], dat[inds]))

        return suff_stat

