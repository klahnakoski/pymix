import copy
import numpy
import random
import sys
from core.distributions.product import ProductDistribution
from core.pymix_util.errors import ConvergenceFailureEM, InvalidPosteriorDistribution
from core.models.labeled import LabeledMixtureModel

from core.models.mixture import MixtureModel
from core.pymix_util.constrained_dataset import ConstrainedDataSet
from core.pymix_util.dataset import DataSet
from core.pymix_util.maths import sum_logs


class ConstrainedMixtureModel(MixtureModel):
    """
    Class for a mixture model containing the pairwise constrained version of the E-Step
    """

    def __init__(self, G, pi, components, compFix=None, struct=0):
        MixtureModel.__init__(self, G, pi, components, compFix=compFix, struct=struct, identifiable=0)

    def __copy__(self):
        copy_model = MixtureModel.__copy__(self)
        return CMMfromMM(copy_model)

    def EM(self, data, max_iter, delta, prior_positive,
        prior_negative, previous_posterior, prior_type, normaliziering=False, silent=False,
        mix_pi=None, mix_posterior=None, tilt=0):
        """
        Reestimation of mixture parameters using the EM algorithm.
        This method do some initial checking and call the EM from
        MixtureModel with the constrained labels E step

        @param data: DataSet object
        @param max_iter: maximum number of iterations
        @param delta: minimal difference in likelihood between two iterations before
        convergence is assumed.
        @param prior_positive: importance parameter for positive constraints
        @param prior_negative: importance parameter for negative constraints
        @param previous_posterior: matrix containing the posterior of the previous model assigments
        @param prior_type: 1 positive constr.
                           2 negative constr.
                           3 positive and negative constr.
        @param silent: 0/1 flag, toggles verbose output
        @param mix_pi: [internal use only] necessary for the reestimation of
        mixtures as components
        @param mix_posterior:[internal use only] necessary for the reestimation of
        mixtures as components
        @param tilt: 0/1 flag, toggles the use of a deterministic annealing in the training

        @return: tuple of posterior matrix and log-likelihood from the last iteration
        """
        assert isinstance(data, ConstrainedDataSet), 'Data set does not contain labels, Labeled EM can not be performed'
        if data.pairwisepositive == None:
            assert (prior_positive == 0), 'Data set does not contain pairwise constraints, Labeled EM can not be performed'
        if data.pairwisenegative == None:
            assert (prior_negative == 0), 'Data set does not contain pairwise constraints, Labeled EM can not be performed'

        class EArgs:
            def __init__(self):
                self.prior_positive = prior_positive
                self.prior_negative = prior_negative
                self.normaliziering = normaliziering
                self.previous_posterior = previous_posterior
                self.prior_type = prior_type

        return MixtureModel.EM(self, data, max_iter, delta, silent=silent, mix_pi=mix_pi, mix_posterior=mix_posterior,
            tilt=tilt, EStep=self.EStep, EStepParam=EArgs())


    def modelInitialization(self, data, prior_positive, prior_negative, prior_type, normaliziering=False, rtype=1):
        """
        Perform model initialization given a random assigment of the
        data to the models.

        @param data: DataSet object
        @param rtype: type of random assignments.
        0 = fuzzy assingment
        1 = hard assingment
        @return posterior assigments
        """

        if not isinstance(data, ConstrainedDataSet):
            raise TypeError, "DataSet object required, got" + str(data.__class__)

        else:
            if data.internalData is None:
                data.internalInit(self)

        # reset structure if applicable
        if self.struct:
            self.initStructure()

        # generate 'random posteriors'
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

        class EArgs:
            def __init__(self):
                self.prior_positive = prior_positive
                self.prior_negative = prior_negative
                self.normaliziering = normaliziering
                self.previous_posterior = l
                self.prior_type = prior_type


        # peform constrain updates (non random!)
        self.EStep(data, EStepParam=EArgs())

        #print 'Random init l:'
        #for u in range(self.G):
        #    print u,l[u,:].tolist()

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
                if isinstance(self.components[i], ProductDistribution):
                    last_index = 0
                    for j in range(self.components[i].dist_nr):
                        if isinstance(self.components[i].distList[j], MixtureModel):
                            dat_j = data.singleFeatureSubset(j)
                            self.components[i].distList[j].modelInitialization(dat_j, rtype=rtype)
                        else:
                            loc_l = l[i, :]
                            # masking missing values from parameter estimation
                            if data.missingSymbols.has_key(j):
                                ind_miss = data.getMissingIndices(j)
                                for k in ind_miss:
                                    loc_l[k] = 0.0
                            self.components[i][j].MStep(loc_l, data.getInternalFeature(j))

                else:  # components are not ProductDistributions -> invalid
                    raise TypeError

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


    def classify(self, data, prior_positive, prior_negative, previous_posterior,
        prior_type, labels=None, entropy_cutoff=None,
        silent=0, normaliziering=False):
        """
        Classification of input 'data'.  Assignment to mixture components by maximum likelihood
        over the component membership posterior. No parameter
        reestimation.

        @param data: DataSet object
        @param labels: optional sample IDs
        @param prior_positive: importance parameter for positive constraints
        @param prior_negative: importance parameter for negative constraints
        @param previous_posterior: matrix containing the posterior of the previous model assigments
        @param prior_type: 1 positive constr.
                           2 negative constr.
                           3 positive and negative constr.
        @param entropy_cutoff: entropy threshold for the posterior distribution. Samples which fall
        above the threshold will remain unassigned
        @param silent: 0/1 flag, toggles verbose output

        @return: list of class labels
        """

        assert isinstance(data, ConstrainedDataSet), 'Data set does not contain labels, Labeled EM can not be performed'
        if data.pairwisepositive == None:
            assert (prior_positive == 0), 'Data set does not contain pairwise constraints, Labeled EM can not be performed'
        if data.pairwisenegative == None:
            assert (prior_negative == 0), 'Data set does not contain pairwise constraints, Labeled EM can not be performed'

        class EArgs:
            def __init__(self):
                self.prior_positive = prior_positive
                self.prior_negative = prior_negative
                self.normaliziering = normaliziering
                self.previous_posterior = previous_posterior
                self.prior_type = prior_type

        return MixtureModel.classify(self, data, labels=labels, entropy_cutoff=entropy_cutoff,
            silent=silent, EStep=self.EStep, EStepParam=EArgs())

    def EStep(self, data, mix_posterior=None, mix_pi=None, EStepParam=None):
        """
        Reestimation of mixture parameters using the EM algorithm.

        @param data: DataSet object
        @param mix_pi: [internal use only] necessary for the reestimation of
        mixtures as components
        @param mix_posterior:[internal use only] necessary for the reestimation of
        mixtures as components
        @param EStepParam: additional paramenters for more complex EStep implementations, in
        this implementaion it is ignored

        @return: tuple of log likelihood matrices and sum of log-likelihood of components
        """
        prior_pos = EStepParam.prior_positive
        prior_neg = EStepParam.prior_negative
        norm = EStepParam.normaliziering
        prior_type = EStepParam.prior_type
        # HACK - I am updating at each iteration the previous posterior,
        # since this parameter is not part of of regular EStep
        previous_posterior = EStepParam.previous_posterior
        positive_constraints = data.pairwisepositive
        negative_constraints = data.pairwisenegative

        log_l = numpy.zeros((self.G, data.N), dtype='Float64')
        log_col_sum_nopen = numpy.zeros(data.N, dtype='Float64')  # array of column sums of log_l without penalty
        log_col_sum = numpy.zeros(data.N, dtype='Float64')  # array of column sums of log_l
        log_pi = numpy.log(self.pi)  # array of log mixture coefficient

        # computing log posterior distribution
        for i in range(self.G):
            log_l[i] = log_pi[i] + self.components[i].pdf(data)

        #log_col_sum_nopen[i] = sumlogs(log_l[:,i]) # sum over jth column of log_l without penalization

        #changing order of indices assigments
        indices = range(data.N)

        random.shuffle(indices)

        pen = numpy.zeros(self.G, dtype='Float64')
        penn = numpy.zeros(self.G, dtype='Float64')

        for x, i in enumerate(indices):

        #          print '\n----- sample'+str(i)
        #          print 'log_l=',log_l[:,i]
        #          print 'norm l', numpy.exp( log_l[:,i] - sumlogs(log_l[:,i]) )

            # Leaves -Inf values unchanged
            pen[:] = 0.0
            penn[:] = 0.0
            for y, j in enumerate(indices):

                #print x,y,'->',i,j

                # calculating penalities
                # in a Gibbs sampling manner (using either previous or current posteriors
                if y > x: # if posterior not yet calculated, use of previous one posterior
                    coc = numpy.multiply(previous_posterior[:, j], 1.0) # posterior of y

                    if prior_type == 1 or prior_type == 3:
                        if positive_constraints[i][j] > 0.0:
                            if norm:
                                pen += numpy.divide(numpy.multiply(1 - coc, positive_constraints[i][j]), ((1 - self.pi)))
                            else:
                                pen += numpy.multiply(1 - coc, positive_constraints[i][j])

                            #                      print '   +  1 - coc',j, 1 - coc
                            #                      print '   +',pen

                    if prior_type == 2 or prior_type == 3:
                        if negative_constraints[i][j] > 0.0:
                            if norm:
                                penn += numpy.divide(numpy.multiply(coc, negative_constraints[i][j]), self.pi)
                            else:
                                penn += numpy.multiply(coc, negative_constraints[i][j])

                            #                      print '   - coc',j, coc
                            #                      print '   -',penn


                elif y < x:
                    coc = numpy.multiply(numpy.exp(log_l[:, j]), 1)

                    if prior_type == 1 or prior_type == 3:
                        if positive_constraints[i][j] > 0.0:
                            if norm:
                                pen += numpy.divide(numpy.multiply(1 - coc, positive_constraints[i][j]), ((1 - self.pi)))
                            else:
                                pen += numpy.multiply(1 - coc, positive_constraints[i][j])

                            #                      print '   + coc',j, coc
                            #                      print '   +',pen

                    if prior_type == 2 or prior_type == 3:
                        if negative_constraints[i][j] > 0.0:
                            if norm:
                                penn += numpy.divide(numpy.multiply(coc, negative_constraints[i][j]), self.pi)
                            else:
                                penn += numpy.multiply(coc, negative_constraints[i][j])

                            #                        print '   - coc',j, coc
                            #                        print '   -',penn

                            #print '\n-------', i
                            #print log_l[:,i]
                            #          print - numpy.multiply(pen,prior_pos)
                            #          print - numpy.multiply(penn,prior_neg)

            log_l[:, i] += (-numpy.multiply(pen, prior_pos) - numpy.multiply(penn, prior_neg))
            # l[k,i] = log( (a_k * * P[seq i| model k]) + P[W+|y] * P[W-|y] )

            #          print '-> log_l=',log_l[:,i]
            #          print '-> norm l=',numpy.exp( log_l[:,i] - sumlogs(log_l[:,i]) )

            #print '->',log_l[:,i]

            log_col_sum[i] = sum_logs(log_l[:, i]) # sum over jth column of log_l
            # if posterior is invalid, check for model validity
            if log_col_sum[i] == float('-inf'):

                # if self is at the top of hierarchy, the model is unable to produce the
                # sequence and an exception is raised. Otherwise normalization is not necessary.
                if mix_posterior is None and not mix_pi:
                    print "\n---- Invalid -----\n", self, "\n----------"
                    #print "\n---------- Invalid ---------------"
                    print "mix_pi = ", mix_pi
                    #print "x[",i,"] = ", sequence[j]
                    print "l[:,", i, "] = ", log_l[:, i]
                    raise InvalidPosteriorDistribution, "Invalid posterior distribution."
            # for valid posterior, normalize and go on
            else:
                # normalizing log posterior
                log_l[:, i] = log_l[:, i] - log_col_sum[i]

                # adjusting posterior for lower hierarchy mixtures
                if mix_posterior is not None:
                    log_mix_posterior = numpy.zeros(len(mix_posterior), dtype='Float64')
                    for k in range(len(mix_posterior)):
                        if mix_posterior[k] == 0.0:
                            log_mix_posterior[k] = float('-inf')
                        else:
                            log_mix_posterior[k] = numpy.log(mix_posterior[k])

                    # multiplying in the posterior of upper hierarch mixture
                    log_l[:, i] = log_l[:, i] + log_mix_posterior[i]

        # final penalty (P(W|Y))
        penalty = 0.0
        for x, i in enumerate(indices):
            for y, j in enumerate(indices):
                coc = numpy.multiply(numpy.exp(log_l[:, j]), 1)
                if prior_type == 1 or prior_type == 3:
                    if positive_constraints[i][j] > 0.0:
                        if norm:
                            penalty += numpy.sum(numpy.divide(numpy.multiply(1 - coc, positive_constraints[i][j]), ((1 - self.pi))))
                        else:
                            penalty += numpy.sum(numpy.multiply(1 - coc, positive_constraints[i][j]))
                if prior_type == 2 or prior_type == 3:
                    if negative_constraints[i][j] > 0.0:
                        if norm:
                            penalty += numpy.sum(numpy.divide(numpy.multiply(coc, negative_constraints[i][j]), self.pi))
                        else:
                            penalty += numpy.sum(numpy.multiply(coc, negative_constraints[i][j]))

        # HACK - I am updating at each iteration the previous posterior,
        #since this parameter is not part of of regular EStep
        EStepParam.previous_posterior = numpy.exp(log_l)
        # computing data log likelihood as criteria of convergence
        log_p = numpy.sum(log_col_sum) + penalty
        return log_l, numpy.sum(log_col_sum) + penalty


    def randMaxEM(self, data, nr_runs, nr_steps, delta, prior_positive,
        prior_negative, prior_type, tilt=0, silent=False):
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
                    previous_posterior = candidate_model.modelInitialization(data, prior_positive, prior_negative, prior_type, )    # randomizing parameters of the model copy
                except InvalidPosteriorDistribution:
                    pass
                else:
                    init = 1

            try:
            #                                 def EM(self, data, max_iter, delta, prior_positive, prior_negative, previous_posterior, prior_type, normaliziering=False, silent = False,  mix_pi=None, mix_posterior= None, tilt = 0):
                (l, log_p) = candidate_model.EM(data, nr_steps, delta, prior_positive, prior_negative, previous_posterior, prior_type, silent=silent)  # running EM
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
                    best_l = copy.copy(l)
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

        return best_l, best_logp  # return final data log likelihood


def CMMfromMM(mm):
    """
    Convenience function. Takes a MixtureModel or and returns a ConstrainedMixtureModel with
    the same parameters.

    @param mm: MixtureModel object
    """
    return ConstrainedMixtureModel(mm.G, mm.pi, mm.components, mm.compFix, mm.struct)


def LMMfromMM(mm):
    """
    Convenience function. Takes a MixtureModel or and returns a LabeledMixtureModel with
    the same parameters.

    @param mm: MixtureModel object
    """
    return LabeledMixtureModel(mm.G, mm.pi, mm.components, mm.compFix, mm.struct)
