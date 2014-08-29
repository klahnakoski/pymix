from core.distributions.prob import ProbDistribution


class PriorDistribution(ProbDistribution):
    """
    Prior distribution base class for the Bayesian framework
    """
    def pdf(self, m):
        """
        Returns the log-density of the ProbDistribution object(s) 'm' under the
        prior.

        @param m: single appropriate ProbDistribution object or list of ProbDistribution objects
        """
        raise NotImplementedError, "Needs implementation"

    def posterior(self,m,x):
        raise NotImplementedError, "Needs implementation"

    def marginal(self,x):
        raise NotImplementedError, "Needs implementation"

    def mapMStep(self, dist, posterior, data, mix_pi=None, dist_ind = None):
        """
        Maximization step of the maximum aposteriori EM procedure. Reestimates the distribution parameters
        of argument 'dist' using the posterior distribution, the data and a conjugate prior.

        MUST accept either numpy or DataSet object of appropriate values. numpys are used as input
        for the atomar distributions for efficiency reasons.

        @param dist: distribution whose parameters are to be maximized
        @param posterior: posterior distribution of component membership
        @param data: DataSet object or 'numpy' of samples
        @param mix_pi: mixture weights, necessary for MixtureModels as components.
        @param dist_ind: optional index of 'dist', necessary for ConditionalGaussDistribution.mapMStep (XXX)
        """
        raise NotImplementedError


    def mapMStepMerge(self, group_list):
        """
        Computes the MAP parameter estimates for a candidate merge in the structure
        learning based on the information of two CandidateGroup objects.

        @param group_list: list of CandidateGroup objects
        @return: CandidateGroup object with MAP parameters
        """
        raise NotImplementedError

    def mapMStepSplit(self, toSplitFrom, toBeSplit):
        """
        Computes the MAP parameter estimates for a candidate merge in the structure
        learning based on the information of two CandidateGroup objects.

        @return: CandidateGroup object with MAP parameters

        """
        raise NotImplementedError

    def updateHyperparameters(self, dists, posterior, data):
        """
        Update the hyperparameters in an empirical Bayes fashion.

        @param dists: list of ProbabilityDistribution objects
        @param posterior: numpy matrix of component membership posteriors
        @param data: DataSet object
        """
        raise NotImplementedError
