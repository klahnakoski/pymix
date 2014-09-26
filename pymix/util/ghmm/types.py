
kNotSpecified = 0
kLeftRight = 1
kSilentStates = pow(2, 2)   # NOT NEEED:  A MARKOV MODEL WITH k OF N STATES BEING Silent IS THE SAME AS MODEL WITH (N - k) STATES
kTiedEmissions = pow(2, 3)
kUntied = -1
kHigherOrderEmissions = pow(2, 4)
kBackgroundDistributions = pow(2, 5)
kNoBackgroundDistribution = -1
kLabeledStates = pow(2, 6)
kTransitionClasses = pow(2, 7)
kDiscreteHMM = pow(2, 8)
kContinuousHMM = pow(2, 9)
kPairHMM = pow(2, 10)
kMultivariate = pow(2, 11)
