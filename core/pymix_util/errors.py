
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


