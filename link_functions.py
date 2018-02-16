import numpy as np


class LinkFunction(object):
    def __init__(self):
        pass

    def __call__(self, args):
        pass

    def inv(self):
        """
        Returns the inverse function
        """
        pass

    def deriv(self):
        """
        Returns another function object representing the derivative of the
        function
        """
        pass

    def inv_deriv(self):
        """
        Returns derivative of the inverse.
        """
        pass

class IDLink(LinkFunction):

    def __call__(self, args):
        return args.copy()

    def inv(self):
        return self

    def deriv(self):
        def f(args):
            return np.ones(args.shape)
        return f

    def inv_deriv(self):
        def f(args):
            return np.ones(args.shape)
        return f

class LogLink(LinkFunction):

    def __call__(self, args):
        # TODO: deal with sparse?
        return np.log(args)

    def inv(self):
        return lambda x: np.exp(x)

    def deriv(self):
        def f(args):
            return 1.0/args
        return f

    def inv_deriv(self):
        return lambda x: np.exp(x)
