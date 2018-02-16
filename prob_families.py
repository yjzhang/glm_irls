import numpy as np

# error distribution families for IRLS/GLMs
# the only necessary function is the variance as a function of the mean?

class Dist(object):

    def __init__(self):
        pass

    def var(self, means):
        """
        Returns variance as a function of the means (and
        possibly other parameters).
        """
        pass

class Poisson(object):

    def var(self, means):
        return means.copy()

class Normal(object):

    def __init__(self, v=1.0):
        self.v = v

    def var(self, means):
        return np.ones(means.shape)*self.v
