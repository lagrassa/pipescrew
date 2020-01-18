import numpy as np
from scipy.stats import multivariate_normal as mvn

class Belief():
    def __init__(self, mu, cov):
        self.mvn = mvn(mean=mu, cov=cov)
    def mean(self):
        return self.mvn.mean
    def cov(self):
        return self.mvn.cov
