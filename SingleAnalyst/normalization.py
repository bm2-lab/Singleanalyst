import numpy as np
from .basic import scRefData, dataNormalization
from .utils import build_mapper


"""
before select
after filter
"""


class logtranData(dataNormalization):
    """
    Not normalize data, just apply log1p to data to imporve normality
    """
    def __init__(self):
        self.process = "log1p transformation"

    def normalize(self, expression):
        return np.log1p(expression)


class quantileNormalization(dataNormalization):
    """
    Quantile normalization
    https://en.wikipedia.org/wiki/Quantile_normalization
    consider log trans imporve normality
    """
    def __init__(self, logtran=False):
        self.process = "Quantile normalization"
        self.log = logtran

    def normalize(self, expression):
        h, w = expression.shape
        zm = expression == 0
        spc = np.argsort(-expression, axis=0)
        n_em = np.zeros_like(expression)
        for i in np.arange(h):
            em_i = expression[spc[i, :], np.arange(w)]
            zf = em_i > 0
            me = np.mean(em_i[zf])
            n_em[spc[i, :], np.arange(w)] = me
        n_em[zm] = 0
        if self.log:
            n_em = np.log1p(n_em)
        return n_em


class logNormlization(dataNormalization):
    """
    sum log normlization
    """
    def __init__(self, scalefactor=10000):
        self.process = "sum log normlization"
        self.scalefactor = scalefactor

    def normalize(self, expression):
        total_expression_e_cell = expression.sum(axis=0)

        log_norm_expr = np.log1p(
            expression / total_expression_e_cell *
            self.scalefactor)
        return log_norm_expr


class lmNormlization(dataNormalization):
    """
    linear fit percentile normlization
    """
    def __init__(self, scalefactor=100):
        self.process = "linear fit percentile normlization"
        self.scalefactor = scalefactor

    def normalize(self, expression):
        h, w = expression.shape
        norm_expr = np.zeros_like(expression)
        for i in np.arange(w):
            ex_c = expression[:, i]
            mapper = build_mapper(ex_c)
            norm_expr[:, i] = mapper.mapping(ex_c)
        return norm_expr
