import numpy as np
from .basic import scRefData, featureSelection
from .utils import find_variable_genes, dropout_linear_model
from .process import find_de_tt, find_de_anova
"""
after normalization
befor cluster or nn_indexing
"""


class highlyVarSelecter(featureSelection):
    """
    select highly varable genes;
    """
    def __init__(self, num_features=None):
        self.num_features = num_features
        self.process = "select highly varable genes"

    def __call__(self, sco):
        expression = sco.expression_matrix
        vg = find_variable_genes(expression)
        if self.num_features is not None:
            if len(vg) < self.num_features:
                print("not enough candidate genes")
                self.num_features = len(vg)
            vg = np.random.choice(vg, size=self.num_features, replace=False)
        self.selected_features = vg
        return self._proc_o(sco)


class randomSelecter(featureSelection):
    def __init__(self, num_features=500):
        self.num_features = num_features
        self.process = "select genes randomly"

    def __call__(self, sco):
        h, _ = sco.expression_matrix.shape
        self.selected_features = np.random.choice(
            np.arange(h), size=self.num_features, replace=False)
        return self._proc_o(sco)


class dropOutSelecter(featureSelection):
    """
    linear modle high drop out select
    """
    def __init__(self, num_features=None, large=False):
        self.process = "select genes by dropout"
        self.num_features = num_features
        self.large = large

    def __call__(self, sco):
        expression = sco.expression_matrix
        (s_features, _) = dropout_linear_model(
            expression, self.num_features, self.large)
        self.selected_features = s_features
        return self._proc_o(sco)


class manualSelecter(featureSelection):
    """
    manual select
    give list of genes
    """
    def __init__(self, gene_list):
        self.process = "manual select genes"
        self.gene_list = gene_list

    def __call__(self, sco):
        self.selected_features = sco.gene_to_index(self.gene_list)
        return self._proc_o(sco)


class markerSelecter_tt(featureSelection):
    """
    for labeled data only
    select cluster marker as feature
    """
    def __init__(self, num_features=500):
        self.process = "select genes by cluster marker"
        self.num_features = num_features

    def __call__(self, sco):
        assert hasattr(sco, 'labels'), "noly for labeled data"
        lab = sco.labels
        fr = find_de_tt(lab, sco.expression_matrix, self.num_features)
        self.selected_features = fr
        return self._proc_o(sco)

class markerSelecter_anova(featureSelection):
    """
    for labeled data only
    select cluster marker as feature
    """
    def __init__(self, num_features=500):
        self.process = "select genes by cluster marker"
        self.num_features = num_features

    def __call__(self, sco):
        assert hasattr(sco, 'labels'), "noly for labeled data"
        lab = sco.labels
        fr = find_de_anova(lab, sco.expression_matrix, self.num_features)
        self.selected_features = fr
        return self._proc_o(sco)