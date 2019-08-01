import numpy as np
from .basic import scRefData, cellDataFilter, geneDataFilter

"""
after reading data from file
before normalization
"""


class minGeneCellfilter(cellDataFilter):
    """
    remove cell with too little detected genes
    """
    def __init__(self, min_gene_num=10):
        self.process = "remove Cells with little detected genes"
        self._min_gene_num = min_gene_num

    def __call__(self, sco):
        bm = sco.expression_matrix > 0
        c = bm.sum(axis=0)
        f = c > self._min_gene_num
        self.cell_filter = np.squeeze(np.copy(f))
        return self._proc_o(sco)


class minCellGenefilter(geneDataFilter):
    """
    remove gense witch expressed only in a small number of cells
    """
    def __init__(self, min_cell_num=10):
        self.process = "remove Genes detected too little"
        self._min_cell_num = min_cell_num

    def __call__(self, sco):
        bm = sco.expression_matrix > 0
        c = bm.sum(axis=1)
        f = c > self._min_cell_num
        self.gene_filter = np.squeeze(np.copy(f))
        return self._proc_o(sco)


class genesPercentageCellfilter(cellDataFilter):
    """
    filter for mitochondria/ERCC gene percentage
    give genes list and threshold
        or consider using some modle
    """
    def __init__(self, gene_list, threshold):
        self.process = "remove cell with gene expression percentage"
        self.gene_list = gene_list
        self.threshold = threshold

    def __call__(self, sco):
        ti = sco.gene_to_index(self.gene_list)
        bm = sco.expression_matrix > 0
        c = bm.sum(axis=0)
        ct = bm[ti].sum(axis=0)
        p = ct / c
        self.cell_filter = p < self.threshold
        return self._proc_o(sco)


class manualCellSelecter(cellDataFilter):
    """
    for manually select cell
    """
    def __init__(self, mfilter):
        self.process = "manually select cell"
        self.mfilter = mfilter

    def __call__(self, sco):
        f = np.zeros(sco.cell_num, dtype='bool')
        f[self.mfilter] = True
        self.cell_filter = f
        return self._proc_o(sco)
