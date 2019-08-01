import pandas as pd
import numpy as np

from .basic import scBase, singleCellData, baseTool
from .index import indexRef, faiss_baseline_nn, faiss_IVF_nns, faiss_IVFPQ_nns, lshNN
from .cluster import cellCluster


class queryData(singleCellData):
    def __init__(self, sco):
        import copy
        expression_matrix = copy.deepcopy(sco.expression_matrix)
        gene_ref = copy.deepcopy(sco.gene_ref)
        cell_info = copy.deepcopy(sco.cell_info)
        proc = copy.deepcopy(sco.processed)
        meta_info = copy.deepcopy(sco.meta_info)
        super(queryData, self).__init__(expression_matrix, gene_ref, cell_info, meta_info, proc)

    def get_qxm(self, gene_list):
        genes = self.gene_ref.get_list()
        expr = self.expression_matrix.T

        df_o = pd.DataFrame(expr, columns=genes)
        df_r = pd.DataFrame(columns=gene_list)

        md = pd.concat([df_o, df_r], axis=0, sort=False)

        df_q = md[gene_list]

        return np.ascontiguousarray(np.array(df_q.fillna(0.0)))


class doQuery(baseTool):
    def __init__(self, index, query_name, **karg):
        """
        index: index for query
        """
        self.process = 'query indexed reference'
        self.index = index
        self.query_name = query_name

    def __call__(self, sco):
        pred = self.index.query(sco)
        for i, p in enumerate(pred):
            i_name = '{}_{}'.format(i, self.query_name)
            sco.cell_info.add_data(i_name, p)


class clusterIndex(object):
    """
    for cell cluster ref index
    * collect informative genes?
    """
    def __init__(self):
        self.cluster_list = []

    def add_dataset(self, cell_cluster):
        self.cluster_list.append(cell_cluster)

    def query_all(self, qx, **karg):
        """
        search one by one
        *visualization ?
        """
        result = []
        for i, _ in enumerate(self.cluster_list):
            result.append(self.query_one(i, qx, **karg))
        return result

    def query_one(self, index, qx, **karg):
        """
        search in given dataset
        using original implements
        """
        clu = self.cluster_list[index]
        rg = clu.gene_ref.get_list()
        qxm = qx.get_qxm(rg)

        return clu.get_p_labels(qxm, **karg)

    def summary(self):
        d = {}
        d['index'] = range(len(self.cluster_list))
        d['dataset'] = [i.meta_info['name'] for i in self.cluster_list]
        d['type_num'] = [i.index_type_num for i in self.cluster_list]

        return pd.DataFrame(data=d)



class indexCollection(object):
    """
    collection of cluster for ref search
    """
    def __init__(self, **karg):
        self.datasets = []

    def add_dataset(self, index_ref):
        """
        add dataset to database for search
        """
        # *add check for Refdata
        self.datasets.append(index_ref)

    def query(self, sco, **arg):
        result = []
        for i, _ in enumerate(self.datasets):
            result.append(self.query_one(i, sco, **arg))
        return result

    def query_one(self, index, qx, k=3):
        """
        search one dataset
        """
        ref = self.datasets[index]
        ref_gene = ref.gene_ref.get_list()
        qxm = qx.get_qxm(ref_gene)

        return ref.get_predict(qxm, k=k)

    def query_all(self, qx, k=3):
        """
        search one by one
        and summary
        """
        result = []
        for i in range(len(self.datasets)):
            result.append(self.query_one(i, qx, k=k))
        return result
    
    def summary(self):
        """
        show all data set info
        return a list of info array
        format by pandas
        """
        d = {}
        d['index'] = range(len(self.datasets))
        d['dataset'] = [i.meta_info['name'] for i in self.datasets]

        return pd.DataFrame(data=d)

