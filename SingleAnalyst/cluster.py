import numpy as np
from scipy.stats import pearsonr

from .basic import scRefData, scBase, indexedList, infoTable
from .vis import index_heatmap
# get cell type ref by exsit dataset


def _cor_ndarr(xm, ym, m):
    ym = np.copy(ym)
    xm = np.copy(xm)

    def alx(x):
        def aly(y):
            cor = m(x, y)[0]
            if np.isnan(cor):
                print("get nan!")
                return 0
            else:
                return cor

        return np.apply_along_axis(aly, 1, ym)
    return np.apply_along_axis(alx, 1, xm)


class cellCluster(scBase):
    """
    get cell-type ref form data for comparison
    """
    def __init__(
            self, ref_sco,
            threshold=0.75, s_cal=pearsonr, f_cal=np.median):
        """
        ref_sco: processed refdata
        s_cal: similarity calculation method 
        f_cal: feature static calculation method
        """
        self.params = {
            'threshold': threshold,
            's_cal': s_cal,
            'f_cal': f_cal
        }
        self.gene_ref = ref_sco.gene_ref
        self.le = ref_sco.le
        self.labels = ref_sco.labels
        x = ref_sco.expression_matrix
        h, _ = x.shape

        y = ref_sco.labels
        self.index_type_num = len(np.unique(y))
        index_expression = np.zeros((h, self.index_type_num))
        self.index_type = np.zeros(self.index_type_num, dtype='int')

        for i, t in enumerate(np.unique(y)):
            cell_type_mask = y == t
            x_i = x[:, cell_type_mask]
            x_i = np.copy(x_i)

            v = np.apply_along_axis(f_cal, 1, x_i)
            index_expression[:, i] = v
            self.index_type[i] = t

        # filter out all-zores features
        _zf = index_expression.sum(axis=1) != 0
        # print(index_expression)
        self.index_expression = index_expression[_zf, :]
        _fg = self.gene_ref.index_to_value(np.arange(len(_zf))[_zf])
        self.gene_ref = indexedList(_fg)
        self.index_type_name = self.le.inverse_transform(self.index_type)

        self.cell_info = infoTable(['cell_type'], [self.index_type_name])

    def get_p_labels(self, qxm):
        s_cal = self.params['s_cal']
        similarity_p = _cor_ndarr(qxm, self.index_expression.T, s_cal)

        def max_labels(c_s_l):
            return np.argmax(c_s_l, axis=0)
        max_l = np.apply_along_axis(max_labels, 1, similarity_p)
        p_type = self.le.inverse_transform(max_l) 
        p_score = similarity_p[range(len(max_l)), max_l]
        return (p_type, p_score)

    def cluster_heatmap(self):
        plt = index_heatmap(self.index_expression, self.index_type_name)
        return plt

