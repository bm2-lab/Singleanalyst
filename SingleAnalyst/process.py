import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from scipy import stats
import copy

from .basic import singleCellData, baseTool, infoTable
from .vis import plot_2demb, plot_2demb_labels

"""
PCA, t-SNE,and DEtest
"""


class manualDataSplit(object):
    """
    gen test train data pair
    """
    def __init__(self, mfilter):
        self.mfilter = mfilter

    def _gen_new_o(self, sco):
        
        _em = copy.deepcopy(sco.expression_matrix[:, self.cell_filter])
        _gr = copy.deepcopy(sco.gene_ref)
        _ci = copy.deepcopy(sco.cell_info.get_filtered(self.cell_filter))
        _mi = copy.deepcopy(sco.meta_info)
        return singleCellData(_em, _gr, _ci, _mi)

    def __call__(self, sco):
        f = np.zeros(sco.cell_num, dtype='bool')
        f[self.mfilter] = True
        self.cell_filter = f
        return self._gen_new_o(sco)


# train_test_split for sco
def tt_split(sco, **options):
    x = np.arange(sco.cell_num)
    x_train, x_test = train_test_split(x, **options)
    train = manualDataSplit(x_train)
    test = manualDataSplit(x_test)
    return (train(sco), test(sco))


def add_mimic_data(sco, num_add=0):
    _em = copy.deepcopy(sco.expression_matrix)
    _gr = copy.deepcopy(sco.gene_ref)
    _ci = copy.deepcopy(sco.cell_info)
    _mi = copy.deepcopy(sco.meta_info)
    # ['cell_list', 'cell_type']
    # work at very begining
    _cell_list = copy.deepcopy(_ci.get_data_byname('cell_list'))
    _cell_type = copy.deepcopy(_ci.get_data_byname('cell_type'))

    _nf, _nc = _em.shape

    _o_cell = np.array(_cell_type, dtype='<S8')
    _o_len = np.arange(_nc)

    _add_m = np.zeros((_nf, num_add), dtype='float')

    for i in range(num_add):
        _p = np.random.choice(_o_len)
        _p_s = _o_cell[_p]
        _p_t_m = _o_cell == _p_s
        _p2 = np.random.choice(_o_len[_p_t_m])
        _c1 = np.squeeze(np.array(_em[:, _p]))
        _c2 = np.squeeze(np.array(_em[:, _p2]))
        _mix = np.random.random_sample(_nf)
        _mi_c = (_c1*_mix) + (_c2*(1-_mix))

        _add_m[:, i] = _mi_c

        _mi_n = "mimic_" + str(i)

        _cell_list.append(_mi_n)
        _cell_type.append(_p_s)

    _cell_info = infoTable(
        ['cell_list', 'cell_type'],
        [_cell_list, _cell_type])
    _a_em = np.append(_em, _add_m, axis=1)

    return singleCellData(_a_em, _gr, _cell_info, _mi)






# interactive making cluster
#     notice: X below should be em.T

#     from .vis import plot_2demb, plot_2demb_labels
#     for visual inspect

class data_embeding(baseTool):
    def __init__(self, maxf=100, uesd=50, **options):
        """
        **options: options for tsne
        """
        self.process = "t-SNE embeding"
        self.tsne = TSNE(TSNE(n_components=2, **options))
        self.maxf = 100
        self.used = 50

    def __call__(self, sco):
        X = sco.expression_matrix.T
        _, w = X.shape
        if w > self.maxf:
            pca = PCA(n_components=self.used)
            X = pca.fit_transform(X)
        X = self.tsne.fit_transform(X)
        x_list = [[i[0], i[1]] for i in X]
        sco.cell_info.add_data('2d_embeding', x_list)


class data_cluster(baseTool):
    def __init__(self, **options):
        """
        ** options: options for DBSCAN
        use cell_info['2d_embeding']
        """
        self.process = "DBSCAN on embeded space"
        self.dbscan = DBSCAN(**options)

    def __call__(self, sco):
        try:
            e_X = sco.cell_info['2d_embeding']
        except KeyError:
            raise Exception("Need 2d embeding frist")
        e_X = np.array(e_X)
        self.dbscan.fit(e_X)
        pl = self.dbscan.labels_

        sco.cell_info.add_data('predicted_cluster', pl)


def get_2dembedding(X, **options):
    """
    pca, t-sne get 2d embedding for visual inspecting
    """
    _, w = X.shape
    # use PCA frist if dim is high
    # 50 by default
    if w > 100:
        pca = PCA(n_components=50)
        X = pca.fit_transform(X)
    # t-sne on features selected data
    X = TSNE(n_components=2, **options).fit_transform(X)
    return X


def get_cluster(X, **options):
    """
    DBSCAN clustering warper;
        for extending
    """
    db = DBSCAN(**options).fit(X)
    return db.labels_


def make_2dembeding(sco, **options):
    """
    work with sco
        expression_matrix and X have different rotation
    """
    X = sco.expression_matrix.T
    X = get_2dembedding(X, **options)
    plt = plot_2demb(X)
    return (X, plt)


def make_clusters(x_em, **options):
    """
    make cluster within embeded 2d space
    """
    cl = get_cluster(x_em, **options)
    plt = plot_2demb_labels(x_em, cl)
    return (cl, plt)


def find_de_tt(l, M, nmarkers):
    """
    find marker gene for cluster;
    using only t-test for now,
    finder markers pairwisely;
        consider add more method later
    input:
        l, cluster labels should be int;
        expression matrix, M
        number of markers, nmarkers
    output:
        marker genes index for each cluster
    """
    M = M.T
    labels = np.unique(l)
    mai = {}
    mas = {}
    ma = {}
    for i in labels:
        mai_i = []
        mas_i = []
        for j in labels:
            if i == j:
                continue
            id1 = l == i
            id2 = l == j
            d1 = M[id1, :]
            d2 = M[id2, :]
            mi_ij, ms_ij = _find_makers_twosample(d1, d2)
            mai_i.append(mi_ij)
            mas_i.append(ms_ij)
        mai[i] = mai_i
        mas[i] = mas_i
        ma[i] = _merge_m(mai_i, nmarkers)
    l_merged = [ma[i] for i in labels]
    mergedres = _merge_m(l_merged, nmarkers)
    return mergedres


def _find_makers_twosample(data1, data2, n=None, p=0.05):
    """
    simple warper for ttest_ind;
    n, the uplimit of deg found
    """
    _, w1 = data1.shape
    _, w2 = data2.shape
    assert w1 == w2, "data not match"
    res = np.zeros((w1, 3))
    for i in np.arange(w1):
        # since we work on logtrans data
        d1_i = np.exp(data1[:, i])
        d2_i = np.exp(data2[:, i])
        t_s, p = stats.ttest_ind(d1_i, d2_i, equal_var=False)
        f_c = np.mean(d1_i) / np.mean(d2_i)
        # use 2**log2(fc) for fold change
        log2fc = np.exp2(np.abs(np.log2(f_c)))
        res[i, :] = [t_s, p, log2fc]
    pcheck = res[:, 1] < 0.05
    ssi = np.argsort(-np.abs(res[:, 2]))
    mi = ssi[pcheck]
    if n is not None:
        if len(mi) < n:
            print("Not find enough genes")
            n = len(mi)
        mi = mi[: n]
    return (mi, res[mi, :])


def _merge_m(ilist, n):
    """
    merge pairwise result
    """
    all_ind = np.unique(np.concatenate(ilist))
    if all_ind is None or len(all_ind) < n:
        print("not enough DEgenes")
        return all_ind
    res = []
    ra = 0
    ni = 0
    run = True
    while run:
        r = np.unique([i[ra] for i in ilist])
        for i in r:
            if i in res:
                continue
            else:
                ni += 1
                res.append(i)
                if ni >= n:
                    run = False
        ra += 1
    return res


def find_de_anova(l, M, nmarkers):
    """
    anova for groups
    find marker gene for cluster;
    input:
        l, cluster labels should be int;
        expression matrix, M
        number of markers, nmarkers
    output:
        marker genes index for each cluster
    """
    M = M.T
    labels = np.unique(l)
    _, gene_n = M.shape
    mask_list = [l == i for i in labels]
    F_values = np.zeros(gene_n)
    p_values = np.zeros(gene_n)
    gene_index = np.zeros(gene_n)
    for i in range(gene_n):
        d_i = np.squeeze(M[:, i])
        e_i_l = [d_i[m] for m in mask_list]
        F_values[i], p_values[i] = stats.f_oneway(e_i_l)
    
    pass_p = p_values <= 0.05
    p_values = p_values[pass_p]
    F_values = F_values[pass_p]

    sort_index = np.argsort(-F_values)
    p_values = p_values[sort_index]
    F_values = F_values[sort_index]
    gene_index = gene_index[pass_p][sort_index]

    return (gene_index, F_values, p_values)

