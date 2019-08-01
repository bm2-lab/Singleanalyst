import numpy as np
from sklearn.metrics import cohen_kappa_score

from .vis import plot_tniann
import time

# plot for data qc

# plot for cell expression
#   单个细胞的基因表达分布，用来对比norm的效果？


# 评估ref-cluster效果
# cohen_kappa_score for cluster-ref
# cell type query evaluation
# cluster to cluster?
def test_ref_cluster(qx, tl, refind):
    pl = refind.get_p_labels(qx)
    return cohen_kappa_score(pl, tl)


# 评估ann效果
# IoU for ann
# nn in ann plot for ann q
def get_IoU(tnn, pnn, k=None):
    """
    nn: index for knn
    """
    h, w = tnn.shape
    h1, w1 = pnn.shape
    # assert h == h1 and w == w1, "data not match"
    IoU = np.zeros(h)
    
    for i in np.arange(h):
        i_i = np.intersect1d(tnn[i], pnn[i])
        u_i = np.union1d(tnn[i], pnn[i])
        if k is None:
            IoU[i] = len(i_i) / len(u_i)
        else:
            IoU[i] = len(i_i) / k
        
    return IoU


def get_nniknn_acc(tn, pnn):
    """
    tn: the ture 1st nn
    pnn: knn index
    """
    h = tn.shape[0]
    h1, w1 = pnn.shape
    assert h == h1, "data not match"
    nin = np.zeros(h, dtype='bool')
    for i in np.arange(h):
        nin[i] = tn[i] in pnn[i]
    acc = nin.sum() / h
    return acc


def test_nniknn(rangek, qx, tn, ref_index_o):
    """
    test query with different k
    """
    h = tn.shape[0]
    h1, w = qx.shape
    assert h == h1, "data not match"
    res = np.zeros(rangek)
    pnni, pnnd = ref_index_o.get_knn(qx, rangek)
    for i, k in enumerate(np.arange(rangek)):
        res[i] = get_nniknn_acc(tn, pnni[:, :i + 1])

    return res


def test_nikn_mulit(rangek, qx, tn, ref_index_o_list):
    h = tn.shape[0]
    rlist = []
    mlist = []
    for o in ref_index_o_list:
        start =time.clock()
        rlist.append(test_nniknn(rangek, qx, tn, o))
        mlist.append(o.__class__)
        end = time.clock()
        print('Running time: %s Seconds'%(end-start))
    plot_tniann(rangek, rlist, mlist)
    return rlist
