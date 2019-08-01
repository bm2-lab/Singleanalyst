import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
import faiss
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections

from .utils import PCA, get_knn_hamming
from .basic import scBase, scRefData
from .cluster import cellCluster 
from .process import get_2dembedding
from .vis import plot_2demb_nn
# 考虑faiss的实现
BIG_LIMIT = np.Infinity


class indexRef(scBase):
    """
    build reference from given dataset
    """
    def __init__(self, sco, nn, metrics=euclidean_distances):
        self.le = sco.le
        self.labels = sco.labels
        X = np.ascontiguousarray(sco.expression_matrix.T)
        self.nn = nn
        self.nn.fit(X)
        self.ref_X = X
        self.m = metrics
        self.meta_info = sco.meta_info
        self.gene_ref = sco.gene_ref

        self.emb_X = None

    def get_predict(self, qxm, k=10, method=0):
        """
        knn type prediction
        find knn and nn beased label assain
            pre-compute inner distence ?
        """
        h, _ = qxm.shape
        x_knni, x_knnd = self.nn.get_knn(qxm, k)
        l_nn = self.labels[x_knni]
        ul = [np.unique(i, return_counts=True) for i in l_nn]

        res = np.zeros(h)
        if method == 0:
            for i, u in enumerate(ul):
                i_c = u[1]
                i_u = u[0]
                if i_c.max() / i_c.sum() > 0.75:
                    res[i] = i_u[i_c.argmax()]
                else:
                    res[i] = -1

        res_l = self.le.inverse_transform(res[res!=-1].astype("int"))
        # use dtype=object and get arbitrary length strings 
        res_n = np.empty_like(res, dtype=object)

        res_n[res!=-1] = res_l
        res_n[res==-1] = 'unassigned'
        return res_n

    def get_knn_vis(self, qx, k=10, **karg):
        if self.emb_X is None:
            self.emb_X = get_2dembedding(self.ref_X, **karg)
        x_knni, _ = self.nn.get_knn(qx.reshape(1,-1), k)   
        types = self.le.inverse_transform(range(max(self.labels)+1))
        return plot_2demb_nn(self.emb_X, x_knni[0], self.labels, types)



class NNs(object):
    """
    the basic nns object
    """
    def __init__(self):
        """
        init for fit
        """
        raise NotImplementedError

    def fit(self, X):
        raise NotImplementedError

    def get_knn(self, qxm, k=3):
        """
        qxm with shape h, w;
        get knn in (m1, m2) with shape ((h,k), (h k))
            (m1, m2) is index and distance matrix
        """
        raise NotImplementedError


class directNN(NNs):
    """
    use sklearn.metrics.pairwise
    direct comput nn
    """
    def __init__(self, metrics=euclidean_distances):
        self.params = {
            "method": "direct pairwise comput",
            "metrics": metrics
        }

    def fit(self, X):
        self.X = X
        self.m = self.params["metrics"]

    def get_knn(self, qxm, k=3):
        h, w = qxm.shape
        d = self.m(qxm, self.X)
        di = np.argsort(d, axis=1)
        return (di[:, 0:k], d[np.arange(h)[:, np.newaxis], di[:, :k]])


class pqNN(NNs):
    """
    product quantization
    """
    def __init__(self, m=None, k=None):
        """
        n: sample number
        f: feature number
        """
        self.params = {
            "method": "product quantization, numpy",
            'm': m,
            'k': k
        }

    def fit(self, X):
        """
        X: n * f data
        """
        self.m = self.params['m']
        self.k = self.params['k']
        self.n, self.f = X.shape
        if self.m is None:
            self.m = min(np.int(np.ceil(self.f / 10)), 100)
        if self.k is None:
            self.k = max(np.int(np.sqrt(self.n)), 2)
        # split feature space
        chunks_i = [int(i * self.f / (self.m)) for i in range(self.m + 1)]
        self.ci_list = \
            [np.arange(chunks_i[i], chunks_i[i + 1]) for i in range(self.m)]
        self.cluster_centers = np.zeros((self.f, self.k))
        self.predict_labels = np.zeros((self.m, self.n), dtype='int')
        for i in range(self.m):
            x_i = X[:, self.ci_list[i]]
            km = KMeans(n_clusters=self.k, n_jobs=-2)
            km.fit(x_i)
            self.predict_labels[i, :] = km.labels_
            self.cluster_centers[self.ci_list[i], :] = \
                km.cluster_centers_.T

    def get_knn(self, qxm, k=3):
        h, w = qxm.shape
        if k > self.n or w > self.f:
            raise Exception("Invalid input")
        cosm = np.zeros((h, self.n))
        dist = np.zeros((self.m, h, self.k))

        for mi in range(self.m):
            qxm_i = qxm[:, self.ci_list[mi]]
            cc_i = self.cluster_centers[self.ci_list[mi], :].T
            pl_i = self.predict_labels[mi]
            d_v = euclidean_distances(qxm_i, cc_i)
            dist[mi, :, :] = d_v**2
        for ni in range(self.m):
            k_i = self.predict_labels[ni, :]
            cosm += dist[ni, :, :][:, k_i]
        cosm = np.sqrt(cosm)

        di = cosm.argsort()
        return (di[:, :k], cosm[np.arange(h)[:, np.newaxis], di[:, :k]])


class lshNN(NNs):
    """
    Locality-sensitive hashing by random projection
        consider some options
    nearpy implementation
    """
    def __init__(self, b=16):
        self.params = {
            "method": "product quantization, numpy",
            'b': b
        }

    def fit(self, X):
        b = self.params['b']
        self.n, self.f = X.shape
        # Use NearPy lsh for fast ann
        rbp = RandomBinaryProjections('rbp', b)

        self.engine = Engine(self.f, lshashes=[rbp])
        for i in np.arange(self.n):
            v = np.squeeze(np.copy(X[i, :]))
            self.engine.store_vector(v, i)

    def _get_one_knn(self, v, k=3):
        v = np.squeeze(np.copy(v))
        vl = v.shape
        if vl[0] != self.f:
            # print(vl)
            raise Exception("Data Not Match")
        N = self.engine.neighbours(v)
        nni = -np.ones(k, dtype='int')
        nnd = np.empty(k)
        nnd[:] = np.nan
        for i in np.arange(k):
            try:
                nni[i] = N[i][1]
                nnd[i] = N[i][2]
            except IndexError:
                break
        return (nni, nnd)

    def get_knn(self, x, k=3):
        self.n, self.f = x.shape
        nni = -np.ones((self.n, k), dtype='int')
        nnd = np.empty((self.n, k))
        nnd[:] = np.nan
        for i in np.arange(self.n):
            i_i, i_d = self._get_one_knn(x[i, :], k)
            nni[i, :] = i_i
            nnd[i, :] = i_d
        return (nni, nnd)

class faiss_baseline_nn(NNs):
    """
    warper around faiss
    brute-force L2 distance search with faiss
    """
    def __init__(self):
        self.params = {
            "method": "brute-force L2, faiss"
        }

    def fit(self, X):
        h, w = X.shape
        self.index = faiss.IndexFlatL2(w)
        self.index.add(X.astype('float32'))

    def get_knn(self, qxm, k=3):
        D, I = self.index.search(qxm.astype('float32'), k)
        return (I, D)


class faiss_IVF_nns(NNs):
    """
    warper around faiss
    with different method
    """
    def __init__(self, nlist=100, nprobe=10):
        """
        Faster search
        nlist: num for Voronoi cells
        nprobe: num of cell used for search
        """
        self.params = {
            "method": "IVFFlat, faiss",
            'nlist': nlist,
            'nprobe': nprobe
        }

    def fit(self, X):
        nlist = self.params['nlist']
        nprobe = self.params['nprobe']
        h, w = X.shape
        # although not explicit called, quantizer need in
        # object name space
        self.quantizer = faiss.IndexFlatL2(w)
        self.index = faiss.IndexIVFFlat(
            self.quantizer, w, nlist, faiss.METRIC_L2)
        self.index.train(X.astype('float32'))

        self.index.add(X.astype('float32'))
        self.index.nprobe = nprobe

    def get_knn(self, qxm, k=3):
        D, I = self.index.search(qxm.astype('float32'), k)
        return (I, D)


class faiss_IVFPQ_nns(NNs):
    """
    Use Product Quantizer for lower memory usage
    """
    def __init__(self, nlist=100, nprobe=10, m=8, b=8):
        """
        nlist: num for Voronoi cells
        nprobe: num of cell used for search
        m: number of bytes per vector
        b: sub-vector encode
        """
        self.params = {
            "method": "IVFPQ, faiss",
            'nlist': nlist,
            'nprobe': nprobe,
            'm': m,
            'b': b
        }

    def fit(self, X):
        nlist = self.params['nlist']
        nprobe = self.params['nprobe']
        m = self.params['m']
        b = self.params['b']
        h, w = X.shape
        d = int((w + m - 1) / m) * m
        self.remapper = faiss.RemapDimensionsTransform(w, d, True)
        self.quantizer = faiss.IndexFlatL2(d)
        self.index_pq = faiss.IndexIVFPQ(self.quantizer, d, nlist, m, b)
        self.index = faiss.IndexPreTransform(
            self.remapper, self.index_pq)
        self.index.train(X.astype('float32'))

        self.index.add(X.astype('float32'))
        self.index.nprobe = nprobe

    def get_knn(self, qxm, k=3):
        D, I = self.index.search(qxm.astype('float32'), k)
        return (I, D)
