import numpy as np
from scipy.spatial import distance
from scipy import stats

import faiss


"""
some test
    highly variable genes
"""

SF = 100


class PCA(object):
    """
    Warper around faiss.PCAMatrix
    """
    def __init__(self, n_components, **kwarg):
        self.npca = n_components

    def fit(self, X):
        _, w = X.shape
        self.ndim = w
        self._pca = faiss.PCAMatrix(w, self.npca)
        # .astype('float32') is needed
        self._pca.train(X.astype('float32'))
        return self

    def fit_transform(self, X):
        self.fit(X.astype('float32'))
        return self._pca.apply_py(X.astype('float32'))

    def transform(self, X):
        return self._pca.apply_py(X.astype('float32'))


def get_knn_hamming(qx, indx, k=3):
    """
    full pairwise compute;
    When dataset is large,
    consider more menmory friendly way;
    """
    d = distance.cdist(qx, indx, metric='hamming')
    sid = np.argsort(d, axis=1)
    return (sid[:, :k], d[:, :k])


class lm(object):
    def __init__(self, x, y, large=True):
        if not large:
            from sklearn.linear_model import LinearRegression as model
        else:
            print("use SGDRegressor")
            from sklearn.linear_model import SGDRegressor as model
        self.regr = model()
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        self.regr.fit(x, y)
        py = self.regr.predict(x).reshape(-1, 1)
        self.residuals = y - py
        self.intercept = self.regr.intercept_
        self.slope = self.regr.coef_
        self.x_0 = self.get_predict_inv(0)

    def get_residuals(self):
        return np.squeeze(np.copy(self.residuals))

    def get_predict(self, x):
        return np.squeeze(np.copy(self.regr.predict(x)))

    def get_predict_inv(self, y):
        px = (y - self.intercept) / self.slope
        return np.squeeze(np.copy(px))

    def summary(self):
        print(
            "slope:{:.3f},\nintercept:{:.3f},\nx_0:{:.3f}".format(
                self.slope, self.intercept, self.x_0))

# scipy implementation
# class lm(object):
#     def __init__(self, x, y):
#         self.x = x
#         self.y = y
#         self.slope, self.intercept, self.r_value, \
#             self.p_value, self.std_err = stats.linregress(x, y)
#         self.x_0 = self.get_predict_inv(0)

#     def summary(self):
#         print(
#             "slope:{:.3f},\nintercept:{:.3f},\nx_0:{:.3f}".format(
#                 self.slope, self.intercept, self.x_0))

#     def get_residuals(self):
#         return self.y - (self.x * self.slope + self.intercept)

#     def get_predict(self, x):
#         return x * self.slope + self.intercept

#     def get_predict_inv(self, y):
#         return (y - self.intercept) / self.slope


class lmc(lm):
    def mapping(self, e_v):
        x_p = self.get_predict_inv(e_v)
        x_m = (x_p - self.x_0) / (SF - self.x_0)
        x_m[x_m > 1.0] = 1.0
        x_m[x_m < 0] = 0
        return x_m


class oneCellExpressionArray(object):
    def __init__(self, c_array):
        self.expression_array = c_array
        self.genes_num = len(c_array)
        self.sorted_index = np.argsort(c_array)
        self.dropout_counts = (c_array == 0).sum()
        self.dropout_rate = self.dropout_counts / self.genes_num


def build_mapper(c_array):
    # expected log transformation of raw data
    cell = oneCellExpressionArray(np.log10(c_array + 1))
    s_num = (1 - cell.dropout_rate) * 10 + 2
    top_cut = int(np.ceil(cell.genes_num * 0.95))
    bottom_cut = int(cell.dropout_counts-1)
    x_interval = np.linspace(0, SF, cell.genes_num)
    top_cut_x = x_interval[top_cut]
    bottom_cut_x = x_interval[bottom_cut]
    s_x = np.linspace(bottom_cut_x, top_cut_x, s_num)
    s_x_i = np.floor(np.linspace(bottom_cut, top_cut, s_num)).astype(int)
    s_ind = cell.sorted_index[s_x_i]
    s_y = cell.expression_array[s_ind]

    mapper = lmc(s_x, s_y)
    return mapper


def dropout_linear_model(counts, n_features, large):
    h, w = counts.shape
    bm = counts == 0
    dropouts = bm.sum(axis=1) / w * 100
    dropouts_filter = (dropouts != 0) & (dropouts != 100)
    dropouts_filter = np.squeeze(np.copy(dropouts_filter))
    counts = counts[dropouts_filter, :]
    dropouts = dropouts[dropouts_filter]
    GiCsum = counts.sum(axis=1)
    # linear fit log counts
    fit = lm(np.log10(dropouts), GiCsum, large)
    residuals = fit.get_residuals()

    r_sort_ind = np.argsort(-residuals)[:n_features]
    s_features = np.arange(h)[dropouts_filter][r_sort_ind]
    s_scores = residuals[r_sort_ind]
    return (s_features, s_scores)


def bins_cut(marray, nbins, methods="equal_number"):
    """
    place element into bins
    return gene in each bin by list
    """
    marray = np.squeeze(np.copy(marray))
    in_array = np.argsort(marray)
    sarray = marray[in_array]

    if methods == "equal_number":
        return np.array_split(in_array, nbins)

    elif methods == "equal_width":
        mmax = np.max(marray)
        mmin = np.min(marray)
        cuta = np.linspace(start=mmin, stop=mmax, num=nbins+1)

        binslist = []
        for i in range(nbins):
            bottom = cuta[i]
            top = cuta[i+1]
            ub = sarray > bottom
            dt = sarray < top
            ia = in_array[ub * dt]
            if np.any(ia):
                binslist.append(ia)

        return binslist


def find_variable_genes(M, *, bins=20, cutoff=1.7):
    """
    M: genes x cells, numpy ndarray;
    return: highly variable genes index in M
    method mod from Macosko et al
    """
    mm = M.mean(axis=1)
    mv = M.var(axis=1)
    mdi = mv/mm

    fvg_ind = []
    bins_cut_list = bins_cut(mm, bins)
    for i in bins_cut_list:
        zbinpick = stats.zscore(mdi[i])
        if np.any(zbinpick > cutoff):
            m = np.squeeze(np.copy(zbinpick > cutoff))
            fvg_ind.append(i[m])

    return np.concatenate(fvg_ind)
