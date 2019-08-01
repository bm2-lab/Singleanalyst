import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import variation
from scipy.special import logsumexp
from sklearn.decomposition import PCA

from .basic import scRefData

sns.set()
sns.set_style("whitegrid")

PLT_SIZE = (8, 8)
DOT_SIZE = 10


# plot for data qc
def plot_g_e(sco, log=False):
    """
    plot log expression sum and detected gene counts
    for cell inspect
        should not be applyed to normalized expression
    log:
        if input data already log transformed
    """
    expression = sco.expression_matrix
    bm = expression > 0
    if log:
        x = logsumexp(expression, axis=0)
    else:
        x = np.sum(expression, axis=0)
        x = np.log1p(x)
    y = np.sum(bm, axis=0)
    x = np.copy(x)
    y = np.copy(y)

    fig, ax = plt.subplots(figsize=PLT_SIZE)
    ax.scatter(np.squeeze(x), np.squeeze(y))

    ax.set_xlabel('log sum expression')
    ax.set_ylabel('detected genes')

    return fig

def plot_PCA(sco):
    # expression matrix n x m, n:gene, m:cell
    x = sco.expression_matrix.T
    pca = PCA(n_components=2)
    x_new  = pca.fit_transform(x)

    fig, ax = plt.subplots(figsize=PLT_SIZE)
    ax.scatter(x_new[:,0], x_new[:,1])

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA')

    return fig


# plot for cell expression inspection
#   distribution plot
def dist_plot(sco):
    """
    gene number distribution & expression distribution
    """
    expression = sco.expression_matrix
    bm = expression > 0
    genes_counts = np.sum(bm, axis=0)
    expression_sum = np.sum(expression, axis=0)
    fig, ax_l = plt.subplots(1, 2, figsize=PLT_SIZE)
    sns.distplot(genes_counts, ax=ax_l[0])
    sns.distplot(expression_sum, ax=ax_l[1])
    ax_l[1].yaxis.set_ticks_position('right')
    ax_l[0].set_title("Gene Number Distribution")
    ax_l[1].set_title("Expression Distribution")
    return fig

# 评估ann效果
# IoU for ann
def plot_IoU(ioulist, mlist):
    """
    box plot for IoU of different ann
    """
    ioum = np.stack(ioulist, axis=1)
    import pandas as pd
    p = pd.DataFrame(ioum, columns=mlist)
    fig, ax = plt.subplots(figsize=PLT_SIZE)
    sns.boxplot(data=p, palette="PRGn", ax=ax)
    return fig


# nn in ann plot for ann q
def plot_tniann(rangek, ninlist, mlist):
    fig, ax = plt.subplots(figsize=PLT_SIZE)
    for i in ninlist:
        ax.plot(np.arange(rangek), i)
    ax.legend(mlist)
    return fig


# ref-cluster
# heatmap for cluster-ref
def index_heatmap(index_expression, cell_list):
    # index_expression.shape h, w
    # h: gene_e in cells, w, celltype
    import pandas as pd
    p = pd.DataFrame(index_expression, columns=cell_list)
    fig = sns.clustermap(
        p, cmap="bwr", linewidths=0, figsize=PLT_SIZE)
    # sns return?
    return fig


# violin plot for select feature
def gene_violinplot(sco, gene):
    gi = sco.gene_to_index([gene])
    ge = sco.expression_matrix[gi, :]
    ge = np.squeeze(np.copy(ge))
    # ge = ge.reshape(1, -1)
    cell_type_array = sco.cell_info['cell_type']
    cell_type_array = np.squeeze(np.copy(cell_type_array))
    # cell_type_array = cell_type_array.reshape(1, -1)
    data = {
        'cell_type': cell_type_array,
        'expression': ge}
    import pandas as pd
    p = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=PLT_SIZE)
    sns.violinplot(
        x='expression',
        y='cell_type',
        data=p, ax=ax)
    ax.set_title(gene)
    return fig


# 2D plot for cluster inspection
def plot_2demb(X):
    fig, ax = plt.subplots(figsize=PLT_SIZE)
    ax.scatter(X[:, 0], X[:, 1], s=DOT_SIZE)
    ax.set_title("t-sne embedding")
    return fig


def plot_2demb_labels(X, y):
    """
    inspect cluster labels
    """
    # all_l = np.unique(y)
    fig, ax = plt.subplots(figsize=PLT_SIZE)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=DOT_SIZE)
    ax.set_title("cluster")
    return fig


def plot_2demb_nn(X, nni, y, labels):
    """
    nn in original 2d embedding
    """
    y = np.array(y)
    cmap = plt.cm.cool
    norm = plt.Normalize(vmin=min(y), vmax=max(y)+1)
    fig, ax = plt.subplots(figsize=PLT_SIZE)
    for i, l in enumerate(labels):
        i_li = y == i
        ax.scatter(X[i_li, 0], X[i_li, 1], c=cmap(norm(y[i_li])), label=l, s=DOT_SIZE)
    ax.scatter(X[nni, 0], X[nni, 1], c='red', marker='^', label="finded nn", s=DOT_SIZE*1.5)
    ax.legend()
    ax.set_title("Finded NN")
    return fig

