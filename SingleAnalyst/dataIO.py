import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csc_matrix, csr_matrix
import json
import os
import feather
import h5py

from .basic import singleCellData, indexedList, infoTable

"""
read expressin in .mtx
read info in json
"""

def read_data_mj(path):
    """
    m, .mtx matrix
    j, .json info
    """
    jsonf = open(os.path.join(path,'info.json'), 'r')
    jd = json.load(jsonf)
    
    print(jd[1])

    meta_info = dict(zip(jd[1][0], jd[1][1]))
    gene_list = np.array(jd[2])
    cell_list = np.array(jd[3])
    cell_type_list = np.array(jd[4])

    gene_ref = indexedList(gene_list)
    cell_info = infoTable(
        ['cell_list', 'cell_type'],
        [cell_list, cell_type_list])
    
    if 'processed' in meta_info:
        proc = meta_info['processed']
        proc.append('read data')
    else:
        proc = None

    ex_m = mmread(os.path.join(path, 'expr_m.mtx')).todense()
    return singleCellData(ex_m, gene_ref, cell_info, meta_info=meta_info, proc=proc)


def read_data_npz(path):
    """
    npz, npz data
    """
    data = np.load(path,allow_pickle=True)
    # data1.files
    # ['meta_info', cells', 'genes', 'counts', 'labels']
    gene_list = data['genes']
    cell_list = data['cells']
    cell_type_list = data['labels']
    ex_m = data['counts']
    meta_info = data['meta_info']
    meta_info = dict(zip(meta_info[0], meta_info[1]))

    gene_ref = indexedList(gene_list)
    cell_info = infoTable(
        ['cell_list', 'cell_type'],
        [cell_list, cell_type_list])

    if 'processed' in meta_info:
        proc = meta_info['processed']
        proc.append('read data')
    else:
        proc = None

    return singleCellData(ex_m, gene_ref, cell_info, meta_info=meta_info, proc=proc)

def read_data_h5(path):
    """
    npz, npz data
    """
    jsonf = open(os.path.join(path,'info_hdf5.json'), 'r')
    jd = json.load(jsonf)
    
    print(jd[1])

    meta_info = dict(zip(jd[1][0], jd[1][1]))
    gene_list = np.array(jd[2])
    cell_list = np.array(jd[3])
    cell_type_list = np.array(jd[4])

    gene_ref = indexedList(gene_list)
    cell_info = infoTable(
        ['cell_list', 'cell_type'],
        [cell_list, cell_type_list])
    
    if 'processed' in meta_info:
        proc = meta_info['processed']
        proc.append('read data')
    else:
        proc = None

    f = h5py.File(os.path.join(path, 'expr_m.hdf5'), "r")
    ex_m = np.array(f['counts'])

    return singleCellData(ex_m, gene_ref, cell_info, meta_info=meta_info, proc=proc)

def read_data_feather(path):
    """
    feather, feather data for em
    feather info
    """
    jsonf = open(os.path.join(path,'info_feather.json'), 'r')
    jd = json.load(jsonf)
    
    print(jd[1])

    meta_info = dict(zip(jd[1][0], jd[1][1]))
    gene_list = np.array(jd[2])
    cell_list = np.array(jd[3])
    cell_type_list = np.array(jd[4])

    gene_ref = indexedList(gene_list)
    cell_info = infoTable(
        ['cell_list', 'cell_type'],
        [cell_list, cell_type_list])
    
    if 'processed' in meta_info:
        proc = meta_info['processed']
        proc.append('read data')
    else:
        proc = None

    dm = feather.read_dataframe(os.path.join(path,'expr_m.feather'))
    ex_m = np.array(dm)

    return singleCellData(ex_m, gene_ref, cell_info, meta_info=meta_info, proc=proc)

def save_data(sco, outputpath):
    mmf = os.path.join(outputpath, 'expr_m.mtx')
    # with open(outputpath + 'expr_m.mtx', 'w') as f:
    #     mmwrite(f, sco.expression_matrix)
    mmwrite(mmf, csc_matrix(sco.expression_matrix))

    sc_info = []
    sc_info.append(['meta_info'])
    meta_info = sco.meta_info
    p = sco.processed
    p.append('save data')
    meta_info['processed'] = p
    meta_info = list([list(meta_info.keys()), list(meta_info.values())])
    sc_info.append(meta_info)

    sc_info[0].append("genes_list")
    sc_info.append(list(sco.gene_ref.get_list()))
    for i in sco.cell_info.data_names:
        sc_info[0].append(i)
        sc_info.append(list(sco.cell_info[i]))  

    with open(os.path.join(outputpath, 'info.json'), 'w') as f:
        json.dump(sc_info, f)

def save_data_npz(sco, outputpath, sparse=True):
    if sparse == True:
        m = csc_matrix(sco.expression_matrix)
    else:
        m = sco.expression_matrix
    meta_info = sco.meta_info
    p = sco.processed
    p.append('save data')
    meta_info['processed'] = p
    meta_info = list([list(meta_info.keys()), list(meta_info.values())])

    genes = list(sco.gene_ref.get_list())
    cells = list(sco.cell_info['cell_list'])
    labels = list(sco.cell_info['cell_type'])

    np.savez_compressed(outputpath, meta_info=meta_info, cells=cells, genes=genes, counts=m, labels=labels)

def save_data_h5(sco, outputpath, sparse=True):
    if sparse == True:
        m = csc_matrix(sco.expression_matrix)
    else:
        m = sco.expression_matrix

    f = h5py.File(os.path.join(outputpath, 'expr_m.hdf5'), "w")
    f.create_dataset('counts', data=m)

    sc_info = []
    sc_info.append(['meta_info'])
    meta_info = sco.meta_info
    p = sco.processed
    p.append('save data')
    meta_info['processed'] = p
    meta_info = list([list(meta_info.keys()), list(meta_info.values())])
    sc_info.append(meta_info)

    sc_info[0].append("genes_list")
    sc_info.append(list(sco.gene_ref.get_list()))
    for i in sco.cell_info.data_names:
        sc_info[0].append(i)
        sc_info.append(list(sco.cell_info[i]))  

    with open(os.path.join(outputpath, 'info_hdf5.json'), 'w') as f:
        json.dump(sc_info, f)

    # (outputpath, meta_info=meta_info, cells=cells, genes=genes, counts=m, labels=labels)


def save_data_feather(sco, outputpath):
    m = sco.expression_matrix

    import pandas as pd
    feather.write_dataframe(pd.DataFrame(m), os.path.join(outputpath,'expr_m.feather'))

    sc_info = []
    sc_info.append(['meta_info'])
    meta_info = sco.meta_info
    p = sco.processed
    p.append('save data')
    meta_info['processed'] = p
    meta_info = list([list(meta_info.keys()), list(meta_info.values())])
    sc_info.append(meta_info)

    sc_info[0].append("genes_list")
    sc_info.append(list(sco.gene_ref.get_list()))
    for i in sco.cell_info.data_names:
        sc_info[0].append(i)
        sc_info.append(list(sco.cell_info[i]))  

    with open(os.path.join(outputpath, 'info_feather.json'), 'w') as f:
        json.dump(sc_info, f)