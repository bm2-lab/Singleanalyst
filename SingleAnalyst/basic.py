import numpy as np
from sklearn import preprocessing


class indexedList(object):
    """
    provide value to index 
    """
    def __init__(self, values_list):
        _index = np.arange(len(values_list))
        self.index_values = dict(zip(_index, values_list))
        self.value_index = dict(zip(values_list, _index))
        self.len = len(values_list)

    def __len__(self):
        return self.len

    def value_to_index(self, v_list):
        try:
            ind_l = [self.value_index[i] for i in v_list]
        except KeyError:
            raise Exception("target value not in list")
        return np.array(ind_l)

    def index_to_value(self, i_list):
        try:
            ind_v = [self.index_values[i] for i in i_list]
        except KeyError:
            raise Exception("index out of value list's range")
        return np.array(ind_v)

    def get_filtered(self, findex):
        # make it compliant with bool mask
        index = np.arange(self.len)
        index = index[findex]
        _t = self.index_to_value(index)
        return indexedList(_t)

    def get_list(self):
        index = np.arange(self.len)
        return self.index_to_value(index)

class infoTable(object):
    def __init__(self, dataNames, dataSets):
        """
        collection of named list
        """
        assert len(dataNames) == len(dataSets), "Data not Match"
        self.data_num = len(dataNames)
        self.data_names = dataNames
        self._dnd = dict(zip(dataNames, np.arange(self.data_num)))
        self.data_sets = []
        self.data_length = len(dataSets[0])
        for i in dataSets:
            assert self.data_length == len(i), "Data not Match"
            self.data_sets.append(i)
        self.index = np.arange(self.data_length)

    def __len__(self):
        return self.data_length

    def add_data(self, name, data):
        assert self.data_length == len(data), "Data not Match"
        assert name not in self._dnd, "Data Already Exists"
        self.data_names.append(name)
        self.data_sets.append(data)
        self._dnd[name] = self.data_names
        self.data_num += 1

    def get_data_byname(self, name):
        assert name in self._dnd, "Data Not Exists"
        return (self.data_sets[self._dnd[name]])

    def __getitem__(self, key):
        return self.get_data_byname(key)

    def get_filtered(self, findex):
        index_filtered = self.index[findex]
        _ds = []
        for i in self.data_sets:
            _ds.append([i[j] for j in index_filtered])
        return infoTable(self.data_names, _ds)


class scBase(object):
    def __init__(self, gene_ref):
        # gene_ref: indexedList of genes for analysis
        assert isinstance(gene_ref, indexedList), "Arguments Error"
        self.gene_ref = gene_ref

    def gene_to_index(self, g_list):
        return self.gene_ref.value_to_index(g_list)

    def index_to_gene(self, i_list):
        return self.gene_ref.index_to_value(i_list)


class singleCellData(scBase):
    """
    Data object
    composed with expresison_matrix, gene_ref,
    and  cell_info
    """
    def __init__(self, expression_matrix, gene_ref, cell_info, meta_info, proc=None):
        """
        warper of singleCell data
        """
        self.expression_matrix = expression_matrix

        # assert isinstance(gene_ref, indexedList),\
        #     "Arguments Error: gene_ref"
        self.gene_ref = gene_ref

        # assert isinstance(cell_info, infoTable),\
        #     "Arguments Error: cell_info"
        self.cell_info = cell_info

        self.gene_num, self.cell_num = expression_matrix.shape

        if (len(cell_info) != self.cell_num):
            print(len(cell_info))
            print(self.cell_num)
            raise Exception("Cells Number Not Match")
        if (len(gene_ref) != self.gene_num):
            print(len(gene_ref))
            print(self.gene_num)
            raise Exception("Genes Number Not Match")

        if proc is None:
            self.processed = ['read data']
        else:
            self.processed = proc
        self.proc_func = []
        self.meta_info = meta_info

        self.meta_info["processed"] = proc

    def apply_proc(self, proc_func):
        """
        save process infomation
        """
        try:
            proc_func(self)
        except Exception as E:
            print("process fail")
            raise E
        self.processed.append(proc_func.process)
        self.proc_func.append(proc_func)
        self.gene_num, self.cell_num = self.expression_matrix.shape
        return self


class scRefData(singleCellData):
    """
    init labels for data with “cell_type” or other kind of annotation*
    *planed, not implemented yet
    """
    def __init__(self, sco):
        import copy
        expression_matrix = copy.deepcopy(sco.expression_matrix)
        gene_ref = copy.deepcopy(sco.gene_ref)
        cell_info = copy.deepcopy(sco.cell_info)
        meta_info = copy.deepcopy(sco.meta_info)
        proc = copy.deepcopy(sco.processed)
        super(scRefData, self).__init__(expression_matrix, gene_ref, cell_info, meta_info, proc)
        # celltype labels for analysis,biology irrelevant encode
        if hasattr(sco, 'le'):
            self.le = sco.le
        else:
            self.le = preprocessing.LabelEncoder()
            self.le.fit(self.cell_info['cell_type'])
        self.labels = self.le.transform(self.cell_info['cell_type'])


class baseTool(object):
    def __init__(self):
        self.process = None
        raise NotImplementedError

    def __call__(self, sco):
        """
        take singleCellData as input
        return a processed singleCellData
        """
        raise NotImplementedError


class cellDataFilter(baseTool):
    def __init__(self):
        self.cell_filter = None
        self.process = None
        raise NotImplementedError

    def _proc_o(self, sco):
        sco.expression_matrix = sco.expression_matrix[:, self.cell_filter]
        sco.cell_info = sco.cell_info.get_filtered(self.cell_filter)
        return sco


class geneDataFilter(baseTool):
    def __init__(self):
        self.gene_filter = None
        self.process = None
        raise NotImplementedError

    def _proc_o(self, sco):
        sco.expression_matrix = sco.expression_matrix[self.gene_filter, :]
        sco.gene_ref = sco.gene_ref.get_filtered(self.gene_filter)
        return sco


class dataNormalization(baseTool):
    """
    Normalization only change the value in expression_matrix
    """
    def __init__(self):
        self.process = None
        raise NotImplementedError

    def normalize(self, sco):
        raise NotImplementedError

    def _proc_o(self, sco):
        sco.expression_matrix = self.normalize(sco.expression_matrix)
        sco.normalized = True
        return sco

    def __call__(self, sco):
        return self._proc_o(sco)



class featureSelection(baseTool):
    """
    generally, it is just geneDataFilter
    """
    def __init__(self):
        self.selected_features = None
        self.process = None
        raise NotImplementedError

    def _proc_o(self, sco):
        sco.expression_matrix = sco.expression_matrix[self.selected_features, :]
        sco.gene_ref = sco.gene_ref.get_filtered(self.selected_features)
        return sco