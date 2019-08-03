# SingleAnalyst

## Introduction
SingleAnalyst is an integrated platform for single-cell RNA-seq data analysis,
focusing on the cell type assignment problem in single cell RNA-seq analysis.

SingleAnalyst implemented various quality control, normalization and feature selection methods
for data preprocessing, and featured a k-nearest neighbors based cell type annotation and assignment methods 

 
## Requirement
* python3 >= 3.6
* linux / WSL

## install
1. install some dependency by conda (pip did not work properly for those package)
    ```sh
    conda install numpy bitarray
    conda install faiss-cpu -c pytorch
    ```
2. install package
    ```sh
    pip install .
    ```

## Usage
### Data preprocessing
#### Read data

Read data, and create a singleCellData object.
```python
from SingleAnalyst.basic import indexedList, infoTable, singleCellData
gene_info = indexedList(gene_list)
cell_info = infoTable(
    ['cell_list', 'cell_type'],
    [cell_list, cell_type_list])
ex_m = mmread(os.path.join(path, 'expr_m.mtx')).todense()
dataset = singleCellData(ex_m, gene_info, cell_info)
```

Or, read from saved data
```python
import SingleAnalyst
datapath = 'output/xin/'
data_set = SingleAnalyst.dataIO.read_data_mj(datapath)
```
#### quality control
Filter out low quality data
```python
f1 = scr.filter.minGeneCellfilter()
f2 = scr.filter.minCellGenefilter()

dataset = dataset.apply_proc(f1)
dataset = dataset.apply_proc(f2)
```

#### normalization
Data normalization
```
norm = scr.normalization.logNormlization()
dataset.apply_proc(norm)
```

#### feature selection
Select informative feature
```python
s1 = scr.selection.dropOutSelecter(num_features=500)
s2 = scr.selection.highlyVarSelecter(num_features=500)
s3 = scr.selection.randomSelecter(num_features=500)

dataset.apply_proc(s1)
```

### index build and similar search
Split data for test
```python
train_d, test_d = scr.process.tt_split(dataset)
refdata = scr.RefData.queryData(train_d)
q_xdata = scr.RefData.queryData(test_d)
```

#### build index for reference data
```python
nn_indexer = scr.index.faiss_baseline_nn()

index = scr.index.indexRef(refdata, nn=nn_indexer)
```

#### knn search and celltype annotation
```python
qxm = q_xdata.get_qxm(gene_list=index.gene_ref.get_list())

res = index.get_predict(qxm=qxm)

# visually insapect knn result  
i_qx = qxm[19,:]
nnf = index.get_knn_vis(i_qx)
```


## Contacts
yuyifei@tongji.edu.cn or qiliu@tongji.edu.cn
