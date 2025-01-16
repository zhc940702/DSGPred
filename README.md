# Requirement

* python == 3.8.0
* pytorch == 2.2.1+cu118
* torch-geometric == 2.6.1
* Numpy == 1.24.3
* scikit-learn == 1.3.2
* scipy == 1.10.1
* pandas == 2.0.3
* fitlog == 0.9.15

If you use anaconda or miniconda, you can install them using conda

But before install torch-geometric, you need to first download and install in order the corresponding versions of torch_scatter, torch_sparse, torch_cluster and torch_spline_conv from https://data.pyg.org/whl/torch-2.2.1%2Bcu118.html, and then run 

```
conda install torch_geometric
```

ATTENTION：Because there may be version incompatibility issues between fitlog and numpy packages, the following error occurs: 

```
AttributeError: module 'numpy' has no attribute 'str'.
```

if the error occurs, please modify line 775 in \site-packages\fitlog\fastlog\logger.py 

```
if isinstance(value, (np.str, str)) or value is None:
```

to

```
if isinstance(value, (np.str_, str)) or value is None:
```



# Files

## 1. data

#### 	1. cell

​		cell_index.pkl: the index of cell lines 

​		cn_580cell_706gene.pkl, exp_580cell_706gene.pkl, mu_580cell_706gene.pkl: three types of cell line feature data

#### 	2. drug_fp

​		It stores six types of drug fingerprints collected, including Extended-Connectivity FingerPrints (**ECFP**), PubChem Substructure FingerPrints (**PSFP**) 		, Daylight FingerPrints (**DFP**) , RDKit 2D normalized FingerPrints (**RDKFP**), Explainable Substructure Partition FingerPrints (**ESPFP**), and Extended 		Reduced Graph FingerPrints (**ERGFP**).

#### 	3. drug_graph

​	zxy_drug_ADR.pkl,zxy_drug_disease.pkl,zxy_drug_gene_all.pkl,zxy_drug_miRNA.pklzxy_stitch_combined_score.pkl: Five biological entity data of drugs 

​	drug_ADR_dict.npy,drug_dis_dict.npy,drug_miRNA_dict.npy,drug_drug_dict.npy,drug_target_dict.npy: Five drug features calculated from entity data

#### 	4. IC50

​		ic_170drug_580cell.pkl:IC50 data for 170 drugs and 580 cell lines, as a feature of drugs and cell lines

​		samples_82833.pkl:benchmark dataset, association data between drugs and cell lines.

#### 	5. drug_graph.npy

​		Instance-specific graphs designed for drugs

## 2. EarlyStopping.py

The training stop function stops when the model is trained 10 times in a row and the results no longer improve.

## 3. en_decoder.py

Implementation of the feedforward neural network layer

## 4. module.py

Specific designs of some modules, such as drug and cell line feature fusion modules, interaction modules, etc.


## 5. network.py

 This function contains the network framework of our entire model

## 6. utils.py

This function contains the necessary processing subroutines, such as model training function, model validation function, data processing function, etc.

## 7. main.py

The main function of the model, which contains some hyperparameter settings. Running this function can train and test the model



# Train and test 

mode: Set the mode to train or test, then you can train the model or test the model

epochs: Define the maximum number of epochs.

batch_size: Define the number of batch size for training.

data_bath: All input data should be placed in the folder of this path. (The data folder we uploaded contains all the required data.)

weight_path: Define the path to save the model.



All files of Data and Code should be stored in the same folder to run the model.

run:

```
main.py
```

or  type in the command line:

```
python main.py --mode train --epochs 200  --batch_size 128  --rawpath data/ --weight_path best
```



