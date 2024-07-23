# FH-GNN

## Introduction
* Source code for the paper "Fingerprint-enhanced hierarchical molecular graph neural networks for property prediction".

* We propose a novel Fingerprint-enhanced Hierarchical Graph Neural Network (FH-GNN) for molecular property prediction, which simultaneously learned information from both hierarchical molecular graphs and molecular fingerprints.

![Fingerprint-enhanced Hierarchical Graph Neural Network](images/fig1.tif)


## Dataset
All data used in this paper are publicly available on [Molecule-Net](https://github.com/deepchem/deepchem/tree/master/deepchem/molnet/load_function).

## Environment
* base dependencies:
```
  - numpy == 1.21.5
  - rdkit == 2018.03.4
  - pandas == 1.3.5
  - python == 3.7.16
  - pytorch == 1.12.1
  - scikit-learn == 1.0.2
```

## Usage

#### Args:
- --dataset : The name of input dataset.
- --data_dir : The path of input CSV file.
- --save_dir : The path to save output model.
- --batch_size : The input batch size for training.
- --epochs : The number of epochs to train.
- --lr : The learning rate for the prediction layer.
- --depth : The depth of molecule encoder.

#### Quick Run
```bash
python train.py
```
