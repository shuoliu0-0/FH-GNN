import os
import torch
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
from typing import Union
from itertools import repeat
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from featurization import atom_features, bond_features, get_fp_feature, motif_decomp


class HiMolGraph:
    """
    A HiMolGraph represents the Hierarchical Molecular graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, mol: Union[str, Chem.Mol]):
        """
        Computes the graph structure and featurization of a molecule.

        :param mol: A SMILES string or an RDKit molecule.
        """
        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            mol = Chem.MolFromSmiles(mol)

        # atoms
        atom_features_list = [atom_features(atom) for atom in mol.GetAtoms()]
        num_atoms = len(atom_features_list)
        # Initialize atom to bond mapping for each atom
        a2b = []
        for _ in range(num_atoms):
            a2b.append([])
        x_nosuper = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        self.n_bonds = 0  # number of bonds
        b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = []  # mapping from bond index to the index of the reverse bond
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_features(bond)
                edges_list.append((i, j))
                edge_features_list.append(atom_features_list[i] + edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(atom_features_list[j] + edge_feature)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                a2b[j].append(b1)  # b1 = a1 --> a2
                b2a.append(i)
                a2b[i].append(b2)  # b2 = a2 --> a1
                b2a.append(j)
                b2revb.append(b2)
                b2revb.append(b1)
                self.n_bonds += 2

            edge_index_nosuper = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            edge_attr_nosuper = torch.tensor(np.array(edge_features_list),
                                             dtype=torch.long)
        else:
            print('bond<0')
            edge_index_nosuper = torch.empty((2, 0), dtype=torch.long)
            edge_attr_nosuper = torch.empty((0,), dtype=torch.long)

        # add super node
        super_x_list = list(np.sum(np.array(atom_features_list), axis=0))
        super_x = torch.tensor([super_x_list]).to(x_nosuper.device)

        fp_x = get_fp_feature(mol)
        fp_x = torch.tensor([fp_x]).to(x_nosuper.device)

        # add motif
        breaks_bonds, cliques = motif_decomp(mol)
        num_motif = len(cliques)
        if num_motif > 0:
            motif_x = []
            for k, motif in enumerate(cliques):
                a2b.append([])
                num_edge = 0
                for idx, (i, j) in enumerate(edges_list):
                    if i in motif and j in motif:
                        num_edge += 1
                m_x = list(np.sum(np.array(atom_features_list)[motif], axis=0))
                motif_x.append(m_x)

            motif_edge_attr = []
            for idx, (i, j) in enumerate(edges_list):
                if f'{i}_{j}' in list(breaks_bonds.keys()):
                    motif_edge_attr.append(motif_x[breaks_bonds[f'{i}_{j}'][0]] + edge_features_list[idx][89:])
                    motif_edge_attr.append(motif_x[breaks_bonds[f'{i}_{j}'][1]] + edge_features_list[idx][89:])

                    b1 = self.n_bonds
                    b2 = b1 + 1
                    a2b[num_atoms + breaks_bonds[f'{i}_{j}'][1]].append(b1)  # b1 = a1 --> a2
                    b2a.append(num_atoms + breaks_bonds[f'{i}_{j}'][0])
                    a2b[num_atoms + breaks_bonds[f'{i}_{j}'][0]].append(b2)  # b2 = a2 --> a1
                    b2a.append(num_atoms + breaks_bonds[f'{i}_{j}'][1])
                    b2revb.append(b2)
                    b2revb.append(b1)
                    self.n_bonds += 2
            for k, motif in enumerate(cliques):
                for i in motif:
                    motif_edge_attr.append(atom_features_list[i] + [0] * 9)
                    a2b[i].append(self.n_bonds)  # b1 = a1 --> a2
                    b2a.append(i)
                    b2revb.append(self.n_bonds)
                    self.n_bonds += 1
            a2b.append([])
            motif_x0 = torch.tensor(np.array(motif_x)).to(x_nosuper.device)
            x = torch.cat((x_nosuper, motif_x0, super_x), dim=0)

            super_edge_attr = []
            for i in range(num_motif):
                super_edge_attr.append(motif_x[i] + [0] * 9)
                a2b[num_atoms + i].append(self.n_bonds)  # b1 = a1 --> a2
                b2a.append(num_atoms + i)
                b2revb.append(self.n_bonds)
                self.n_bonds += 1

            # print(np.array(motif_edge_attr))
            motif_edge_attr = torch.tensor(np.array(motif_edge_attr))
            motif_edge_attr = motif_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)
            super_edge_attr = torch.tensor(np.array(super_edge_attr))
            super_edge_attr = super_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)
            edge_attr = torch.cat((edge_attr_nosuper, motif_edge_attr, super_edge_attr), dim=0)
            num_part = torch.tensor(np.array([[num_atoms, num_motif, 1]]))

        else:
            x = torch.cat((x_nosuper, super_x), dim=0)
            a2b.append([])
            super_edge_attr = []
            for i in range(num_atoms):
                super_edge_attr.append(atom_features_list[i] + [0] * 9)
                a2b[i].append(self.n_bonds)  # b1 = a1 --> a2
                b2a.append(i)
                b2revb.append(self.n_bonds)
                self.n_bonds += 1

            super_edge_attr = torch.tensor(np.array(super_edge_attr))
            super_edge_attr = super_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)
            edge_attr = torch.cat((edge_attr_nosuper, super_edge_attr), dim=0)
            num_part = torch.tensor(np.array([[num_atoms, 0, 1]]))

        self.n_atoms = len(x)  # number of atoms
        self.f_atoms = np.array(x).tolist()  # mapping from atom index to atom features
        self.f_bonds = np.array(edge_attr).tolist()  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = a2b  # mapping from atom index to incoming bond indices
        self.b2a = b2a  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = b2revb  # mapping from bond index to the index of the reverse bond

        self.num_part = num_part
        self.fp_x = fp_x


def _load_bace_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array
    containing indices for each of the 3 folds, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['mol']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['Class']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    folds = input_df['Model']
    folds = folds.replace('Train', 0)  # 0 -> train
    folds = folds.replace('Valid', 1)  # 1 -> valid
    folds = folds.replace('Test', 2)  # 2 -> test
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    assert len(smiles_list) == len(folds)

    return smiles_list, rdkit_mol_objs_list, folds.values, labels.values


def _load_bbbp_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    print('preprocessed_smiles_list:', len(preprocessed_smiles_list))
    labels = input_df['p_np']
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)

    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values


def _load_tox21_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
             'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # convert nan to 0
    labels = labels.fillna(0)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)

    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_sider_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    tasks = ['Hepatobiliary disorders',
             'Metabolism and nutrition disorders', 'Product issues', 'Eye disorders',
             'Investigations', 'Musculoskeletal and connective tissue disorders',
             'Gastrointestinal disorders', 'Social circumstances',
             'Immune system disorders', 'Reproductive system and breast disorders',
             'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
             'General disorders and administration site conditions',
             'Endocrine disorders', 'Surgical and medical procedures',
             'Vascular disorders', 'Blood and lymphatic system disorders',
             'Skin and subcutaneous tissue disorders',
             'Congenital, familial and genetic disorders',
             'Infections and infestations',
             'Respiratory, thoracic and mediastinal disorders',
             'Psychiatric disorders', 'Renal and urinary disorders',
             'Pregnancy, puerperium and perinatal conditions',
             'Ear and labyrinth disorders', 'Cardiac disorders',
             'Nervous system disorders',
             'Injury, poisoning and procedural complications']
    labels = input_df[tasks]
    labels = labels.replace(0, -1)
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)

    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_clintox_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    preprocessed_rdkit_mol_objs_list = [m if m != None else None for m in
                                        rdkit_mol_objs_list]
    preprocessed_smiles_list = [AllChem.MolToSmiles(m) if m != None else
                                None for m in preprocessed_rdkit_mol_objs_list]
    tasks = ['FDA_APPROVED', 'CT_TOX']
    labels = input_df[tasks]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    assert len(smiles_list) == len(preprocessed_rdkit_mol_objs_list)
    assert len(smiles_list) == len(preprocessed_smiles_list)
    assert len(smiles_list) == len(labels)
    return preprocessed_smiles_list, preprocessed_rdkit_mol_objs_list, \
           labels.values


def _load_esol_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    # NB: some examples have multiple species
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['measured log solubility in mols per litre']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)

    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_freesolv_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['expt']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)

    return smiles_list, rdkit_mol_objs_list, labels.values


def _load_lipophilicity_dataset(input_path):
    """

    :param input_path:
    :return: list of smiles, list of rdkit mol obj, np.array containing the
    labels (regression task)
    """
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    labels = input_df['exp']
    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)

    return smiles_list, rdkit_mol_objs_list, labels.values


def mol_to_graph_data_obj_simple(mol, smi=None):
    data = Data(smi=smi)
    return data


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='bace',
                 empty=False):

        self.dataset = dataset
        self.root = root
        super(MoleculeDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if key == 'smi':
                data[key] = item[slices[idx]:slices[idx + 1]]
                continue
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []

        if self.dataset == 'bace':
            smiles_list, rdkit_mol_objs, folds, labels = \
                _load_bace_dataset(self.raw_paths[0])
            himol_graph = {}
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                himol_graph[smiles_list[i]] = HiMolGraph(rdkit_mol)
                data = mol_to_graph_data_obj_simple(rdkit_mol, smiles_list[i])
                # manually add mol id
                data.id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data.fold = torch.tensor([folds[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
            out_path = os.path.dirname(self.raw_paths[0])
            save_path = os.path.join(out_path, f'process_all.pkl')
            pickle.dump(himol_graph, open(save_path, 'wb'))

        elif self.dataset == 'bbbp':
            smiles_list, rdkit_mol_objs, labels = \
                _load_bbbp_dataset(self.raw_paths[0])
            himol_graph = {}
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    himol_graph[smiles_list[i]] = HiMolGraph(rdkit_mol)
                    data = mol_to_graph_data_obj_simple(rdkit_mol, smiles_list[i])
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor([labels[i]])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])
            out_path = os.path.dirname(self.raw_paths[0])
            save_path = os.path.join(out_path, f'process_all.pkl')
            pickle.dump(himol_graph, open(save_path, 'wb'))

        elif self.dataset == 'tox21':
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(self.raw_paths[0])
            himol_graph = {}
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                n_atoms = rdkit_mol.GetNumAtoms()
                if n_atoms == 1:
                    continue
                himol_graph[smiles_list[i]] = HiMolGraph(rdkit_mol)
                data = mol_to_graph_data_obj_simple(rdkit_mol, smiles_list[i])
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
            out_path = os.path.dirname(self.raw_paths[0])
            save_path = os.path.join(out_path, f'process_all.pkl')
            pickle.dump(himol_graph, open(save_path, 'wb'))

        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = \
                _load_sider_dataset(self.raw_paths[0])
            himol_graph = {}
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                himol_graph[smiles_list[i]] = HiMolGraph(rdkit_mol)
                data = mol_to_graph_data_obj_simple(rdkit_mol, smiles_list[i])
                # manually add mol id
                data.id = torch.tensor(
                    [i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
            out_path = os.path.dirname(self.raw_paths[0])
            save_path = os.path.join(out_path, f'process_all.pkl')
            pickle.dump(himol_graph, open(save_path, 'wb'))

        elif self.dataset == 'clintox':
            smiles_list, rdkit_mol_objs, labels = \
                _load_clintox_dataset(self.raw_paths[0])
            himol_graph = {}
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol != None:
                    himol_graph[smiles_list[i]] = HiMolGraph(rdkit_mol)
                    data = mol_to_graph_data_obj_simple(rdkit_mol, smiles_list[i])
                    # manually add mol id
                    data.id = torch.tensor(
                        [i])  # id here is the index of the mol in
                    # the dataset
                    data.y = torch.tensor(labels[i, :])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])
            out_path = os.path.dirname(self.raw_paths[0])
            save_path = os.path.join(out_path, f'process_all.pkl')
            pickle.dump(himol_graph, open(save_path, 'wb'))

        elif self.dataset == 'esol':
            smiles_list, rdkit_mol_objs, labels = \
                _load_esol_dataset(self.raw_paths[0])
            himol_graph = {}
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                himol_graph[smiles_list[i]] = HiMolGraph(rdkit_mol)
                data = mol_to_graph_data_obj_simple(rdkit_mol, smiles_list[i])
                # manually add mol id
                data.id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
            out_path = os.path.dirname(self.raw_paths[0])
            save_path = os.path.join(out_path, f'process_all.pkl')
            pickle.dump(himol_graph, open(save_path, 'wb'))

        elif self.dataset == 'freesolv':
            smiles_list, rdkit_mol_objs, labels = \
                _load_freesolv_dataset(self.raw_paths[0])
            himol_graph = {}
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                himol_graph[smiles_list[i]] = HiMolGraph(rdkit_mol)
                data = mol_to_graph_data_obj_simple(rdkit_mol, smiles_list[i])
                # manually add mol id
                data.id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
            out_path = os.path.dirname(self.raw_paths[0])
            save_path = os.path.join(out_path, f'process_all.pkl')
            pickle.dump(himol_graph, open(save_path, 'wb'))

        elif self.dataset == 'lipophilicity':
            smiles_list, rdkit_mol_objs, labels = \
                _load_lipophilicity_dataset(self.raw_paths[0])
            himol_graph = {}
            for i in range(len(smiles_list)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                himol_graph[smiles_list[i]] = HiMolGraph(rdkit_mol)
                data = mol_to_graph_data_obj_simple(rdkit_mol, smiles_list[i])
                # manually add mol id
                data.id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor([labels[i]])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
            out_path = os.path.dirname(self.raw_paths[0])
            save_path = os.path.join(out_path, f'process_all.pkl')
            pickle.dump(himol_graph, open(save_path, 'wb'))

        else:
            raise ValueError('Invalid dataset name')

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

