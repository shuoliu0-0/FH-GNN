import os
import pickle
from typing import List, Tuple, Union
import torch
from rdkit import Chem
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from rdkit.Chem import MACCSkeys
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import rdMolDescriptors

fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

ATOM_FDIM = 89
BOND_FDIM = 9
FEATURES = {
    'atomic_num': [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26,
                   27, 28, 29, 30, 31, 32, 33, 34, 35, 38, 39, 40, 42, 43, 46, 47, 48, 49, 50, 51, 53,
                   56, 57, 60, 62, 63, 64, 66, 70, 78, 79, 80, 81, 82, 83, 88, 98],
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-1, -2, 1, 2, 3, 0],
    'chiral_tag': [Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                   Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                   Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.UNSPECIFIED,
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2],
    'stereo': [Chem.rdchem.BondStereo.STEREONONE,
               Chem.rdchem.BondStereo.STEREOZ,
               Chem.rdchem.BondStereo.STEREOE], }


def get_atom_fdim() -> int:
    """Gets the dimensionality of atom features."""
    return ATOM_FDIM


def get_bond_fdim(atom_messages: bool = False) -> int:
    """
    Gets the dimensionality of bond features.

    :param atom_messages whether atom messages are being used. If atom messages, only contains bond features.
    Otherwise contains both atom and bond features.
    :return: The dimensionality of bond features.
    """
    return BOND_FDIM + (not atom_messages) * get_atom_fdim()


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * len(choices)
    index = choices.index(value)
    encoding[index] = 1

    return encoding


def bond_features(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [0] * BOND_FDIM
    else:
        bt = bond.GetBondType()
        fbond = [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
        fbond += onek_encoding_unk(bond.GetStereo(), FEATURES['stereo'])
    return fbond


def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum(), FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0]
    if functional_groups is not None:
        features += functional_groups
    return features


def pharm_feats(mol, factory=factory):
    types = [i.split('.')[1] for i in factory.GetFeatureDefs().keys()]
    feats = [i.GetType() for i in factory.GetFeaturesForMol(mol)]
    result = [0] * len(types)
    for i in range(len(types)):
        if types[i] in list(set(feats)):
            result[i] = 1
    return result


### form fixed representations (TODO: save all fixed reps in advance?)
MORGAN_RADIUS = 2
MORGAN_NUM_BITS = 2048


def get_fp_feature(mol):
    try:
        fp_atomPairs = list(rdMolDescriptors.GetHashedAtomPairFingerprint(mol, nBits=2048, use2D=True))
        fp_maccs = list(MACCSkeys.GenMACCSKeys(mol))
        fp_morganBits = list(GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_NUM_BITS))
        fp_morganCounts = list(AllChem.GetHashedMorganFingerprint(mol, radius=MORGAN_RADIUS, nBits=MORGAN_NUM_BITS))
        fp_pharm = pharm_feats(mol)
    except Exception:
        fp_atomPairs = [0 for i in range(2048)]
        fp_maccs = [0 for i in range(167)]
        fp_morganBits = [0 for i in range(2048)]
        fp_morganCounts = [0 for i in range(2048)]
        fp_pharm = [0 for i in range(27)]
    return fp_atomPairs + fp_maccs + fp_morganBits + fp_morganCounts + fp_pharm


def get_cliques_link(breaks, cliques):
    clique_id_of_node = {}
    for idx, clique in enumerate(cliques):
        for c in clique:
            clique_id_of_node[c] = idx
    breaks_bond = {}
    for bond in breaks:
        breaks_bond[f'{bond[0]}_{bond[1]}'] = [clique_id_of_node[bond[0]], clique_id_of_node[bond[1]]]
    return breaks_bond


def motif_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [], []

    cliques = []
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])

    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])
            breaks.append([bond[0][0], bond[0][1]])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if n_atoms > len(c) > 0]

    breaks = get_cliques_link(breaks, cliques)
    return breaks, cliques


class BatchMolGraph:
    def __init__(self, smiles, atom_fdim, bond_fdim, fp_fdim, data_name):
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.fp_fdim = fp_fdim
        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond

        # fp
        fp_x_out = torch.empty((0, self.fp_fdim))

        data_file = open(f'/home/lius/himol-mol/dataset/{data_name}/raw/process_all.pkl', 'rb')
        mol_graphs = pickle.load(data_file)
        data_file.close()

        mol_atom_num = []
        for smi in smiles:
            mol_graph = mol_graphs[smi[0]]
            mol_atom_num.append(int(mol_graph.num_part[0][0]))
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

            # fp_graph
            fp_x_out = torch.cat((fp_x_out, mol_graph.fp_x))

        self.max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages
        self.smiles = smiles
        self.mol_atom_num = mol_atom_num

        self.fp_x = fp_x_out

    def get_components(self, atom_messages: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                                                   torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                                                   List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :param atom_messages: Whether to use atom messages instead of bond messages. This changes the bond features
        to contain only bond features rather than a concatenation of atom and bond features.
        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        if atom_messages:
            f_bonds = self.f_bonds[:, :get_bond_fdim(atom_messages=atom_messages)]
        else:
            f_bonds = self.f_bonds

        return self.f_atoms, f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a

