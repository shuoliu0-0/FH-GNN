import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import init
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.models import MLP

from featurization import BatchMolGraph


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    return target


class HGNNEncoder(nn.Module):
    """A message passing neural network for encoding a molecule."""

    def __init__(self, atom_fdim, bond_fdim, hidden_size, depth, device):
        """Initializes the HGNNEncoder.

        :param args: Arguments.
        :param atom_fdim: Atom features dimension.
        :param bond_fdim: Bond features dimension.
        :param atom_messages: Whether to use atoms to pass messages instead of bonds.
        """
        super(HGNNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.depth = depth
        self.device = device
        self.bias = False
        self.attention = True

        # Dropout
        self.dropout_layer = nn.Dropout(p=0.1)

        # Activation
        self.act_func = nn.ReLU()

        # Cached zeros
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)

        # Input
        self.W_i = nn.Linear(self.bond_fdim, self.hidden_size, bias=self.bias)

        # Shared weight matrix across depths (default)
        self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.W_o = nn.Linear(self.atom_fdim + self.hidden_size, self.hidden_size)

        self.W_a = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
        self.W_b = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, mol_graph):
        """
        Encodes a batch of molecular graphs.

        :param mol_graph: A BatchMolGraph representing a batch of molecular graphs.
        :param features_batch: A list of ndarrays containing additional features.
        :return: A PyTorch tensor of shape (num_molecules, hidden_size) containing the encoding of each molecule.
        """

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope = mol_graph.get_components()
        f_atoms, f_bonds, a2b, b2a, b2revb = f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(
            self.device), b2a.to(self.device), b2revb.to(self.device)

        # Input
        inputs = self.W_i(f_bonds)  # num_bonds x hidden_size
        message = self.act_func(inputs)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
            a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
            rev_message = message[b2revb]  # num_bonds x hidden
            message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.act_func(inputs + message)  # num_bonds x hidden_size
            message = self.dropout_layer(message)  # num_bonds x hidden

        nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        # new_feature
        new_feature = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                new_feature.append(self.cached_zero_vector)
            else:
                cur_hiddens = a_input.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)

                mol_vec = mol_vec.sum(dim=0) / a_size
                new_feature.append(mol_vec)
        new_feature = torch.stack(new_feature, dim=0)  # (num_molecules, hidden_size)

        atom_hiddens = self.act_func(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)

                att_w = torch.matmul(self.W_a(cur_hiddens), cur_hiddens.t())
                att_w = F.softmax(att_w, dim=1)
                att_hiddens = torch.matmul(att_w, cur_hiddens)
                att_hiddens = self.act_func(self.W_b(att_hiddens))
                att_hiddens = self.dropout_layer(att_hiddens)
                mol_vec = (cur_hiddens + att_hiddens)

                mol_vec = mol_vec.sum(dim=0) / a_size
                mol_vecs.append(mol_vec)

        # out_data.close()
        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)

        return mol_vecs  # num_molecules x hidden


class WeightFusion(nn.Module):
    def __init__(self, feat_views, feat_dim, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(WeightFusion, self).__init__()
        self.feat_views = feat_views
        self.feat_dim = feat_dim
        self.weight = Parameter(torch.empty((1, 1, feat_views), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(int(feat_dim), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:

        return sum([inputs[i] * weight for i, weight in enumerate(self.weight[0][0])]) + self.bias


class FHGNN(nn.Module):
    def __init__(self,
                 data_name,
                 atom_fdim: int = None,
                 bond_fdim: int = None,
                 fp_fdim: int = None,
                 hidden_size=256,
                 depth=5,
                 device='cpu',
                 out_dim=2, ):
        super(FHGNN, self).__init__()
        self.data_name = data_name
        self.device = device
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.fp_fdim = fp_fdim
        self.encoder = HGNNEncoder(self.atom_fdim, self.bond_fdim, hidden_size, depth, device)
        self.mlp_fp = nn.Sequential(
            nn.Linear(fp_fdim, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, hidden_size)
        )
        self.feature_fusion = WeightFusion(2, hidden_size, device=device)
        self.mlp = MLP([hidden_size, hidden_size, out_dim], dropout=0.1)

    def forward(self, batch):
        mol_batch = BatchMolGraph(batch.smi, atom_fdim=self.atom_fdim, bond_fdim=self.bond_fdim,
                                  fp_fdim=self.fp_fdim, data_name=self.data_name)
        ligand_x = self.encoder.forward(mol_batch)
        fp_x = self.mlp_fp(mol_batch.fp_x.to(self.device).to(torch.float32))

        ligand_x = self.feature_fusion(torch.stack([ligand_x, fp_x], dim=0))
        x = self.mlp(ligand_x)
        return x
