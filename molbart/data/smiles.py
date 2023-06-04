import argparse
import os
import pickle

import torch
from rdkit import Chem
from rdkit.Chem.rdFMCS import FindMCS
from tqdm import tqdm

import numpy as np
import networkx as nx
import logging
from typing import List


BOND_TYPES = [None,
              Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]

BOND_STEREO = [Chem.rdchem.BondStereo.STEREOE,
               Chem.rdchem.BondStereo.STEREOZ,
               Chem.rdchem.BondStereo.STEREONONE]

def get_graph_from_smiles(smi: str):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        print("smi:", smi, "mol:", mol)
    rxn_graph = RxnGraph(prod_mol=mol)
    # rxn_graph = RxnGraph(frag_mol=mol)

    return rxn_graph


def get_bond_features(bond: Chem.Bond) -> List[int]:
    """Get bond features.

    Parameters
    ----------
    bond: Chem.Bond,
        bond object
    """
    bt = bond.GetBondType()
    bond_features = [int(bt == bond_type) for bond_type in BOND_TYPES[1:]]
    bs = bond.GetStereo()
    bond_features.extend([int(bs == bond_stereo) for bond_stereo in BOND_STEREO])
    bond_features.extend([int(bond.GetIsConjugated()), int(bond.IsInRing())])

    return bond_features


def get_graph_features_from_smi(smi):

    atom_features = []
    bond_types = []
    edges = []

    smi = smi.replace(' ', '')
    if not smi.strip():
        smi = "CC"          # hardcode to ignore
    graph = get_graph_from_smiles(smi).prod_mol

    mol = graph.mol
    assert mol.GetNumAtoms() == len(graph.G_dir)

    G = nx.convert_node_labels_to_integers(graph.G_dir, first_label=0)

    # node iteration to get sparse atom features
    for v, attr in G.nodes(data="label"):
        atom_feat = get_atom_features_sparse(mol.GetAtomWithIdx(v),
                                             use_rxn_class=False,
                                             rxn_class=graph.rxn_class)
        atom_features.append(atom_feat)
    bond_fea_num = 9
    nodes = len(G.nodes)
    edges = torch.zeros(nodes, nodes, bond_fea_num)

    # edges = [[[0] * bond_fea_num] * (nodes)] * (nodes)

    # temp_adj = torch.ones(1, nodes)
    # temp_adj2 = torch.ones(nodes+1, 1)
    adjacencys = torch.zeros(nodes, nodes)
    # adjacencys = torch.cat((temp_adj, adjacencys), dim=0)
    # adjacencys = torch.cat((temp_adj2, adjacencys), dim=1)

    # get bond type and edge
    for u, v, attr in G.edges(data='label'):
        bond_feat = torch.tensor(get_bond_features(mol.GetBondBetweenAtoms(u, v)))
        edges[u][v] = bond_feat
        edges[v][u] = bond_feat
        adjacencys[v][u] = 1
        adjacencys[u][v] = 1
        adjacencys[u][u] = 1
        adjacencys[v][v] = 1

    # atom_features = torch.tensor(atom_features, dtype=torch.float32)
    # bond_types = torch.tensor(bond_types, dtype=torch.int64)
    # edges = torch.tensor(edges, dtype=torch.int64)

    tokens = smi_tokens(smi)
    # smile2node = torch.zeros(nodes, len(tokens) + 3)
    # smile2node = torch.zeros(nodes, len(tokens) + 2)
    # for atom in mol.GetAtoms():
    #     atom.SetAtomMapNum(atom.GetIdx() + 1)
    # smi = Chem.MolToSmiles(mol)
    # smi_ts = smi_tokens(smi)
    # for i, tok in enumerate(smi_ts):
    #     if len(tok.strip().split(":")) == 2:
    #         # smile2node[int(tok.strip().split(":")[1].split("]")[0]) - 1][i + 2] = 1
    #         smile2node[int(tok.strip().split(":")[1].split("]")[0]) - 1][i + 1] = 1

    adjacencys = adjacencys.numpy().tolist()
    edges = edges.tolist()
    # smile2node = smile2node.numpy().tolist()

    # return atom_features, edges, adjacencys, smile2node
    return atom_features, edges, adjacencys



def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re

    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)



def smi_tokens(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re

    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return tokens


def get_atom_map(src, tgt):
    src = smi_tokenizer(src)
    tgt = smi_tokenizer(tgt)

    src_chars = src.strip().split(' ')
    tgt_chars = tgt.strip().split(' ')
    if src[0] == '<':
        src_smi = ''.join(src_chars[1:])
    else:
        src_smi = ''.join(src_chars)
    tgt_smi = ''.join(tgt_chars)
    tgt_mols = Chem.MolFromSmiles(tgt_smi)
    tgt_smis = tgt_smi.split('.')
    src_mol = Chem.MolFromSmiles(src_smi)
    # atom_map = torch.zeros(src_mol.GetNumAtoms(), tgt_mols.GetNumAtoms())
    atom_map = np.zeros((src_mol.GetNumAtoms(), tgt_mols.GetNumAtoms()), dtype=int)
    # cross_attn = torch.zeros(len(src_chars), len(tgt_chars))
    cross_attn = np.zeros((len(src_chars), len(tgt_chars)), dtype=int)
    not_atom_indices_src = list()
    atom_indices_src = list()
    pad_indices_src = list()
    not_atom_indices_tgt = list()
    atom_indices_tgt = list()
    pad_indices_tgt = list()
    for smi in tgt_smis:
        tgt_mol = Chem.MolFromSmiles(smi)
        mols = [src_mol, tgt_mol]
        result = FindMCS(mols, timeout=10)
        result_mol = Chem.MolFromSmarts(result.smartsString)
        src_mat = src_mol.GetSubstructMatches(result_mol)
        tgt_mat = tgt_mols.GetSubstructMatches(result_mol)
        if len(src_mat) > 0 and len(tgt_mat) > 0:
            for i, j in zip(src_mat[0], tgt_mat[0]):
                atom_map[i, j] = 1
    # match = atom_map.sum(0)
    # for i in range(match.size(0)):
    #     if match[i] == 0:
    #         atom_map[:, i] = 1

    for j, cha in enumerate(src_chars):
        if (len(cha) == 1 and not cha.isalpha()) or (len(cha) > 1 and cha[0] not in ['[', 'B', 'C']):
            not_atom_indices_src.append(j)
        else:
            atom_indices_src.append(j)
    for j, cha in enumerate(tgt_chars):
        if (len(cha) == 1 and not cha.isalpha()) or (len(cha) > 1 and cha[0] not in ['[', 'B', 'C']):
            not_atom_indices_tgt.append(j)
        else:
            atom_indices_tgt.append(j)
    for x in range(len(src_chars)):
        for y in range(len(tgt_chars)):
            if x in pad_indices_src or y in pad_indices_tgt:
                cross_attn[x, y] = 0
            elif x in atom_indices_src and y in atom_indices_tgt:
                cross_attn[x, y] = atom_map[atom_indices_src.index(x), atom_indices_tgt.index(y)]
            elif x in not_atom_indices_src and y in not_atom_indices_tgt:
                cross_attn[:, y] = 0
                cross_attn[x, :] = 0
                cross_attn[x, y] = 0
    cross_attn = cross_attn.tolist()
    return cross_attn



# Symbols for different atoms
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
             'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti',
             'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb',
             'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Ta', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir',
             'Ce', 'Gd', 'Ga', 'Cs', '*', 'unk']
ATOM_DICT = {symbol: i for i, symbol in enumerate(ATOM_LIST)}

MAX_NB = 10
DEGREES = list(range(MAX_NB))
HYBRIDIZATION = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3,
                 Chem.rdchem.HybridizationType.SP3D,
                 Chem.rdchem.HybridizationType.SP3D2]
HYBRIDIZATION_DICT = {hb: i for i, hb in enumerate(HYBRIDIZATION)}

FORMAL_CHARGE = [-1, -2, 1, 2, 0]
FC_DICT = {fc: i for i, fc in enumerate(FORMAL_CHARGE)}

VALENCE = [0, 1, 2, 3, 4, 5, 6]
VALENCE_DICT = {vl: i for i, vl in enumerate(VALENCE)}

NUM_Hs = [0, 1, 3, 4, 5]
NUM_Hs_DICT = {nH: i for i, nH in enumerate(NUM_Hs)}

CHIRAL_TAG = [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
              Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
              Chem.rdchem.ChiralType.CHI_UNSPECIFIED]
CHIRAL_TAG_DICT = {ct: i for i, ct in enumerate(CHIRAL_TAG)}

RS_TAG = ["R", "S", "None"]
RS_TAG_DICT = {rs: i for i, rs in enumerate(RS_TAG)}

BOND_TYPES = [None,
              Chem.rdchem.BondType.SINGLE,
              Chem.rdchem.BondType.DOUBLE,
              Chem.rdchem.BondType.TRIPLE,
              Chem.rdchem.BondType.AROMATIC]
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],
}

BOND_STEREO = [Chem.rdchem.BondStereo.STEREOE,
               Chem.rdchem.BondStereo.STEREOZ,
               Chem.rdchem.BondStereo.STEREONONE]

BOND_DELTAS = {-3: 0, -2: 1, -1.5: 2, -1: 3, -0.5: 4, 0: 5, 0.5: 6, 1: 7, 1.5: 8, 2: 9, 3: 10}
BOND_FLOATS = [0.0, 1.0, 2.0, 3.0, 1.5]

RXN_CLASSES = list(range(10))

# ATOM_FDIM = len(ATOM_LIST) + len(DEGREES) + len(FORMAL_CHARGE) + len(HYBRIDIZATION) \
#             + len(VALENCE) + len(NUM_Hs) + 1
ATOM_FDIM = [len(ATOM_LIST), len(DEGREES), len(FORMAL_CHARGE), len(HYBRIDIZATION), len(VALENCE),
             len(NUM_Hs), len(CHIRAL_TAG), len(RS_TAG), 2]
# BOND_FDIM = 6
BOND_FDIM = 9
BINARY_FDIM = 5 + BOND_FDIM
INVALID_BOND = -1


def get_atom_features_sparse(atom: Chem.Atom, rxn_class: int = None, use_rxn_class: bool = False) -> List[int]:
    """Get atom features as sparse idx.

    Parameters
    ----------
    atom: Chem.Atom,
        Atom object from RDKit
    rxn_class: int, None
        Reaction class the molecule was part of
    use_rxn_class: bool, default False,
        Whether to use reaction class as additional input
    """
    feature_array = []
    symbol = atom.GetSymbol()
    symbol_id = ATOM_DICT.get(symbol, ATOM_DICT["unk"])
    feature_array.append(symbol_id)

    if symbol in ["*", "unk"]:
        padding = [999999999] * len(ATOM_FDIM) if use_rxn_class else [999999999] * (len(ATOM_FDIM) - 1)
        feature_array.extend(padding)

    else:
        degree_id = atom.GetDegree()
        if degree_id not in DEGREES:
            degree_id = 9
        formal_charge_id = FC_DICT.get(atom.GetFormalCharge(), 4)
        hybridization_id = HYBRIDIZATION_DICT.get(atom.GetHybridization(), 4)
        valence_id = VALENCE_DICT.get(atom.GetTotalValence(), 6)
        num_h_id = NUM_Hs_DICT.get(atom.GetTotalNumHs(), 4)
        chiral_tag_id = CHIRAL_TAG_DICT.get(atom.GetChiralTag(), 2)

        rs_tag = atom.GetPropsAsDict().get("_CIPCode", "None")
        rs_tag_id = RS_TAG_DICT.get(rs_tag, 2)

        is_aromatic = int(atom.GetIsAromatic())
        feature_array.extend([degree_id, formal_charge_id, hybridization_id,
                              valence_id, num_h_id, chiral_tag_id, rs_tag_id, is_aromatic])

        if use_rxn_class:
            feature_array.append(rxn_class)

    return feature_array


def get_bond_features(bond: Chem.Bond) -> List[int]:
    """Get bond features.

    Parameters
    ----------
    bond: Chem.Bond,
        bond object
    """
    bt = bond.GetBondType()
    bond_features = [int(bt == bond_type) for bond_type in BOND_TYPES[1:]]
    bs = bond.GetStereo()
    bond_features.extend([int(bs == bond_stereo) for bond_stereo in BOND_STEREO])
    bond_features.extend([int(bond.GetIsConjugated()), int(bond.IsInRing())])

    return bond_features



from rdkit import Chem
from typing import List, Tuple, Union


def get_sub_mol(mol, sub_atoms):
    new_mol = Chem.RWMol()
    atom_map = {}
    for idx in sub_atoms:
        atom = mol.GetAtomWithIdx(idx)
        atom_map[idx] = new_mol.AddAtom(atom)

    sub_atoms = set(sub_atoms)
    for idx in sub_atoms:
        a = mol.GetAtomWithIdx(idx)
        for b in a.GetNeighbors():
            if b.GetIdx() not in sub_atoms:
                continue
            bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
            bt = bond.GetBondType()
            if a.GetIdx() < b.GetIdx():  # each bond is enumerated twice
                new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)

    return new_mol.GetMol()


class RxnGraph:
    """
    RxnGraph is an abstract class for storing all elements of a reaction, like
    reactants, products and fragments. The edits associated with the reaction
    are also captured in edit labels. One can also use h_labels, which keep track
    of atoms with hydrogen changes. For reactions with multiple edits, a done
    label is also added to account for termination of edits.
    """

    def __init__(self,
                 prod_mol: Chem.Mol = None,
                 frag_mol: Chem.Mol = None,
                 reac_mol: Chem.Mol = None,
                 rxn_class: int = None) -> None:
        """
        Parameters
        ----------
        prod_mol: Chem.Mol,
            Product molecule
        frag_mol: Chem.Mol, default None
            Fragment molecule(s)
        reac_mol: Chem.Mol, default None
            Reactant molecule(s)
        rxn_class: int, default None,
            Reaction class for this reaction.
        """
        if prod_mol is not None:
            self.prod_mol = RxnElement(mol=prod_mol, rxn_class=rxn_class)
        if frag_mol is not None:
            self.frag_mol = MultiElement(mol=frag_mol, rxn_class=rxn_class)
        if reac_mol is not None:
            self.reac_mol = MultiElement(mol=reac_mol, rxn_class=rxn_class)
        self.rxn_class = rxn_class

    def get_attributes(self, mol_attrs: Tuple = ('prod_mol', 'frag_mol', 'reac_mol')) -> Tuple:
        """
        Returns the different attributes associated with the reaction graph.

        Parameters
        ----------
        mol_attrs: Tuple,
            Molecule objects to return
        """
        return tuple(getattr(self, attr) for attr in mol_attrs if hasattr(self, attr))


class RxnElement:
    """
    RxnElement is an abstract class for dealing with single molecule. The graph
    and corresponding molecule attributes are built for the molecule. The constructor
    accepts only mol objects, sidestepping the use of SMILES string which may always
    not be achievable, especially for a unkekulizable molecule.
    """

    def __init__(self, mol: Chem.Mol, rxn_class: int = None) -> None:
        """
        Parameters
        ----------
        mol: Chem.Mol,
            Molecule
        rxn_class: int, default None,
            Reaction class for this reaction.
        """
        self.mol = mol
        self.rxn_class = rxn_class
        self._build_mol()
        self._build_graph()

    def _build_mol(self) -> None:
        """Builds the molecule attributes."""
        self.num_atoms = self.mol.GetNumAtoms()
        self.num_bonds = self.mol.GetNumBonds()
        self.amap_to_idx = {atom.GetAtomMapNum(): atom.GetIdx()
                            for atom in self.mol.GetAtoms()}
        self.idx_to_amap = {value: key for key, value in self.amap_to_idx.items()}

    def _build_graph(self) -> None:
        """Builds the graph attributes."""
        self.G_undir = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        self.G_dir = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))

        for atom in self.mol.GetAtoms():
            self.G_undir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()
            self.G_dir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()

        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = BOND_TYPES.index(bond.GetBondType())
            self.G_undir[a1][a2]['label'] = btype
            self.G_dir[a1][a2]['label'] = btype
            self.G_dir[a2][a1]['label'] = btype

        self.atom_scope = (0, self.num_atoms)
        self.bond_scope = (0, self.num_bonds)

    def update_atom_scope(self, offset: int) -> Union[List, Tuple]:
        """Updates the atom indices by the offset.

        Parameters
        ----------
        offset: int,
            Offset to apply
        """
        # Note that the self. reference to atom_scope is dropped to keep self.atom_scope non-dynamic
        if isinstance(self.atom_scope, list):
            atom_scope = [(st + offset, le) for st, le in self.atom_scope]
        else:
            st, le = self.atom_scope
            atom_scope = (st + offset, le)

        return atom_scope

    def update_bond_scope(self, offset: int) -> Union[List, Tuple]:
        """Updates the bond indices by the offset.

        Parameters
        ----------
        offset: int,
            Offset to apply
        """
        # Note that the self. reference to bond_scope is dropped to keep self.bond_scope non-dynamic
        if isinstance(self.bond_scope, list):
            bond_scope = [(st + offset, le) for st, le in self.bond_scope]
        else:
            st, le = self.bond_scope
            bond_scope = (st + offset, le)

        return bond_scope


class MultiElement(RxnElement):
    """
    MultiElement is an abstract class for dealing with multiple molecules. The graph
    is built with all molecules, but different molecules and their sizes are stored.
    The constructor accepts only mol objects, sidestepping the use of SMILES string
    which may always not be achievable, especially for an invalid intermediates.
    """

    def _build_graph(self) -> None:
        """Builds the graph attributes."""
        self.G_undir = nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))
        self.G_dir = nx.DiGraph(Chem.rdmolops.GetAdjacencyMatrix(self.mol))

        for atom in self.mol.GetAtoms():
            self.G_undir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()
            self.G_dir.nodes[atom.GetIdx()]['label'] = atom.GetSymbol()

        for bond in self.mol.GetBonds():
            a1 = bond.GetBeginAtom().GetIdx()
            a2 = bond.GetEndAtom().GetIdx()
            btype = BOND_TYPES.index(bond.GetBondType())
            self.G_undir[a1][a2]['label'] = btype
            self.G_dir[a1][a2]['label'] = btype
            self.G_dir[a2][a1]['label'] = btype

        frag_indices = [c for c in nx.strongly_connected_components(self.G_dir)]
        self.mols = [get_sub_mol(self.mol, sub_atoms) for sub_atoms in frag_indices]

        atom_start = 0
        bond_start = 0
        self.atom_scope = []
        self.bond_scope = []

        for mol in self.mols:
            self.atom_scope.append((atom_start, mol.GetNumAtoms()))
            self.bond_scope.append((bond_start, mol.GetNumBonds()))
            atom_start += mol.GetNumAtoms()
            bond_start += mol.GetNumBonds()

from molbart.data.rdchiral.template_extractor import extract_from_reaction, get_changed_atoms, mols_from_smiles_list, \
    replace_deuterated
import re

def smi_tokenizer_list(smi):
    """Tokenize a SMILES sequence or reaction"""
    pattern = "(\[[^\]]+]|Bi|Br?|Ge|Te|Mo|K|Ti|Zr|Y|Na|125I|Al|Ce|Cr|Cl?|Ni?|O|S|Pd?|Fe?|I|b|c|Mn|n|o|s|<unk>|>>|Li|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    if smi != ''.join(tokens):
        print('ERROR:', smi, ''.join(tokens))
    assert smi == ''.join(tokens)
    return tokens

def canonical_smiles_with_am(smi):
    """Canonicalize a SMILES with atom mapping"""
    atomIdx2am, pivot2atomIdx = {}, {}
    mol = Chem.MolFromSmiles(smi)
    atom_ordering = []       # 有序的原子数组
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):          # 去除Number属性
            atomIdx2am[atom.GetIdx()] = atom.GetProp('molAtomMapNumber')          # 原子序号：映射的下标
            atom.ClearProp('molAtomMapNumber')
        else:
            atomIdx2am[atom.GetIdx()] = '0'
        atom_ordering.append(atom.GetIdx())

    unmapped_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=atom_ordering, canonical=False)    # 包含atomsToUse的子结构
    mol = Chem.MolFromSmiles(unmapped_smi)
    cano_atom_ordering = list(Chem.CanonicalRankAtoms(mol))      # ？

    for i, j in enumerate(cano_atom_ordering):
        pivot2atomIdx[j + 1] = i                                 # Atom canonical 下的排名: 序号
        mol.GetAtomWithIdx(i).SetIntProp('molAtomMapNumber', j + 1)  #

    new_tokens = []
    for token in smi_tokenizer_list(Chem.MolToSmiles(mol)):
        if re.match('.*:([0-9]+)]', token):
            pivot = re.match('.*(:[0-9]+])', token).group(1)
            # pivot2atomIdx[int(pivot[1:-1])] 原子在Atom中的序号
            token = token.replace(pivot, ':{}]'.format(atomIdx2am[pivot2atomIdx[int(pivot[1:-1])]]))
        new_tokens.append(token)

    canonical_smi = ''.join(new_tokens)
    # canonical reactants order
    if '.' in canonical_smi:
        canonical_smi_list = canonical_smi.split('.')
        canonical_smi_list = sorted(canonical_smi_list, key=lambda x: (len(x), x))
        canonical_smi = '.'.join(canonical_smi_list)
    return canonical_smi


def get_nonreactive_mask(cano_prod_am, raw_prod, raw_reacts, radius=0):
    """Retrieve the ground truth reaction center by RDChiral"""
    reactants = mols_from_smiles_list(replace_deuterated(raw_reacts).split('.'))
    # print(reactants)
    products = mols_from_smiles_list(replace_deuterated(raw_prod).split('.'))
    changed_atoms, changed_atom_tags, err = get_changed_atoms(reactants, products)
    # print(changed_atom_tags)

    for _ in range(radius):
        mol = Chem.MolFromSmiles(cano_prod_am)
        changed_atom_tags_neighbor = []
        for atom in mol.GetAtoms():
            if atom.GetSmarts().split(':')[1][:-1] in changed_atom_tags:
                for n_atom in atom.GetNeighbors():
                    changed_atom_tags_neighbor.append(n_atom.GetSmarts().split(':')[1][:-1])
        changed_atom_tags = list(set(changed_atom_tags + changed_atom_tags_neighbor))
    # print(changed_atom_tags)

    nonreactive_mask = []
    for i, token in enumerate(smi_tokenizer_list(cano_prod_am)):
        if token[0] == '[' and token[-1] == ']' and re.match('.*:([0-9]+)]', token):
            am = re.match('.*:([0-9]+)]', token).group(1)
            if am in changed_atom_tags:
                nonreactive_mask.append(False)
                continue
        nonreactive_mask.append(True)

    if sum(nonreactive_mask) == len(nonreactive_mask):  # if the reaction center is not detected
        nonreactive_mask = [False] * len(nonreactive_mask)
    # print(nonreactive_mask)
    return nonreactive_mask