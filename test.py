import pandas as pd
from pathlib import Path

# from rdchiral.template_extractor import extract_from_reaction

import rdkit.Chem as Chem
import re

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    # return ' '.join(tokens)
    return tokens



def extract_template(reaction):
    try:
        return extract_from_reaction(reaction)
    except KeyboardInterrupt:
        print('Interrupted')
        raise KeyboardInterrupt
    except Exception as e:
        return {
            'reaction_id': reaction['_id'],
            'error': str(e)
        }

# df = pd.read_csv("templates.csv")

# data_path = "data/uspto_50.pickle"
# path = Path(data_path)
# df = pd.read_pickle(path)
# reaction_type = df["reaction_type"]
# types = set()
# for type in reaction_type:
#     types.add(type)

# r_smi = []
# p_smi = []

# reactants_mol = df["reactants_mol"]
# products_mol = df["products_mol"]

# for i, r_mol in enumerate(reactants_mol):
#     reactants_mol[i] = Chem.MolToSmiles(r_mol, canonical=True)

# for i, p_mol in enumerate(products_mol):
#     products_mol[i] = Chem.MolToSmiles(p_mol, canonical=True)
# print(types)
# # print(df["reaction_type"])
# df.to_csv("uspto_50_smi.csv", index=False)
# print(df)



def process_zinc4Bert():
    import os
    import pandas as pd
    import random

    root_path = "data/data/zinc"
    with open("sample_10000000.txt", "w") as f:
        for name in os.listdir(root_path):
            print(name)
            df = pd.read_csv(root_path + "/" + name)
            smiles = df["smiles"]
            s_smi = random.sample(list(smiles), 1000000)    
            for smi in s_smi:

                f.write(smi + "\n")
            print(type(smiles))


# process_zinc4Bert()
def read_uspto_50():
    data_path = "data/uspto_50.pickle"
    path = Path(data_path)
    df = pd.read_pickle(path)
    print(df.columns.values)

# read_uspto_50()



def read_vocab():
    path = "bart_vocab_downstream.txt"
    t2i = dict()
    i2t = dict()     
    with open(path, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            token = line.strip()
            t2i[token] = i
            i2t[i] = token
    return t2i, i2t



def read_templates():
    path = "templates.csv"
    df = pd.read_csv(path)

    t2i, i2t = read_vocab()
    t2i_len = len(t2i)

    for i in range(df.shape[0]):
        smi_t = smi_tokenizer(df["unmapped_template"][i])
        for w in smi_t:
            # if w not in t2i.keys():
            #     print("{%s} not in t2i" % w)
            w_i = t2i.get(w, t2i_len)
            i2t[w_i] = w
            t2i[w] = w_i
            t2i_len = len(i2t)
    print("t2i:", t2i)
    print("-------------------------------------------------------------------")
    print("i2t:", i2t)
    print("-------------------------------------------------------------------")
    print(len(t2i))

# read_templates()


# 去掉原子编号
def function_1(smi):
    # 将带有原子编号的SMILES转换为分子对象
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)

    # 将分子对象转换为没有原子编号的SMILES
    smiles = Chem.MolToSmiles(mol, canonical=True)
    return smiles


def combine_templates_uspto():
    path = "templates.csv"
    df = pd.read_csv(path)

    uspto_50_path = "data/uspto_50.pickle"
    df_50k = pd.read_pickle(uspto_50_path)
    
    dict_data = {
        "reactants_mol": [],
        "products_mol": [],
        "reaction_type": [],
        "set": [],
        "reaction_smiles": [],
        "unmapped_template": [],
        "reaction_smarts": [],
        "reactants": [],
        "products": []
    }

    t2i, i2t = read_vocab()
    t2i_len = len(t2i)

    df_template = []
    for j in range(df.shape[0]):
        df_template.append({
            "reaction_smiles": df["reaction_smiles"][j],
            "unmapped_template": df["unmapped_template"][j],
            "reaction_smarts": df["reaction_smarts"][j],
            "reactants": df["reactants"][j],
            "products": df["products"][j],
            "reac": function_1(df["reactants"][j]),
            "prod": function_1(df["products"][j]),
        })
    count = 0
    # print(df_50k.shape)
    for i in range(df_50k.shape[0]):
        dict_data["reactants_mol"].append(Chem.MolToSmiles(df_50k["reactants_mol"][i], canonical=True))
        dict_data["products_mol"].append(Chem.MolToSmiles(df_50k["products_mol"][i], canonical=True))
        dict_data["reaction_type"].append(df_50k["reaction_type"][i])
        dict_data["set"].append(df_50k["set"][i])

        for k in range(len(df_template)):
            if df_template[k]["reac"] == Chem.MolToSmiles(df_50k["reactants_mol"][i], canonical=True) and df_template[i]["prod"] == Chem.MolToSmiles(df_50k["products_mol"][i], canonical=True):
                dict_data["reaction_smiles"].append(df_template[k]["reaction_smiles"])
                dict_data["unmapped_template"].append(df_template[k]["unmapped_template"])
                dict_data["reaction_smarts"].append(df_template[k]["reaction_smarts"])
                dict_data["reactants"].append(df_template[k]["reactants"])
                dict_data["products"].append(df_template[k]["products"])
            
            else:
                count += 1
                dict_data["reaction_smiles"].append("")
                dict_data["unmapped_template"].append("")
                dict_data["reaction_smarts"].append("")
                dict_data["reactants"].append("")
                dict_data["products"].append("")
    print(count)
    df_combine = pd.DataFrame(dict_data)
    df_combine.to_pickle("uspto_50_template.pickle")
    df_combine.to_csv("uspto_50_template.csv")
    print(df_combine)


# combine_templates_uspto()
# smi = "[CH3:17][S:14](=[O:15])(=[O:16])[N:11]1[CH2:10][CH2:9][NH:8][CH2:13][CH2:12]1"

# mol = Chem.MolFromSmiles(smi)
# smi_new = Chem.MolToSmiles(mol, canonical=True)
# print(smi_new)
# import torch.nn as nn
import torch
# norm = nn.LayerNorm(3)
# batch_norm = nn.BatchNorm1d(3)

# x = torch.randn(4, 2, 3)
# print(x)
# y = norm(x)
# print(y)
# x = x.reshape((-1, 3))
# z = batch_norm(x)
# print(z)
# print(torch.sum(z))

from torch import nn, einsum
from einops import rearrange, repeat

# x = torch.randn(4, 2, 3)
# # y = torch.randn(4, 2, 2, 3)
# z = x.mean(dim=1)

# # z = einsum('b i d, b i j d -> b i j', x, y)

# print(z)
# print(x)


# import rdkit
# import rdkit.Chem as Chem
# import numpy as np

# # smi = "Cc1nc(OCCCCN2CCN(c3cccc4ccc(F)cc34)CC2)nc2c1cccn2C.O"
# # mol = Chem.MolFromSmiles(smi)
# # print(mol)

# # import pandas as pd
# # df = pd.read_pickle("/public_data/oy/dataset/data/template/retro_uspto_50_template.pickle")
# # df.to_csv("/public_data/oy/dataset/data/template/retro_uspto_50_template.csv")


# smi = "O=C1CC[C@H](CN2CCN(CCOc3cc4ncnc(Nc5ccc(F)c(Cl)c5)c4cc3OC3CCCC3)CC2)O1"
# mol = Chem.MolFromSmiles(smi)
# atom_order = list(range(mol.GetNumAtoms()))

# np.random.shuffle(atom_order)
# a = Chem.RenumberAtoms(mol, atom_order)
# a = Chem.MolToSmiles(a, canonical=False)
# print(a)

def read_res():
    path = "result.txt"
    len_count = 0
    count_can_len = 0
    count_can_len_same = 0
    count_source_same = 0
    with open(path, "r") as f:
        lines = f.readlines()

        for i in range(len(lines)):
            if i % 4 == 0:
                pred = lines[i].split(":")[1].strip()
            elif i % 4 == 1:
                targ = lines[i].split(":")[1].strip()
            elif i % 4 == 2:
                source = lines[i].split(":")[1].strip()
            else:
                cor = lines[i].strip()
                if cor == "True" and source == pred:
                    count_source_same += 1
                
                if cor == "False" and len(pred) != len(targ):
                    len_count += 1

                tag = True
                try:
                    pred_can = Chem.MolToSmiles(Chem.MolFromSmiles(pred), canonical=True)
                    targ_can = Chem.MolToSmiles(Chem.MolFromSmiles(targ), canonical=True)
                    tag = (not pred_can == targ_can)
                    # print(tag)
                except Exception as e:
                    tag = True

                if cor == "False" and tag:
                    if len(pred) != len(targ):
                        count_can_len += 1
                    else:
                        count_can_len_same += 1
                    
    print("len_count", len_count)
    print("count_can_len", count_can_len)
    print("count_can_len_same", count_can_len_same)
    print("count_source_same", count_source_same)


def smi_tokens(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re

    # pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    # regex = re.compile(pattern)
    # tokens = [token for token in regex.findall(smi)]
    # assert smi == ''.join(tokens)
    # return tokens
    pattern = "(\[[^\]]+]|Bi|Br?|Ge|Te|Mo|K|Ti|Zr|Y|Na|125I|Al|Ce|Cr|Cl?|Ni?|O|S|Pd?|Fe?|I|b|c|Mn|n|o|s|<unk>|>>|Li|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    if smi != ''.join(tokens):
        print('ERROR:', smi, ''.join(tokens))
    assert smi == ''.join(tokens)
    return tokens

# not_atom_indices_src = []
# atom_indices_src = []

# smi_chars = smi_tokens("C(CCCl)Oc1c(-c2cccc(N(C(NCCCCCCC)=O)C)c2)ccc(CCC(=O)OC)c1")
# for j, cha in enumerate(smi_chars):
#     if (len(cha) == 1 and not cha.isalpha()) or (len(cha) > 1 and cha[0] not in ['[', 'B', 'C']):
#         not_atom_indices_src.append(j)
#     else:
#         atom_indices_src.append(cha)

# print(atom_indices_src)
# print(len(atom_indices_src))
# "O=C1CC[C@H](CN2CCN(CCOc3cc4ncnc(Nc5ccc(F)c(Cl)c5)c4cc3OC3CCCC3)CC2)O1"
a = "O=C1CC[C@H](CN2CCN(CCOc3cc4ncnc(Nc5ccc(F)c(Cl)c5)c4cc3OC3CCCC3)CC2)O1"
print(a[10])

