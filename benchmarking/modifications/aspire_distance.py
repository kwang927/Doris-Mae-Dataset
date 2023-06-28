import pickle
from collections import namedtuple
import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from aspire.examples.ex_aspire_consent_multimatch import AspireConSent, AllPairMaskedWasserstein
from aspire.examples.ex_aspire_consent_multimatch import prepare_abstracts
rep_len_tup = namedtuple('RepLen', ['embed', 'abs_lens'])
ot_distance = AllPairMaskedWasserstein({})

def remove_zero_rows(tensor):
    # Calculate the sum of each row
    row_sums = torch.sum(tensor, dim=1)

    # Find the indices of rows that have non-zero sums
    non_zero_indices = torch.nonzero(row_sums).squeeze()

    # Use the indices to select non-zero rows from the original tensor
    non_zero_rows = tensor[non_zero_indices]
    
    if len(non_zero_rows.size()) == 1:
        return non_zero_rows.unsqueeze(0)

    return non_zero_rows

def find_ot_distance(query_emb, abstract_emb):
    query_emb = torch.tensor(query_emb)
    abstract_emb = torch.tensor(abstract_emb)

    query_emb = remove_zero_rows(query_emb)
    abstract_emb = remove_zero_rows(abstract_emb)

    query_len = query_emb.size()[0]
    query_emb = query_emb.unsqueeze(0)
    qt = rep_len_tup(embed=query_emb.permute(0, 2, 1), abs_lens=[query_len])

    cand_abs_len = abstract_emb.size()[0]
    abstract_emb = abstract_emb.unsqueeze(0)
    ct = rep_len_tup(embed=abstract_emb.permute(0, 2, 1), abs_lens=[cand_abs_len])
    wd, intermediate_items = ot_distance.compute_distance(query=qt, cand=ct, return_pair_sims=True)

    return float(wd)
