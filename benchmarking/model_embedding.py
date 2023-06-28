import pickle
import torch
import os
from math import *
import re
import json
import sys
import numpy as np
from itertools import chain
from tqdm import tqdm
from nltk import word_tokenize, sent_tokenize
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer, util


'''
import for ColBERT
'''
sys.path.append('./ColBERT')
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.modeling.hf_colbert import class_factory
from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message, load_checkpoint


'''
import for Aspire
'''
from aspire.examples.ex_aspire_consent import AspireConSent, prepare_abstracts


def preprocessing(text):
    '''
    Note, this preprocessing function should be applied to any text before getting its embedding. 
    Input: text as string
    
    Output: removing newline, latex $$, and common website urls. 
    '''
    pattern = r"(((https?|ftp)://)|(www.))[^\s/$.?#].[^\s]*"
    text = re.sub(pattern, '', text)
    return text.replace("\n", " ").replace("$","")

def load_model(model_name, cuda =None):
    """
    loading models/tokenizers based on model_name, also based on cuda option specify whether DataParallel.
    Input: cuda option is a string, e.g. "1,3,5" specify cuda1, cuda3, and cuda5 will be used, store parameters on cuda1. 
    """
    if model_name == "ernie":
        tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-large-en")
        model = AutoModel.from_pretrained("nghuyong/ernie-2.0-large-en")
    elif model_name == "simlm":
        model = AutoModel.from_pretrained('intfloat/simlm-base-msmarco-finetuned')
        tokenizer = AutoTokenizer.from_pretrained('intfloat/simlm-base-msmarco-finetuned')
    elif model_name == "spladev2":
        model = AutoModel.from_pretrained('naver/splade_v2_distil')
        tokenizer = AutoTokenizer.from_pretrained('naver/splade_v2_distil')
    elif model_name == "scibert":
        tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    elif model_name == "colbertv2":
        with Run().context(RunConfig(nranks=len(cuda.split(",")), experiment='notebook')):
            config = ColBERTConfig(doc_maxlen=1000, nbits=2)
        config = ColBERTConfig.from_existing(ColBERTConfig.load_from_checkpoint('ColBERT/colbertv2.0'))
        HF_ColBERT = class_factory('bert-base-uncased')
        model = HF_ColBERT.from_pretrained('ColBERT/colbertv2.0', config).LM
        tokenizer = AutoTokenizer.from_pretrained('ColBERT/colbertv2.0')
    elif model_name in ["ot_aspire", "ts_aspire"]:
        hf_model_name = 'allenai/aspire-contextualsentence-singlem-compsci'
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        model = AspireConSent(hf_model_name)
        if cuda!= "cpu": 
            cuda = cuda.split(",")[0]
    elif model_name == "sentbert":
        model = SentenceTransformer('all-mpnet-base-v2')
        tokenizer = None
        if cuda!= "cpu": 
            cuda = cuda.split(",")[0]
    elif model_name == "ance":
        model = SentenceTransformer('sentence-transformers/msmarco-roberta-base-ance-firstp')
        tokenizer = None
        if cuda!= "cpu": 
            cuda = cuda.split(",")[0]
        

    if cuda!= "cpu":
#         os.environ['CUDA_VISIBLE_DEVICES'] = cuda
        torch.cuda.set_device(int(cuda.split(",")[0]))
        
        model.to("cuda")
        cuda_list = cuda.split(",")
        cuda_num = len(cuda_list)
        if cuda_num>1:
            model = torch.nn.DataParallel(model, device_ids = [int(idx) for idx in cuda_list])
    else:
        print("Running model on CPU...")
    num_params= sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{model_name} model parameters: {int(num_params/1000000)} millions.")
    return model, tokenizer
def optimal_batch_size(model_name, model, tokenizer, text, cuda_num, cuda, bs):
    """
    Input model: loaded model
          tokenizer: associated tokenizer
          text: a list of strings, each string is either a query or an abstract
          cuda_num: int, number of cuda available
          bs: bs is the user defined maximum batch size to try
    Return a stable batch_size that could be used for given configuration. 
    """
    if model_name == "ernie": # ernie size is 300 million, usually 3 times bigger than other models
        bs = bs//3
    if model_name in ["sentbert", "ance"]:
        len_token =[]
        for t in tqdm(text, desc= "Finding optimal batch size", leave = False):
            inputs=word_tokenize(preprocessing(t))
            len_token.append(len(inputs))
        sample_text = preprocessing(text[np.argmax(len_token)])
        assert max(len_token)<512, "maximum tokneized length greater than BERT allowed 512"
        print(f"maximum tokenized text length {max(len_token)}")
        batch_size = cuda_num *bs*5
    else:
        len_token =[]
        for t in tqdm(text, desc= "Finding optimal batch size", leave = False):
            inputs=tokenizer(preprocessing(t))
            len_token.append(len(inputs["input_ids"]))
        sample_text = preprocessing(text[np.argmax(len_token)])
        assert max(len_token)<512, "maximum tokneized length greater than BERT allowed 512"
        print(f"maximum tokenized text length {max(len_token)}")
        batch_size = cuda_num *bs
    while batch_size>0:
        sample_batch = [sample_text]*batch_size
        try: 
            print(f"Trying batch size : {batch_size}")
            inputs = tokenization(model_name, tokenizer, sample_batch)
            embedding = encoding(model_name, model, inputs, cuda)
            print(f"Optimal batch size is {batch_size}")
            del inputs, embedding
            torch.cuda.empty_cache()
            return batch_size
        except RuntimeError as e:
            if "memory" in str(e).lower():
                del inputs
                torch.cuda.empty_cache()
                batch_size -= cuda_num
            else:
                raise RuntimeError("Other issues in the model implementation")
    raise ValueError("Cuda memory does not support batch processing")
def get_embedding(model_name, model, tokenizer, text, cuda= "cpu", batch_size= 30):
    """
    Input model: loaded model
          tokenizer: associated tokenizer
          text: a list of strings, each string is either a query or an abstract
          cuda: in the format of "0,1,6,7" or "0", by default, cpu option is used
          batch_size: if not specified, then an optimal batch_size is found by system, else, 
                       the user specified batch_size is used, may run into OOM error.
    Return:  the embedding dictionary, where the key is a string (e.g. an abstract, query/subquery), and the value
             is np.ndarray of the vector, usually 1 or 2 dimensions. 
    """
    if cuda != "cpu":
        if model_name in ["ot_aspire", "ts_aspire", "sentbert", "ance"]: 
            cuda = cuda.split(",")[0]
#             cuda = cuda[0] # aspire, sentbert, ance do not currently support multiple gpus.
        cuda_list = cuda.split(",")
        cuda_num = len(cuda_list)
        
        batch_size = optimal_batch_size(model_name, model, tokenizer, text, cuda_num, cuda, batch_size)
        
        length = ceil(len(text)/batch_size)
    else:
        batch_size = 1
    
    ret = {}  
    length = ceil(len(text)/batch_size)    
    for i in tqdm(range(length), desc = "Begin Embedding...", leave = False):
        curr_batch = text[i*batch_size:(i+1)*batch_size]
        curr_batch_cleaned = [preprocessing(t) for t in curr_batch]
        inputs = tokenization(model_name, tokenizer, curr_batch_cleaned)
        embedding = encoding(model_name, model, inputs, cuda)
        for t, v in zip(curr_batch, embedding):
            ret[t] = v
        del inputs
        torch.cuda.empty_cache()
    return ret
def tokenization(model_name, tokenizer, text):
    '''
    Different tokenization procedures based on different models.
    
    Input: text as list of strings, if cpu option then list has length 1.
    Return: tokenized inputs, could be dictionary for BERT models. 
    '''
    if model_name in ["sentbert", "ance"]:
        return text
    elif model_name in ["ernie", "simlm", "spladev2", "scibert", "colbertv2"]:
        inputs = tokenizer(text, padding=True, truncation=True, max_length=1000, return_tensors="pt")
    elif model_name in ["ot_aspire", "ts_aspire"]:
        ret = [{'TITLE': " ", 'ABSTRACT': sent_tokenize(i)} for i in text]
        inputs = prepare_abstracts(batch_abs=ret, pt_lm_tokenizer=tokenizer)

    return inputs 


def encoding(model_name, model, inputs, cuda):
    '''
    Different encoding procedures based on different models. 
    Input: inputs are tokenized inputs in specific form
    Return: a numpy ndarray embedding on cpu. 
    
    '''
    if cuda != "cpu":
        device = "cuda"
    else:
        device = "cpu"
    with torch.no_grad():
        if model_name in ["ernie", "scibert"]:
            input_ids =inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            embeddings = model(input_ids, attention_mask = attention_mask)
            output = embeddings.pooler_output.detach().cpu()
            del input_ids, attention_mask, embeddings
            torch.cuda.empty_cache()
        elif model_name == "simlm":
            input_ids =inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            embeddings = model(input_ids, attention_mask = attention_mask)
            output = embeddings.last_hidden_state[:, 0, :].detach().cpu()
            del input_ids, attention_mask, embeddings
            torch.cuda.empty_cache()
        elif model_name == "spladev2":
            input_ids =inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            embeddings = model(input_ids, attention_mask = attention_mask)
            output = (torch.sum(embeddings.last_hidden_state * attention_mask.unsqueeze(-1), dim=1) /\
                                                      torch.sum(attention_mask, dim=-1, keepdim=True)).detach().cpu()
            del input_ids, attention_mask, embeddings
            torch.cuda.empty_cache()
        elif model_name == "colbertv2":
            input_ids =inputs['input_ids'].to(device).squeeze()
            attention_mask = inputs['attention_mask'].to(device).squeeze()
            embeddings = model(input_ids, attention_mask)
            output = embeddings.pooler_output.detach().cpu()
            del input_ids, attention_mask, embeddings
            torch.cuda.empty_cache()
        elif model_name in ["ot_aspire", "ts_aspire"]:
            bert_batch, abs_lens, sent_token_idxs = inputs
            clsreps, contextual_sent_reps = model.forward(bert_batch=bert_batch, \
                                                                    abs_lens=abs_lens, sent_tok_idxs=sent_token_idxs)
            output = contextual_sent_reps.detach().cpu()
            del bert_batch, abs_lens, sent_token_idxs, clsreps, contextual_sent_reps 
            torch.cuda.empty_cache()
        elif model_name in ["sentbert", "ance"]:
            embeddings = model.encode(inputs)
            output = embeddings
            del embeddings
            torch.cuda.empty_cache()
            return output
    return output.numpy()


def pre_encoding_by_mode(text, mode = "paragraph"):
    '''
    Input: text as a list of strings, waiting to be embedded
           model: default is processing by paragraph, for models such as sentbert or ance, or by user preference
                  another mode is "sentence", where each string is splits into sentences, and return a flattened
                  list of sentences, with too short sentences are eliminated. 
    Return: list of strings, where strings of less than 5 chars are removed if a model is processing it sentence by sentence.
    '''
    if mode == "paragraph":
        return text
    elif mode == "sentence":
        sent_text = list(chain.from_iterable([sent_tokenize(t) for t in text]))
        ret = [s for s in sent_text if len(preprocessing(s))>=5]
        return ret
    else:
        raise ValueError("only supports paragraph or sentence processing modes")
    

    
    