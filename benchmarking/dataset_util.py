"""
created by Doris Mae June 13th, 2023
util helper functions to manipulate the DORIS-MAE dataset version 0. 
to create question pairs for annotation
to compute relevance scores between query and abstract
to compute ranked list of candidate pool
to create and extend to a larger sub-query test dataset. 
"""
import json
from tqdm import tqdm
import itertools


def create_aspect_id2annotation(data):
    """
    Input: data with at least the "Annotation" key
    
    Output: return the aspect_id to annotation dictionary, where the key is the aspect_id
            and the value is a list of annotated question pairs whose aspect_id is equal to
            the aforementioned aspect_id
    """
    aspect_id2annotation= {}
    Annotation = data['Annotation']

    for a in Annotation:
        if a['aspect_id'] not in aspect_id2annotation.keys():
            aspect_id2annotation[a['aspect_id']]=[]
        aspect_id2annotation[a['aspect_id']].append(a)
    return aspect_id2annotation

def get_question_pairs(data):
    """
    input: the entire dataset, the dataset needs to have at least 
            4 keys: 'aspect2aspect_id', 'aspect_id2aspect', 'Query', 'Corpus'. 
    output: a list of all question pairs ready for annotation
    """
    aspect2aspect_id = data['aspect2aspect_id']
    aspect_id2aspect = data['aspect_id2aspect']
    Query = data['Query']
    Corpus = data['Corpus']
    ret = []
    
    for q in Query:
        aspect_list = []
        for a in q['aspects'].keys():
            aspect_list.append(a)
            aspect_list += q['aspects'][a]
        aspect_list = list(set(aspect_list))
        
        for aspect_id in aspect_list:
            aspect_id = str(aspect_id)
            for abstract_id in q['candidate_pool']:
                question_pair = {}
                question_pair['aspect_id']= aspect_id
                question_pair['aspect'] = aspect_id2aspect[aspect_id]
                question_pair['abstract_id'] = abstract_id
                question_pair['abstract'] = Corpus[abstract_id]['original_abstract']
                
                ret.append(question_pair)
    return ret

def compute_relevance_score(abstract_id, query_id, data, aspect_id2annotation, normalization= False):
    """
    input: abstract_id is the id for the paper abstract
           query_id is the id for the query, note it could a complex query or a subquery, 
           for the case of sub_query, the data needs to have a subquery dataset as well, similar to Query
           which contains only the complex query. 
           data needs to have the following keys for dataset, "Annotation", "Query"
     
    output: the relevance score for the paper abstract conditioned on the query. If normalization is true, return
            the normalized score, i.e. the score divided by the number of aspects/sub-aspects
            
    Exception: if abstract not in the query's candidate pool, raise value error. 
    """
    Query = data['Query']
    
    q = Query[query_id]
    query_candidate_pool = Query[query_id]['candidate_pool']
    
    if abstract_id not in query_candidate_pool:
        raise ValueError("Abstract not in the Query Candidate Pool, Cannot Perform Computation")
    else:
        aspect_list = []
        for a in q['aspects'].keys():
            aspect_list.append(a)
            aspect_list += q['aspects'][a]
        aspect_list = list(set(aspect_list))
        
        ret = 0
        for asp_id in aspect_list:
            for pair in aspect_id2annotation[asp_id]:
                if pair['abstract_id'] == abstract_id:
                    ret += pair['score']
        if normalization:
            return ret/len(aspect_list)
        else:
            return ret
    
def compute_all_relevance_score(data, normalization = False):
    """
    input: data needs to have the following keys for dataset, "Query", "Annotation"
    
    output: return the (normalized) scores for each paper in candidate pool for each query, format
            is list of dictionary, the key of the dictionary is the abstract_id, whose value is the 
            relevance score. 
    """
    Query = data['Query']
    aspect_id2annotation = create_aspect_id2annotation(data)
    
    ret = []
    for query_id in tqdm(range(len(Query))):
        score_dict = {}
        for abstract_id in Query[query_id]['candidate_pool']:
            score_dict[abstract_id] = compute_relevance_score(abstract_id = abstract_id, \
                                                              query_id= query_id, data = data, \
                                                              aspect_id2annotation= aspect_id2annotation,\
                                                             normalization = normalization)
        ret.append(score_dict)
    return ret

def compute_all_gpt_score(data):
    """
    input: data needs to have the following keys for dataset, "Query", "Annotation"
    
    output: return the original and normalized scores for each paper in candidate pool for each query, format
            is list of dictionary, the key of the dictionary is the abstract_id, whose value is the 
            relevance score.
    """
    original_gpt_score = compute_all_relevance_score(data, normalization = False)
    normalized_gpt_score = compute_all_relevance_score(data, normalization = True)
    
    result_dict = []
    for ind in range(len(original_gpt_score)):
        temp_dict = {}
        for abs_id in original_gpt_score[ind].keys():
            temp_dict[abs_id] = (original_gpt_score[ind][abs_id], normalized_gpt_score[ind][abs_id])
        result_dict.append(temp_dict)
    return result_dict

def rank_candidate_pool(score_list):
    """
    score_list: score_list contains a list of dictionary, 
                the key of the dictionary is the abstract_id, whose value is the 
                relevance score. 
    
    output: return a list of list, where the sublist is the sorted abstract_ids in a candidate pool
            in descending order by the scores.
    
    """
    ret =[]
    for score_dict in score_list:
        sorted_keys = sorted(score_dict, key= score_dict.get, reverse =True)
        ret.append(sorted_keys)
    return ret


def create_subquery_dataset(data, num_aspect =2):
    """
    Input: data needs to have at least 1 key, "Query"
    
    Output: return a subquery dataset where each subquery is created by the concatenation
            of the corresponding sentences for any k combination of aspects (not sub-aspects)
            (in sequential order). 
    """
    Query = data['Query']
    
    ret = []
    
    for q in Query:
        aspect_list = list(q['aspects'].keys())
        all_combination = list(itertools.combinations(aspect_list, num_aspect))
        all_combination = [list(i) for i in all_combination]
        
        for comb in all_combination:
            curr_sub_query = {}
            curr_text_list = []
            for asp in comb:
                for sent in q['aspect_id2sent'][asp]:
                    if sent not in curr_text_list:
                        curr_text_list.append(sent)
            curr_sub_query['query_text'] = " ".join(curr_text_list)
            curr_sub_query['query_type'] = q['query_type']
            curr_sub_query['candidate_pool'] = q['candidate_pool']
            curr_sub_query['sent2aspect_id'] = {}
            for k in q['sent2aspect_id'].keys():
                if k in curr_text_list:
                    curr_sub_query['sent2aspect_id'][k] = q['sent2aspect_id'][k]
                    
            aspect_id_list = []
            for k in curr_sub_query['sent2aspect_id'].keys():
                aspect_id_list += curr_sub_query['sent2aspect_id'][k]
            if len(set(aspect_id_list)) > num_aspect:
                continue
                    
            curr_sub_query['aspect_id2sent'] = {}
            for k in q['aspect_id2sent'].keys():
                if k in comb:
                    curr_sub_query['aspect_id2sent'][k] = q['aspect_id2sent'][k]
            
            curr_sub_query['aspects'] = {}
            for k in q['aspects'].keys():
                if k in comb:
                    curr_sub_query['aspects'][k] = q['aspects'][k]
                    
            ret.append(curr_sub_query)
            
    new_data = {}
    for k in data.keys():
        new_data[k] = data[k]
            
    new_data['Query'] = ret
            
    return new_data


def create_aspect_dataset(data):
    """
    Input: data needs to have at least 1 key, "Query"
    
    Output: return a subquery dataset where each subquery is created by the concatenation
            of the corresponding sentences for any k combination of aspects (not sub-aspects)
            (in sequential order). 
    """
    Query = data['Query']
    aspect_id2aspect  = data['aspect_id2aspect']
    
    ret = []
    
    for q in Query:
        aspect_id_list = list(q['aspects'].keys())
        aspect_list = [aspect_id2aspect[aspect_id] for aspect_id in aspect_id_list]
        
        curr_query = {}
        
        for k in q.keys():
            curr_query[k] = q[k]
            
        curr_query['query_text'] = " ".join(aspect_list)
        
                    
        ret.append(curr_query)
        
    new_data = {}
    for k in data.keys():
        new_data[k] = data[k]
            
    new_data['Query'] = ret
        
    return new_data



# if __name__ == "__main__":
#     with open("DORIS_MAE_dataset_v0.json","r") as f:
#         data = json.load(f)
        
#     all_question_pairs = get_question_pairs(data)
#     all_scores = compute_all_relevance_score(data)
#     rank = rank_candidate_pool(all_scores)
#     sub_2 = create_subquery_dataset(data, num_aspect=2)
#     all_scores_sub2 = compute_all_relevance_score({"Query": sub_2, "aspect2annotation": aspect2annotation})

