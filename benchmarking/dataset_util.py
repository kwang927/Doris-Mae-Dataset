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

import random


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

def create_individual_aspect_dataset(data):
    """
    
    """
    Query = data['Query']
    aspect_id2aspect  = data['aspect_id2aspect']
    aspect2aspect_id = data['aspect2aspect_id']
    
    ret = []
    
    for q in Query:
        aspect_id_list = list(q['aspects'].keys())
        aspect_list = [aspect_id2aspect[aspect_id] for aspect_id in aspect_id_list]
        for individual_aspect in aspect_list:
        
            curr_query = {}

            for k in q.keys():
                if k != "aspects":
                    curr_query[k] = q[k]
                else:
                    individual_aspect_id = aspect2aspect_id[individual_aspect]
                    individual_subaspect_list = q["aspects"][individual_aspect_id]
                    tmp_dict = {individual_aspect_id:individual_subaspect_list}
                    curr_query[k] = tmp_dict
                    
            
            curr_query['query_text'] = individual_aspect
        
                    
            ret.append(curr_query)
        
    new_data = {}
    for k in data.keys():
        new_data[k] = data[k]
            
    new_data['Query'] = ret
        
    return new_data


# def create_10_candidate_dataset(data):
    
#     random.seed(42)
#     gpt_result = compute_all_gpt_score(data)
#     sorted_gpt_result = []
#     get_candidate_pool_list = []
#     for each_dict in gpt_result:
#         sorted_dict = dict(sorted(each_dict.items(), key=lambda x: x[1][1], reverse = True))
#         relavance_abs = [each for each in sorted_dict.items() if each[1][1] >= 1]
#         candidate_pool_dict = {}
#         if len(relavance_abs) >= 3:
#             non_rel_abs = [each for each in sorted_dict.items() if each[1][1] < 1]
#             random_rel = random.sample(relavance_abs, 3)
#             sorted_random_rel = sorted(random_rel, key=lambda x: x[1][1], reverse = True)
#             random_non_rel = random.sample(non_rel_abs, 7)
#             sorted_random_non_rel = sorted(random_non_rel, key=lambda x: x[1][1], reverse = True)
#             combined_list = sorted_random_rel + sorted_random_non_rel
#             for abstract in combined_list:
#                 candidate_pool_dict[abstract[0]] = abstract[1]
#         else:
#             rel = [each for each in list(sorted_dict.items())[:3]]
#             non_rel = [each for each in list(sorted_dict.items())[3:]]
#             random_non_rel = random.sample(non_rel, 7)
#             sorted_random_non_rel = sorted(random_non_rel, key=lambda x: x[1][1], reverse = True)
#             combined_list = rel + sorted_random_non_rel
#             for abstract in combined_list:
#                 candidate_pool_dict[abstract[0]] = abstract[1]
#         get_candidate_pool_list.append(candidate_pool_dict)
#         sorted_gpt_result.append(sorted_dict)
    
    
#     new_data = {}
#     for k in data.keys():
#         new_data[k] = data[k]
        
#     for i, q in enumerate(new_data['Query']):
#         new_candi = get_candidate_pool_list[i]
#         q['candidate_pool'] = list(new_candi.keys())
    
#     return new_data

# def create_10_candidate_dataset_sub(data, list_to_follow):
#     smaller_dataset = create_10_candidate_dataset(data)

#     new_queries = []

#     for ind in list_to_follow:
#         new_queries.append(smaller_dataset['Query'][ind])
#     smaller_dataset['Query'] = new_queries

#     return smaller_dataset

def create_60_query_dataset(data):
    query_list = [1,8,10,16,23,28,33,37,44,46] + list(range(50, 100))
    
    new_data = {}
    for k in data.keys():
        new_data[k] = data[k]
        
    new_data['Query'] = [data["Query"][q] for q in query_list]
    
    return new_data

def put_aspect_into_query_and_annotation(dataset):
    queries = dataset['Query'].copy()
    annotation = dataset['Annotation'].copy()
    
    aspect_id2aspect = dataset['aspect_id2aspect']
    for q in queries:
        aspect_in_str = {}
        sent2aspect_str = {}
        aspect_str2sent = {}
        curr_aspect_in_id = q['aspects']
        curr_aspect_id2sent = q['aspect_id2sent']
        curr_sent2aspect_id = q['sent2aspect_id']
        
        for k in curr_aspect_in_id:
            aspect_in_str[aspect_id2aspect[k]] = [aspect_id2aspect[asp_id] for asp_id in curr_aspect_in_id[k]]
        
        for k in curr_aspect_id2sent:
            aspect_str2sent[aspect_id2aspect[k]] = curr_aspect_id2sent[k]
            
        for k in curr_sent2aspect_id:
            sent2aspect_str[k] = [aspect_id2aspect[asp_id] for asp_id in curr_sent2aspect_id[k]]
            
        q['aspect_in_str'] = aspect_in_str
        q['sent2aspect_str'] = sent2aspect_str
        q['aspect_str2sent'] = aspect_str2sent

        
    for a in annotation:
        a['aspect_str'] = aspect_id2aspect[a['aspect_id']]
        
    dataset['Query'] = queries
    dataset['Annotation'] = annotation
        
    return dataset

def combine_datasets(dataset1, dataset2):
    aspect_id2aspect_dataset1 = dataset1['aspect_id2aspect'].copy()
    aspect_id2aspect_dataset2 = dataset2['aspect_id2aspect'].copy()
    
    aspect2id_dataset1 = dataset1['aspect2aspect_id'].copy()
    aspect2id_dataset2 = dataset2['aspect2aspect_id'].copy()
    
    aspect_str_list = []
    
    for k in aspect2id_dataset1:
        aspect_str_list.append(k)
        
    for k in aspect2id_dataset2:
        aspect_str_list.append(k)
        
    aspect_str_list = sorted(list(set(aspect_str_list)))
    print(f"Merged {len(aspect2id_dataset1)+ len(aspect2id_dataset2)-len(aspect_str_list)} aspect(s).")
    
    count = 0
    
    new_aspect_id2aspect = {}
    new_aspect2aspect_id = {}
    
    for s in aspect_str_list:
        new_aspect_id2aspect[str(count)] = s
        new_aspect2aspect_id[s] = str(count)
        count += 1
        
#     print(len(new_aspect2aspect_id))
#     print(len(new_aspect_id2aspect))
#     print(len(aspect_str_list))
#     print(new_aspect_id2aspect)

    dataset1_with_str = put_aspect_into_query_and_annotation(dataset1)
    dataset2_with_str = put_aspect_into_query_and_annotation(dataset2)
    
    new_dataset = {}
    
    new_dataset['aspect2aspect_id'] = new_aspect2aspect_id
    new_dataset['aspect_id2aspect'] = new_aspect_id2aspect
    new_dataset['Corpus'] = dataset1['Corpus']
    new_dataset['Query'] = dataset1_with_str['Query'].copy() + dataset2_with_str['Query'].copy()
    new_dataset['Annotation'] = dataset1_with_str['Annotation'].copy() + dataset2_with_str['Annotation'].copy()
    if 'Test_set' in dataset1.keys():
        new_dataset['Test_set'] = dataset1['Test_set']
    elif 'Test_set' in dataset2.keys():
        new_dataset['Test_set'] = dataset2['Test_set']
    else:
        print("There is no Test_set")
        
    
    
    for q in new_dataset['Query']:
        new_aspect = {}
        new_sent2asp = {}
        new_asp2sent = {}
        
        aspect_in_str = q['aspect_in_str']
        sent2asp_str = q['sent2aspect_str']
        asp_str2sent = q['aspect_str2sent']
        
        for k in aspect_in_str:
            new_aspect[new_aspect2aspect_id[k]] = [new_aspect2aspect_id[asp_str] for asp_str in aspect_in_str[k]]
            
        for k in sent2asp_str:
            new_sent2asp[k] = [new_aspect2aspect_id[asp_str] for asp_str in sent2asp_str[k]]
            
        for k in asp_str2sent:
            new_asp2sent[new_aspect2aspect_id[k]] = asp_str2sent[k]
            
        q['aspects'] = new_aspect
        q['sent2aspect_id'] = new_sent2asp
        q['aspect_id2sent'] = new_asp2sent
        del q['aspect_in_str']
        del q['sent2aspect_str']
        del q['aspect_str2sent']
        
    for a in new_dataset['Annotation']:
        a['aspect_id'] = new_aspect2aspect_id[a['aspect_str']]
        del a['aspect_str']
        
    
    return new_dataset
            


# if __name__ == "__main__":
#     with open("DORIS_MAE_dataset_v0.json","r") as f:
#         data = json.load(f)
        
#     all_question_pairs = get_question_pairs(data)
#     all_scores = compute_all_relevance_score(data)
#     rank = rank_candidate_pool(all_scores)
#     sub_2 = create_subquery_dataset(data, num_aspect=2)
#     all_scores_sub2 = compute_all_relevance_score({"Query": sub_2, "aspect2annotation": aspect2annotation})

