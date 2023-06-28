from dataset_util import *
from rank_performance_util import get_metric
from config import *
import json
from model_scoring import *
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='benchmark evaluation for Doris Mae',
        epilog='Created by Doris Mae'
    )

    parser.add_argument('-o', '--option', required=True, help='specify whether query option, sub-query or aspect option, format for subquery looks like subquery_k, where k is how many aspects are used, typical is 2')
    parser.add_argument('-c', '--cuda', default= "cpu", help= 'specify cuda ids to be used, format is 1,2,3, or cpu')
    parser.add_argument('-b', '--bs', default = 30, help ='user specified batch size based on their own gpu capability, default is 30, which is tested on GeForce RTX 2080 Titan')
    
    args = parser.parse_args()
    option = args.option
    cuda = args.cuda.strip()
    bs = int(args.bs)
    if option not in ["query", "aspect"]:
        assert "subquery" in option.split("_"), "option format error"
    
    # here are all the metrics we used, recall@5, recall@20, Recall-Precision, NDCG (normalized discounted cumulative gain), NDCG with exponential, MRR (mean recirprocal rank), MAP (mean average precsion). 
    test_suites = {'recall': [5,20], 'r_precision':[None], 'ndcg':[0.1], 'ndcg_exp':[0.1], 'mrr':[10], 'map':[None]}


    print("loading dataset.... may take a while")
    with open("dataset/DORIS_MAE_dataset_v0.json", "r") as f:
            dataset = json.load(f)
            print(f"raw dataset size {len(dataset['Query'])}")
            
    if "subquery" in option:
        print(f"creating {option} dataset....")
        num_aspect = int(option.split("_")[-1])
        dataset = create_subquery_dataset(dataset, num_aspect = num_aspect)
        print(f"{option} dataset size {len(dataset['Query'])}")
    elif option == "aspect":
        print(f"creating aspect dataset...")
        dataset = create_aspect_dataset(dataset)
        print(f"{option} dataset size {len(dataset['Query'])}")


    # calculate ground truth ranking for the candidate pool of each query, as provided by GPT
    gpt_result = compute_all_gpt_score(dataset)

    # create a configuration dictionary
    config = create_config( option,cuda,  bs)

    for model_name in config["model_name_dict"].keys():
        query_mode = config["model_name_dict"][model_name]["query_mode"]
        abstract_mode = config["model_name_dict"][model_name]["abstract_mode"]
        aggregation = config["model_name_dict"][model_name]["aggregation"]
        rank = rank_by_model(dataset, model_name, config)
        for test_name in test_suites.keys():
            for k in test_suites[test_name]:    
                value = get_metric(gpt_result, rank, test_name, k)
                if k != None:
                    print(f"{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation} : {test_name}@{k}  : {value}%")
                else:
                    print(f"{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation} : {test_name}  : {value}%")

        print("-"*75+"\n")    

    