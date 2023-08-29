from dataset_util import *
from rank_performance_util import get_metric, get_random_baseline, print_random_baseline, get_full_metric_results
from config import *
import json
from model_scoring import *
import argparse
import os


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='benchmark evaluation for Doris Mae',
        epilog='Created by Doris Mae'
    )

    parser.add_argument('-o', '--option', required=True, help='specify whether query option, sub-query or aspect option, format for subquery looks like subquery_k, where k is how many aspects are used, typical is 2')
    parser.add_argument('-c', '--cuda', default= "cpu", help= 'specify cuda ids to be used, format is 1,2,3, or cpu')
    parser.add_argument('-b', '--bs', default = 30, help ='user specified batch size based on their own gpu capability, default is 30, which is tested on GeForce RTX 2080 Titan')
    parser.add_argument('-p', '--e5_path', default=None, required=False, help ='path of e5v3')
    parser.add_argument('-bt', '--bootstrap', required=True, help ='bootstrap option')
    
    args = parser.parse_args()
    option = args.option
    cuda = args.cuda.strip()
    bs = int(args.bs)
    e5v3_path = args.e5_path
    if args.bootstrap == "True":
        bootstrap = True
    else:
        bootstrap = False
    
    if option not in ["query", "aspect"]:
        assert "subquery" in option.split("_"), f"option format error {option}"
    
    # here are all the metrics we used, recall@5, recall@20, Recall-Precision, NDCG (normalized discounted cumulative gain), NDCG with exponential, MRR (mean recirprocal rank), MAP (mean average precsion). 
    test_suites = {'recall': [5, 20], 'r_precision':[None], 'ndcg':[0.1], 'ndcg_exp':[0.1], 'mrr':[10], 'map':[None]}


    print("loading dataset.... may take a while")
    with open("dataset/DORIS-MAE_dataset_v1.json", "r") as f:
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
    
    level = config["level"]
    if os.path.exists(f"{rank_result_path}/{level}/ranking_random.pickle"):
        with open(f"{rank_result_path}/{level}/ranking_random.pickle", "rb") as f:
            random_result_dict = pickle.load(f)
        print_random_baseline(test_suites, random_result_dict)
    else:
        model_name = "ernie"
        rank = rank_by_model(dataset, model_name, config)
        random_result_dict = get_random_baseline(gpt_result, rank, test_suites, trials = 100)
        print_random_baseline(test_suites, random_result_dict)
        with open(f"{rank_result_path}/{level}/ranking_random.pickle", "wb") as f:
            pickle.dump(random_result_dict, f)

    
    for model_name in config["model_name_dict"].keys():
        query_mode = config["model_name_dict"][model_name]["query_mode"]
        abstract_mode = config["model_name_dict"][model_name]["abstract_mode"]
        aggregation = config["model_name_dict"][model_name]["aggregation"]
        rank = rank_by_model(dataset, model_name, config, e5v3_path)
        # For getting individual scores at each time
        if not bootstrap:
            for test_name in test_suites.keys():
                for k in test_suites[test_name]:    
                    value = get_metric(gpt_result, rank, test_name, k)
                    if k != None:
                        print(f"{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation} : {test_name}@{k}  : {value}%")
                    else:
                        print(f"{model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation} : {test_name}  : {value}%")

        # Perform bootstrap
        else:   
            result_dict = {}
            for _ in range(1000):
                boot_indices = np.random.choice(range(len(rank)), len(rank), replace =True)
                new_rank = [rank[idx] for idx in boot_indices]
                new_gpt_result = [gpt_result[idx] for idx in boot_indices]

                for test_name in test_suites.keys():
                    for k in test_suites[test_name]:    
                        ret_list = get_full_metric_results(new_gpt_result, new_rank, test_name, k)
                        if (test_name, k) not in result_dict:
                            result_dict[(test_name, k)] = []
                        result_dict[(test_name, k)].append(np.mean(ret_list)*100)

            
            for test_name in result_dict:
                ave = np.mean(result_dict[test_name])
                std = np.std(result_dict[test_name])
                if test_name[1] != None:
                    print(f"With bootstrap : {model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation}  : {test_name[0]}@{test_name[1]} : {ave:.5f} +- {std:.5f}%")
                else:
                    print(f"With bootstrap : {model_name}_abstract-{abstract_mode}_query-{query_mode}_{aggregation}  : {test_name[0]} : {ave:.5f} +- {std:.5f}%")
            print("-"*65)
        print("-"*75+"\n")    

    
