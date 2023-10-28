def print_config(config):
    print(f"Cuda : {config['cuda']}")
    print(f"Level : {config['level']}")
    print(f"User defined batch size : {config['bs']}")
    print("="*40+ " Models Specifications "+ "="*40)
    print("="*95)
    for model_name in config['model_name_dict'].keys():
        query_mode = config['model_name_dict'][model_name]['query_mode']
        abstract_mode = config['model_name_dict'][model_name]['abstract_mode']
        aggregation = config['model_name_dict'][model_name]['aggregation']
        metric = config['model_name_dict'][model_name]['metric']
        if metric == "ot":
            metric = "wasserstein"
        print(f"Model name : {model_name} ; Query mode : {query_mode} ; Abstract mode : {abstract_mode} ; Metric : {metric} ; Aggregation : {aggregation}")
    print("="*95)
    print("="*95)
          
def create_config(option, cuda, bs):
    model_name_dict =  {}
    # For all options below, there are ada and llama model options, they can be added by providing pickle files following specific format
    if option == "query":
        model_names = ["e5", "ts_aspire", "ot_aspire", "simcse", "specter", "specterv2", "sentbert", "rocketqa", "ance", "simlm", "spladev2", "colbertv2", "scibert", "ernie","bm25", "tfidf"]
    elif "subquery" in option:
        if int(option.split("_")[-1])==2:
            model_names = ["e5", "ts_aspire", "ot_aspire", "simcse", "specter", "specterv2", "specter-ID", "sentbert", "rocketqa", "ance", "simlm", "spladev2", "colbertv2", "ernie","scibert", "scibertID", "bm25", "tfidf"]
        else:
            model_names = ["e5", "ts_aspire", "ot_aspire", "simcse", "specterv2", "sentbert",  "ance", "simlm", "spladev2", "colbertv2", "ernie", "scibert"]
    elif option == "aspect":
        model_names = ["e5", "ts_aspire", "ot_aspire", "simcse", "specter", "specterv2", "sentbert", "rocketqa", "ance", "simlm", "spladev2", "colbertv2", "ernie","scibert","bm25", "tfidf", "scibertID"]
    elif option == "60_query":
        model_names = ["e5", "ts_aspire", "ot_aspire", "simcse", "specter", "specterv2", "sentbert", "rocketqa", "ance", "simlm", "spladev2", "colbertv2", "ernie","scibert","bm25", "tfidf"]
    elif option == "query_candidate_150":
        model_names = ["e5", "ts_aspire", "ot_aspire", "simcse", "specter", "specterv2", "sentbert", "rocketqa", "ance", "simlm", "spladev2", "colbertv2", "ernie","scibert","bm25", "tfidf"]
   
    for model in model_names:
        model_name_dict[model] = {}
    for model in model_names:  # abstract mode creation 
        if model not in ["sentbert", "ance"]:
            model_name_dict[model]["abstract_mode"] = "paragraph"
        elif model in ["sentbert", "ance"]:
            model_name_dict[model]["abstract_mode"] = "sentence"
    for model in model_names:  # query mode creation , note "ernie", "scibert", "simlm", "spladev2", "colbertv2" could have sentence as query mode if desired. 
        if model not in ["sentbert", "ance"]:
            model_name_dict[model]["query_mode"] = "paragraph"
        elif model in ["sentbert", "ance"]:
            model_name_dict[model]["query_mode"] = "sentence"

    for model in model_names:  # metric creation 
        if model not in [ "specter", "ts_aspire", "ot_aspire", "specterv2", "specter-ID"]:
            model_name_dict[model]["metric"] = "cosine"
        elif model in [ "specter", "ts_aspire", "specterv2", "specter-ID"]:
            model_name_dict[model]["metric"] =  "l2"
        elif model in ["ot_aspire"]:
            model_name_dict[model]["metric"] = "ot"

    for model in model_names:  # aggregation
        if model in ["sentbert", "ance"]:
            model_name_dict[model]["aggregation"] = "mean_max"
        else:
            model_name_dict[model]["aggregation"] = "mean_max"
    
    config = {"cuda": cuda, "model_name_dict": model_name_dict, "level": option, "bs": bs}    
    print_config(config)
    return config
