import pickle
with open("./dataset/embeddings/query/embedding_llama_paragraph_mean_max_1.pickle", "rb") as f:
    data1 = pickle.load(f)
    
with open("./dataset/embeddings/query/embedding_llama_paragraph_mean_max_2.pickle", "rb") as f:
    data2 = pickle.load(f)
    
data = dict(list(data1.items())+list(data2.items()))

with open("./dataset/embeddings/query/embedding_llama_paragraph_mean_max.pickle", "wb") as f:
    pickle.dump(data, f)