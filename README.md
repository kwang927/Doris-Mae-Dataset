 :# Scientific Document Retrieval using Multi-level Aspect-based queries (DORIS-MAE)

The dataset is released at https://doi.org/10.5281/zenodo.8035110 under the CC-BY-NC. 

In scientific research, the ability to effectively retrieve relevant documents based on complex, multifaceted queries is critical. Existing evaluation datasets for this task are limited, primarily due to the high costs of annotating resources that capture complex queries. 

To address this, we propose a novel task, **Scientific Document Retrieval using Multi-level Aspect-based queries (DORIS-MAE)**, which is designed to handle the complex nature of user queries in scientific research. 

We developed a benchmark dataset within the field of computer science, consisting of 50 complex, human-authored primary query cases. For each primary query, we assembled a collection of 100 relevant documents and produced annotated relevance scores for ranking them. Recognizing the significant labor of expert annotation, we also introduce a scalable framework for evaluating the viability of Large Language Models (LLMs) such as ChatGPT-3.5 for expert-level dataset annotation tasks. 

The application of this framework to annotate the DORIS-MAE dataset resulted in a 500x reduction in cost, without compromising quality. Furthermore, due to the multi-tiered structure of these primary queries, our DORIS-MAE dataset can be extended to over 4000 sub-query test cases without requiring additional annotation. 

We evaluated 10 recent retrieval methods on DORIS-MAE, observing notable performance drops compared to traditional datasets. This highlights DORIS-MAE's challenges and the need for better approaches to handle complex, multifaceted queries in scientific research.
