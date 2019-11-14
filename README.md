# Topic Modeling

Data has been removed, names of files and fields have been generalized.

GOAL: Map problem descriptions in one dataset to trouble tickets in a second dataset to expedite diagnoses

METHOD:
1. Preprocessing: Remove stop words, lemmatization
2. Filter to most common unigrams
3. Vectorization by Bag-of-words or TF-IDF
4. Generate topics by Singular Value Decomp or Latent Dirichlet Allocation
5. Cluster problem descriptions and solutions
6. Trouble ticket mapped to k nearest neighbors by Euclidean or Mahalanobis distance
7. n problems are sampled from the problem clusters of dataset 2
8. Those solutions are mapped to clusters and aggregated to make recommendation
