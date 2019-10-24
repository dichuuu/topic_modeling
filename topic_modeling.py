
# coding: utf-8

# # Topic Mapping
# + GOAL: Map problem descriptions in TAAS-INFO to trouble tickets in MRDB to expedite diagnoses
# + METHOD: 
# > 1. Preprocessing: Remove stop words, lemmatization, 
# > 2. Filter to most common unigrams
# > 3. Vectorization by Bag-of-words or TF-IDF
# > 4. Generate topics by Singular Value Decomp or Latent Dirichlet Allocation
# > 5. Trouble ticket mapped to nearest neighbor by Euclidean or Mahalanobis distance

# In[ ]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
import numpy as np
from numpy.linalg import svd
import spacy
from sklearn.decomposition import LatentDirichletAllocation

nlp = spacy.load('en_core_web_sm')

nlp.vocab.add_flag(lambda s: s.lower() in stopwords.words('english'), spacy.attrs.IS_STOP)


# In[ ]:


mrdb_filename = "Filename1.xlsx"
taas_filename = "Filename2.xls"
mrdb_field = "Problem Description Fieldname"
taas_field = "Problem Description Fieldname"
mrdb_solution_field = "Solution Fieldname"
taas_solution_field = "Solution Fieldname"

mrdb_df = pd.read_excel(mrdb_filename)
taas_df = pd.read_excel(taas_filename)


# In[ ]:


words_to_consider = 5000
topics_to_consider = 500


# In[ ]:


threshold = 100
long_mrdb_df = mrdb_df[mrdb_df[mrdb_field].str.len() > threshold]
long_mrdb_df.reset_index(inplace=True)
long_taas_df = taas_df[taas_df[taas_field].str.len() > threshold]
long_taas_df.reset_index(inplace=True)


# In[ ]:


taas_tokenized = []
for d in long_taas_df[taas_field]:
    taas_tokenized.append(nlp(d))


# In[ ]:


mrdb_tokenized = []
for d in long_mrdb_df[mrdb_field]:
    mrdb_tokenized.append(nlp(d))


# In[ ]:


common_unigrams = defaultdict(int)
lemmatized = []
for doc in mrdb_tokenized:
    text = []
    for t in doc:
        if t.is_stop or t.is_punct or t.is_space: 
            continue
        if t.lemma_ == "'s":
            continue
        common_unigrams[t.lemma_] += 1
        text.append(t.lemma_)
    lemmatized.append(' '.join(text))
    
#long_mrdb_df[mrdb_field] = lemmatized


# In[ ]:


lemmatized = []
for doc in taas_tokenized:
    text = []
    for t in doc:
        if t.is_stop or t.is_punct or t.is_space: 
            continue
        if t.lemma_ == "'s":
            continue
        
        text.append(t.lemma_)
    lemmatized.append(' '.join(text))
    
long_taas_df[taas_field] = lemmatized


# In[ ]:


import math

def check(neighbor, idx, data, src_df, src_text, dest_df, dest_text):

    distances, indices = neighbor.kneighbors(np.expand_dims(data[idx], axis=0))
    print(src_df.iloc[idx][src_text])
    prev_dist = 0
    for i in range(len(indices[0])):
        #if math.isclose(prev_dist, distances[0][i]):
        #    continue
        prev_dist = distances[0][i]
        print()
        print("Result {}----------------".format(i+1))
        print("MRDB Trouble Ticket Number: ", dest_df.iloc[indices[0][i]]["MRDB Trouble Ticket Number"], " Distance: ", distances[0][i], "Index: ", indices[0][i])
        print(dest_df.iloc[indices[0][i]][dest_text])


# # Bag-of-Words Vectorization

# In[ ]:


rv = sorted([(v, k) for k, v in common_unigrams.items()], reverse=True)
feature_vec = [v for count, v in rv[:words_to_consider]]
vectorizer = CountVectorizer(vocabulary=feature_vec)
mrdb_data = vectorizer.fit_transform(long_mrdb_df[mrdb_field])
taas_data = vectorizer.fit_transform(long_taas_df[taas_field])


# # TF-IDF Vectorization

# In[ ]:


rv = sorted([(v, k) for k, v in common_unigrams.items()], reverse=True)
feature_vec = [v for count, v in rv[:words_to_consider]]
vectorizer = TfidfVectorizer(vocabulary=feature_vec)
merged = long_mrdb_df[mrdb_field].append(long_taas_df[taas_field])
merged_data = vectorizer.fit_transform(merged)
mrdb_data = merged_data[:len(long_mrdb_df[mrdb_field])]
taas_data = merged_data[len(long_mrdb_df[mrdb_field]):]


# # Latent Dirichlet Allocation Demonstration with 3 Topics

# In[ ]:


lda3 = LatentDirichletAllocation(n_components=3, random_state=0, topic_word_prior=1, doc_topic_prior=1)
lda3.fit(mrdb_data) 


# In[ ]:


# Get the first 100 words in 3 topics

maximums = np.argmax(lda3.exp_dirichlet_component_[:,:], axis=0)
topics = defaultdict(list)

for i in range(100):
    topics[maximums[i]].append(feature_vec[i]) 
    
for i in range(3):
    print("Topic {}". format(i+1), topics[i])
    print()


# In[ ]:


mrdb_transformed = lda3.transform(mrdb_data)
taas_transformed = lda3.transform(taas_data)


# In[ ]:


get_ipython().magic(u'matplotlib inline')


# In[ ]:


# Plot all three topics

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib import rc

blue = (4/255, 47/255, 85/255, 255/255)
red = (218/255, 4/255, 36/255, 255/255)
length, _ = mrdb_transformed.shape


fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(10)
ax = fig.add_subplot(111, projection='3d')


c = blue
m = '.'
xs = taas_transformed[:,0]
ys = taas_transformed[:,1]
zs = taas_transformed[:,2]
ax.scatter(xs, ys, zs, c=c, marker=m, label="TAAS-INFO")

c = red
m = '.'
xs = mrdb_transformed[:,0]
ys = mrdb_transformed[:,1]
zs = mrdb_transformed[:,2]
ax.scatter(xs, ys, zs, c=c, marker=m, label="MRDB")

blue_patch = mpatches.Patch(color=blue, label='TAAS-INFO')
red_patch = mpatches.Patch(color=red, label='MRDB')
plt.legend(handles=[red_patch, blue_patch])
font = {'size'   : 18}
rc('font', **font)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(14)

ax.set_xlabel('Topic 1', fontsize=18, x=0.5,y=-1)
ax.set_ylabel('Topic 2', fontsize=18)
ax.set_zlabel('Topic 3', fontsize=18)



plt.show()


# In[ ]:


# Plot pairwise comparison of topics

import matplotlib.pyplot as plt
import numpy as np
plt.clf()
mc = red
m = '.'
mxs = mrdb_transformed[:,0]
mys = mrdb_transformed[:,1]
mzs = mrdb_transformed[:,2]

tc = blue
m = '.'
txs = taas_transformed[:,0]
tys = taas_transformed[:,1]
tzs = taas_transformed[:,2]


plt.figure(figsize=(3,20))
fig = plt.figure()
fig.set_figheight(3.2)
fig.set_figwidth(13)

ax = fig.add_subplot(131)
ax.scatter(mxs, mys, c=mc, marker=m)
ax.scatter(txs, tys, c=tc, marker=m)
ax.set_xlabel("Topic 1", fontsize=18)
ax.set_ylabel("Topic 2", fontsize=18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

ax2 = fig.add_subplot(132)
ax2.scatter(mys, mzs, c=mc, marker=m)
ax2.scatter(tys, tzs, c=tc, marker=m)
ax2.set_xlabel("Topic 2", fontsize=18)
ax2.set_ylabel("Topic 3", fontsize=18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

ax3 = fig.add_subplot(133)
ax3.scatter(mxs, mzs, c=mc, marker=m)
ax3.scatter(txs, tzs, c=tc, marker=m)
ax3.set_xlabel("Topic 1", fontsize=18)
ax3.set_ylabel("Topic 3", fontsize=18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

plt.subplots_adjust(wspace = 0.3)

blue_patch = mpatches.Patch(color=blue, label='TAAS-INFO')
red_patch = mpatches.Patch(color=red, label='MRDB')
plt.legend(handles=[red_patch, blue_patch])
font = {'size'   : 18}
rc('font', **font)

plt.show()


# # LDA Topic Generation

# In[ ]:


topics_to_consider


# In[ ]:


lda = LatentDirichletAllocation(n_components=topics_to_consider, doc_topic_prior=1, max_iter=100, random_state=0)
lda.fit(mrdb_data) 


# In[ ]:


print(mrdb_transformed[0].min())
print(mrdb_transformed[0].max())


# In[ ]:


mrdb_transformed = lda.transform(mrdb_data)
taas_transformed = lda.transform(taas_data)


# # Nearest Neighbor by Euclidean Distance

# In[ ]:


lda_nbrs = NearestNeighbors(n_neighbors=10, 
                            algorithm='ball_tree', 
                            metric='euclidean').fit(mrdb_transformed)


# In[ ]:


long_mrdb_df[mrdb_solution_field].iloc[3019]


# In[ ]:


check(lda_nbrs, 8554, taas_transformed, long_taas_df, taas_field, long_mrdb_df, mrdb_field)


# # Nearest Neighbor by Mahalanobis Distance

# In[ ]:


vi = np.linalg.pinv(np.cov(mrdb_transformed.T))
lda_maha_nbrs = NearestNeighbors(n_neighbors=10, 
                            algorithm='brute', 
                            metric='mahalanobis', 
                            metric_params={'VI': vi}).fit(mrdb_transformed)


# In[ ]:


check(lda_maha_nbrs, 8554, taas_transformed, long_taas_df, taas_field, long_mrdb_df, mrdb_field)


# # Mahalanobis Distance Example

# In[ ]:


m = np.array([0, 0])
cov = np.array([[9, 0], [0, 1]])
np.random.multivariate_normal(m, cov)


# In[ ]:


x = []
y = []
for i in range(10000):
    x1, y1 =np.random.multivariate_normal(m, cov)
    x.append(x1)
    y.append(y1)


# In[ ]:


import matplotlib.pyplot as plt
pt = [-10, -8, -9]
pty = [0, 0, 1]

plt.figure(figsize=(21, 7))
plt.scatter(x, y, c='black', marker='.')
plt.scatter(pt, pty, c='red', marker='o')
plt.xticks([i-15 for i in range(25)])
plt.annotate('A', (pt[0], pty[0]), fontsize='xx-large', bbox=dict(facecolor='blue', alpha=0.2))
plt.annotate('B', (pt[1], pty[1]), fontsize='xx-large', bbox=dict(facecolor='blue', alpha=0.2))
plt.annotate('C', (pt[2], pty[2]), fontsize='xx-large', bbox=dict(facecolor='blue', alpha=0.2))
plt.show()


# # Singular Value Decomposition Demonstration with 3 Topics

# In[ ]:


u, s, vh = np.linalg.svd(mrdb_data.toarray(), full_matrices=False)


# In[ ]:


smat = np.diag(s)
topic_mat = np.dot(smat, vh)
taas_topic_map = np.dot(taas_data.toarray(), np.linalg.inv(topic_mat)) 


# In[ ]:


# Get topic of first 100 words

maximums = np.argmax(topic_mat[:3,:], axis=0)


topics = defaultdict(list)

for i in range(100):
    topics[maximums[i]].append(feature_vec[i]) 
    
for i in range(3):
    print("Topic {}". format(i+1), topics[i])
    print()


# In[ ]:


# Plot full dataset

length, _ = mrdb_transformed.shape

fig = plt.figure()
fig.set_figheight(10)
fig.set_figwidth(10)
ax = fig.add_subplot(111, projection='3d')
""
c = red
m = '.'
xs = u[:,1]
ys = u[:,2]
zs = u[:,3]
ax.scatter(xs, ys, zs, c=c, marker=m, label="MRDB")

c = blue
m = '.'
xs = taas_topic_map[:,1]
ys = taas_topic_map[:,2]
zs = taas_topic_map[:,3]
ax.scatter(xs, ys, zs, c=c, marker=m, label="TAAS-INFO")

ax.set_xlabel('Topic 1')
ax.set_ylabel('Topic 2')
ax.set_zlabel('Topic 3')


plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
plt.clf()
mc = red
m = '.'
mxs = u[:,1]
mys = u[:,2]
mzs = u[:,3]

tc = blue
m = '.'
txs = taas_topic_map[:,1]
tys = taas_topic_map[:,2]
tzs = taas_topic_map[:,3]


plt.figure(figsize=(3,20))
fig = plt.figure()
fig.set_figheight(4)
fig.set_figwidth(15)

ax = fig.add_subplot(131)
ax.scatter(mxs, mys, c=mc, marker=m)
ax.scatter(txs, tys, c=tc, marker=m)
ax.set_xlabel("Topic 1", fontsize=18)
ax.set_ylabel("Topic 2", fontsize=18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

ax2 = fig.add_subplot(132)
ax2.scatter(mys, mzs, c=mc, marker=m)
ax2.scatter(tys, tzs, c=tc, marker=m)
ax2.set_xlabel("Topic 2", fontsize=18)
ax2.set_ylabel("Topic 3", fontsize=18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

ax3 = fig.add_subplot(133)
ax3.scatter(mxs, mzs, c=mc, marker=m)
ax3.scatter(txs, tzs, c=tc, marker=m)
ax3.set_xlabel("Topic 1", fontsize=18)
ax3.set_ylabel("Topic 3", fontsize=18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)

plt.subplots_adjust(wspace = 0.5)

plt.show()


# # Nearest Neighbor by Euclidean Distance

# In[ ]:


t = topics_to_consider
svd_nbrs = NearestNeighbors(n_neighbors=10, 
                            algorithm='ball_tree', 
                            metric='euclidean').fit(taas_topic_map[:,:t])


# In[ ]:


long_mrdb_df[mrdb_solution_field].iloc[76]


# In[ ]:


check(svd_nbrs, 8554, u[:,:t], long_taas_df, taas_field, long_mrdb_df, mrdb_field)


# # Nearest Neighbor by Mahalanobis Distance

# In[ ]:


svd_maha_nbrs = NearestNeighbors(n_neighbors=10, 
                            algorithm='brute', 
                            metric='mahalanobis',
                            metric_params={'V': np.cov(u[:,:t].T)}).fit(u[:,:t])


# In[ ]:


check(svd_maha_nbrs, 8554, u[:,:t], long_taas_df, taas_field, long_mrdb_df, mrdb_field)


# # Statistics on Solutions

# In[ ]:


# Fix spelling on some mistakes
def fix_spelling(s):
    s = str(s).strip()
    return s.replace('RECIEVED', 'RECEIVED').replace('RECIEVE', 'RECEIVE').replace('RECIVED', 'RECEIVED')

long_mrdb_df['solution'] = long_mrdb_df[mrdb_solution_field].map(lambda x: fix_spelling(x))


# In[ ]:


mrdb_sol_tokenized = []
for d in long_mrdb_df['solution']:
    mrdb_sol_tokenized.append(nlp(d))


# In[ ]:


common_sol_unigrams = defaultdict(int)
lemmatize = []
count_total = 0
replaced_total = 0
null_total = 0
reset_total = 0
extra = []

for doc in mrdb_sol_tokenized:
    count_total += 1
    sol = []
    if doc.__len__() <= 1:
        null_total += 1
    for t in doc:
        if t.is_stop or t.is_punct or t.is_space: 
            continue
        if t.lemma_ == "'s":
            continue
        common_sol_unigrams[t.lemma_] += 1
        sol.append(t.lemma_)
    
    lemmatize.append(' '.join(sol))


# In[ ]:


# Create solution topics

rv = sorted([(v, k) for k, v in common_sol_unigrams.items()], reverse=True)
feature_vec = [v for count, v in rv[:2000]]
vectorizer = TfidfVectorizer(vocabulary=feature_vec)
solution_data = vectorizer.fit_transform(lemmatize)

# Map solution dataset to topics
sol_lda = LatentDirichletAllocation(n_components=200, doc_topic_prior=1, max_iter=50, random_state=0)
sol_lda.fit(solution_data) 

sol_lda_data = sol_lda.transform(solution_data)


# In[ ]:


verbs = defaultdict(int)

for doc in mrdb_sol_tokenized:
    for t in doc:
        if t.pos_ == 'VERB':
            verbs[t.lemma_] += 1


# In[ ]:


sorted_verbs = sorted([(v, k) for k, v in verbs.items()], reverse=True)


# In[ ]:


# sorted list of tuples (counts, verbs)
# this information has been removed
cleaned_sorted = [
 
]


# In[ ]:


import matplotlib.pyplot as plt

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = [x[1] for x in cleaned_sorted]
sizes = [x[0] for x in cleaned_sorted]

fig1, ax1 = plt.subplots(figsize=(20, 20))

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


plt.show()


# In[ ]:



import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(20, 20))

size = 0.3
# vals is list of lists of counts
vals = []
# labels is numpy array of list of verbs
labels = np.array()

sum_vals = [sum(x) for x in vals]
flatten_vals = []
for x in vals:
    flatten_vals += x
    
flatten_labels = []
for x in labels:
    flatten_labels += x
    
cmap = plt.get_cmap("Set3")
ygb_cmap = plt.get_cmap("YlGnBu")
outer_colors = cmap([1, 10, 0, 4])
inner_colors = ygb_cmap(np.arange(28)*10)

ax.pie(sum_vals, radius=1, colors=outer_colors,
       wedgeprops=dict(width=size, edgecolor='w'))

ax.pie(flatten_vals, radius=1-size, labels=flatten_labels, colors=inner_colors, autopct='%1.1f%%',
       wedgeprops=dict(width=size, edgecolor='w'))

ax.set(aspect="equal", title='Solutions')
plt.show()


# # Initial TAAS-INFO to Nearest Solution and Nearest Cluster Solution

# In[ ]:



from sklearn.cluster import KMeans

# Cluster solution topics
sol_kmeans = KMeans(n_clusters=k, random_state=0).fit(sol_lda_data)
sol_cluster_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(sol_kmeans.cluster_centers_)


# In[ ]:


def get_neighbors(neighbor, idx, data, src_df, src_text, dest_df, dest_text):
    distances, indices = neighbor.kneighbors(np.expand_dims(data[idx], axis=0))
    return indices[0]

def get_recommendations(neighbor, indices, dest_df, dest_text, absolute=True):
    """ 
        Absolute True returns the exact solutions applied for the neighbors
        Absolute False returns the solutions of the clusters 
    """
    sols = []
    if absolute:
        for i in range(len(indices)):
            sols.append(dest_df[dest_text].iloc[indices[i]])
        return sols
    else:
        # Map to centroid
        cluster_indices = []
        for i in range(len(indices)):
            topic_sol = sol_lda.transform(solution_data[indices[i]])
            dist, cluster_index = neighbor.kneighbors(topic_sol)
            sols.append(cluster_map[cluster_index[0][0]])
        return sols


# In[ ]:


idc  = get_neighbors(lda_nbrs, 8554, taas_transformed, long_taas_df, taas_field, long_mrdb_df, mrdb_field)


# In[ ]:


# Map TAAS-INFO problem description to neighbors in MRDB and their solutions
absolute_sols = get_recommendations(sol_cluster_nbrs, idc, long_mrdb_df, mrdb_solution_field)


# In[ ]:


# Map TAAS-INFO problem description to neighbors in MRDB and their solution clusters
clusters_sols = get_recommendations(sol_cluster_nbrs, idc, long_mrdb_df, mrdb_solution_field, False)


# In[ ]:


count = 1
print("--- Recommendations ---")
for s in absolute_sols:
    print('{}.'.format(count), s)
    count +=1


# In[ ]:


count = 1
print("--- Recommendations ---")
for s in clusters_sols:
    print('{}.'.format(count), s)
    count +=1


# # Cluster Solution Topics

# In[ ]:



from sklearn import metrics
from scipy.spatial.distance import cdist

distortions = []
K = range(1, 400, 10)
for k in K:
    sol_kmeans = KMeans(n_clusters=k, random_state=0).fit(sol_lda_data)
    # For each point, determine distance to all centroids, determine the closest centroid (min), sum all 
    distortions.append(sum(np.min(cdist(sol_lda_data, sol_kmeans.cluster_centers_, 'euclidean'), axis=1)) / sol_lda_data.shape[0])


# In[ ]:


# Plot the elbow
K = range(1, 400, 10)
plt.plot(K, distortions, 'bo-')
plt.xlabel('K')
plt.ylabel('Distortion')
plt.title('Distortion by K clusters')
plt.show()


# In[ ]:


# Using 70 as best fit for number of clusters
sol_topics = 70
sol_kmeans = KMeans(n_clusters=sol_topics, random_state=0).fit(sol_lda_data)


# # View Clusters

# In[ ]:


from scipy.spatial.distance import cdist
sol_distances = cdist(sol_lda_data, sol_kmeans.cluster_centers_, 'euclidean')


# In[ ]:


sol_cluster_group = np.argmin(sol_distances, axis=1)


# In[ ]:


sol_cluster_dict = defaultdict(list)
for idx, v in enumerate(sol_cluster_group):
    sol_cluster_dict[v].append(idx)


# In[ ]:


sol_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(sol_lda_data)
distances, sol_indices = sol_nbrs.kneighbors(sol_kmeans.cluster_centers_)
cluster_labels ={}
for i in range(len(sol_indices)):
    cluster_labels[i] = str(long_mrdb_df[mrdb_solution_field].iloc[sol_indices[i]].values[0]).strip()


# In[ ]:


cluster_labels


# In[ ]:


def view_cluster(cluster_labels, cluster_dict, cluster_idx, num, data, data_df, data_field):
    """
        cluster_labels: Dict of cluster index to head label
        cluster_dict: Dict of cluster index to elements placed into cluster
        cluster_idx: cluster number
        num: number to sample
        data: full topic dataset
        data_df: Dataframe of the source of solutions
        data_field: The field name for solutions in dataframe
        
    """
    head = cluster_labels[cluster_idx]
    print("HEAD:", head)
    sampled_indices = np.random.choice(cluster_dict[cluster_idx], num, replace=False)
    for idx in sampled_indices:
        print()
        print("Trouble Ticket Number: ", data_df['JCN'].iloc[idx])
        print("Problem Description: ", data_df[data_field].iloc[idx])

view_cluster(cluster_labels, sol_cluster_dict, 1, 10, sol_lda_data, long_mrdb_df, mrdb_solution_field)
    


# In[ ]:


view_cluster(cluster_labels, sol_cluster_dict, 44, 10, sol_lda_data, long_mrdb_df, mrdb_solution_field)


# In[ ]:


for k, v in sol_cluster_dict.items():
    if 5566 in v:
        print(k)


# In[ ]:


long_mrdb_df[long_mrdb_df[mrdb_solution_field].str.contains('PIGTAIL').fillna(False)]


# # Cluster TAAS Problem Topics

# In[ ]:


distortions_prob = []
K = range(1, 400, 10)
for k in K:
    prob_kmeans = KMeans(n_clusters=k, random_state=0).fit(taas_transformed)
    # For each point, determine distance to all centroids, determine the closest centroid (min), sum all 
    distortions_prob.append(sum(np.min(cdist(taas_transformed, prob_kmeans.cluster_centers_, 'euclidean'), axis=1)) / taas_transformed.shape[0])


# In[ ]:


# Plot the elbow
K = range(1, 400, 10)
plt.plot(K, distortions_prob, 'bo-')
plt.xlabel('K')
plt.ylabel('Distortion')
plt.title('Distortion by K clusters')
plt.show()


# In[ ]:


# Using elbow at 80 for the best number of clusters
prob_topics = 80
prob_kmeans = KMeans(n_clusters=prob_topics, random_state=0).fit(taas_transformed)

prob_distances = cdist(taas_transformed, prob_kmeans.cluster_centers_, 'euclidean')
prob_cluster_group = np.argmin(prob_distances, axis=1)

prob_cluster_dict = defaultdict(list)
for idx, v in enumerate(prob_cluster_group):
    prob_cluster_dict[v].append(idx)
    
prob_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(taas_transformed)
distances, prob_indices = prob_nbrs.kneighbors(prob_kmeans.cluster_centers_)
prob_cluster_labels ={}
for i in range(len(prob_indices)):
    prob_cluster_labels[i] = str(long_taas_df[taas_field].iloc[prob_indices[i]].values[0]).strip()


# In[ ]:


prob_cluster_labels


# In[ ]:


view_cluster(prob_cluster_labels, prob_cluster_dict, 50, 10, taas_transformed, long_taas_df, taas_field)


# # Top N Issues Categories

# In[ ]:


n = 10
issue_counts = sorted([(len(v), k) for k, v in prob_cluster_dict.items()], reverse=True)


# In[ ]:


# Pair of count of elements in cluster and cluster index
issue_counts


# In[ ]:


# Print out the top n problem clusters

for count, category in issue_counts[:n]:
    print("Count:", count, "Cluster:", category) 
    print(prob_cluster_labels[category], '\n')


# # Mean Time to Implement Solution Cluster

# In[ ]:


mean_time_field = "Mean Time Fieldname"

sol_implementation_time = {}
for k, v in sol_cluster_dict.items():
    sol_implementation_time[k] = long_mrdb_df[mean_time_field].iloc[v].mean()


# In[ ]:


for t, c in sorted([(v, k) for k, v in sol_implementation_time.items()]):
    print("Mean Time to Resolve:", t, "Cluster:", c)


# In[ ]:


# Solutions to cluster 66 were the fastest

view_cluster(cluster_labels, sol_cluster_dict, 62, 10, sol_lda_data, long_mrdb_df, mrdb_solution_field)


# # Get Solutions for Problem Cluster

# In[ ]:


sol_cluster_nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(sol_kmeans.cluster_centers_)


# In[ ]:


# THIS FUNCTION USES VARIABLES DEFINED OUTSIDE OF THE FUNCTION

def get_solution_recommendation_by_problem_cluster(cluster_idx, p_cluster_dict, num, nearestn, top_n=7):
    """
        cluster_idx: Problem cluster index
        p_cluster_dict: Dict of Problem cluster index to elements in cluster
        num: Int number of elements to sample from problem clusters
        nearestn: Int number of neighbors in MRDB to use, max is 10 based on lda_nbrs variable defined outside of this function
    """
    sol_counts = defaultdict(int)
    sampled_indices = np.random.choice(p_cluster_dict[cluster_idx], num, replace=True)
    for idx in sampled_indices:
        mrdb_indices = get_neighbors(lda_nbrs, idx, taas_transformed, long_taas_df, taas_field, long_mrdb_df, mrdb_field)
        mrdb_indices = mrdb_indices[:nearestn]
        for i in range(len(mrdb_indices)):
            topic_sol = sol_lda.transform(solution_data[mrdb_indices[i]])
            dist, cluster_index = sol_cluster_nbrs.kneighbors(topic_sol)
            sol_counts[cluster_index[0][0]] += 1
    
    labels = []
    sizes = []
    count = 0
    other_total = 0
    sorted_solutions = sorted([(v, k) for k, v in sol_counts.items()], reverse=True)
    for v, k in sorted_solutions:
        if count < top_n:
            labels.append(cluster_labels[k]+ ", {}".format(np.round(sol_implementation_time[k], 2)))
            sizes.append(v)
            count += 1
        else:
            other_total+=v
    if count:
        labels.append("Other")
        sizes.append(other_total)
        
    print(labels)
    
    n_colors = len(labels)
    rate = 1/n_colors
    #ygb_cmap = plt.get_cmap("YlGnBu")
    #inner_colors = ygb_cmap(np.arange(n_colors)*15)
    colors = []
    red = (218/255, 4/255, 36/255, 255/255)
    for i in range(n_colors):
        new_red = list(red)
        new_red[3] -= rate*i

        colors.append(new_red)

        
    fig1, ax1 = plt.subplots(figsize=(20, 20))
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
            shadow=True, startangle=90, textprops={'fontsize': 56})
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.show()

            


# In[ ]:


# Solution clusters for problem cluster 50 (Antennas)

get_solution_recommendation_by_problem_cluster(50, prob_cluster_dict, 100, 5)


# In[ ]:


labels


# In[ ]:


dir(stopwords)


# In[ ]:


print(mrdb_df.size)

