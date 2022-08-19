import numpy as np
import pandas as pd
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.xmeans import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import SIMPLE_SAMPLES
from sklearn import cluster, preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy


## Reading and Extracting Data
filename = "Network_clean_backdoor_downloader_v2.csv"
data = pd.read_csv(filename)
data = data[['DestIP', 'Dport', 'Protocol', 'number_of_flows','total_size_of_flows_orig',
'total_size_of_flows_resp', 'inbound_pckts', 'outbound_pckts', 'url_path_length',
'number_of_URL_query_parameters', 'filename_length','number_of_downloaded_bytes', 'number_of_uploaded_bytes','#Src_IP','goal']]
data = data.dropna()
goals = data['goal']
DestIP = data['DestIP']
data = data.drop('goal',axis = 1)


## Normalising data
sc=preprocessing.StandardScaler()
sc.fit(data)
sample=sc.transform(data)
reduced_DestIP = sample[:,0]
host_dict = dict(zip(reduced_DestIP, DestIP))


### C Plane-clustering using xmeans
amount_initial_centers = 5
initial_centers = kmeans_plusplus_initializer(sample, amount_initial_centers).initialize()
xmeans_instance = xmeans(data = sample, initial_centers =initial_centers ,kmax = 50,ccore=True)
xmeans_instance.process()
clusters = xmeans_instance.get_clusters()
centers = xmeans_instance.get_centers()
Cplane_dict = dict(zip(reduced_DestIP, [set() for i in range(len(reduced_DestIP))]))
for index,x in enumerate(clusters):
    y = [reduced_DestIP[i] for i in x]
    for i in range(len(reduced_DestIP)):
        if reduced_DestIP[i] in y:
            Cplane_dict[reduced_DestIP[i]].add(index)


##### A plane-clustering based on suspicious activity
default_value = set()
A1 =set()
A2 =set()
Aplane_dict = dict(zip(reduced_DestIP, [set() for i in range(len(reduced_DestIP))]))
for index,x in enumerate(goals):
    if x == 'normal':
        continue
    elif x == 'backdoor':
        Aplane_dict[reduced_DestIP[index]].add(0)
        A1.add(reduced_DestIP[index])
    else:
        Aplane_dict[reduced_DestIP[index]].add(1)
        A2.add(reduced_DestIP[index])
Aclusters = []
Aclusters.append(list(A1))
Aclusters.append(list(A2))

#Cross Plane Corelation

#total-hosts
hosts = list(set(sample[:,0]))
print("Total No of hosts: ",len(hosts))

## step1:- filtering out hosts h which we have witnessed at least one kind of suspicious activity
s = set()
l=[]
for index,x in enumerate(sample) :
    if goals.values[index]!='normal':
        s.add(x[0])
        l.append(x[0])
sus_hosts = list(s)


##step2:- we first compute a botnet score s(h) for each of the above suspicious hosts
S = len(set(A1)&set(A2))/len(set(A1)|set(A2))
scores = []
for h in sus_hosts:
    score =0
    C_h = list(Cplane_dict[h])
    A_h = list(Aplane_dict[h])
    for i in A_h:
        for k in C_h:
            Ai=set(Aclusters[i])
            Ck=set([reduced_DestIP[l] for l in clusters[k]])
            inter = Ai&Ck
            union = Ai|Ck
            score = score + len(inter)/len(union)
    if len(A_h)==2:
        score=score+S
    scores.append(score)


##step3:- We filter out the hosts that have a score below a certain detection threshold θ
#small threshold: 0.15
#large threshold: 0.41
most_sus = []
for index,x in enumerate(scores):
    if x > 0.15:
        most_sus.append(sus_hosts[index])
print("Total No of suspicious hosts: ",len(most_sus))

##step4:- calculating similarity metric among hosts to group them
A = []
C = []
for h in most_sus:
    for index,c in enumerate(clusters):
        x = [reduced_DestIP[i] for i in c]
        if h in x:
            C.append(x)
    for index,a in enumerate(Aclusters):
        if h in a:
            A.append(a)

## describing each bot h ∈ B as a binary vector b(h) is vector of {0,1},whereby the i-th element b(h)_i = 1 if h ∈ K_i , and b(h)_i = 0 otherwise
A = list(set(tuple(x) for x in A))
mB = len(A)
C = list(set(tuple(x) for x in C))
nB = len(C)
A.extend(C)
K=A
metric = []
for h in most_sus:
    P = []
    for k in K:
        if h in k:
            P.append(1)
        else:
            P.append(0)
    metric.append(P)

##  calculating similarity metric
similarity = []
distance = []
for index1,i in enumerate(most_sus):
    sim = []
    dis = []
    for index2,j in enumerate(most_sus):
        if(i==j):
            sim.append(-1 * (mB+1))
            dis.append(mB+1)
        else:
            ans = 0
            for x in range(mB):
                if metric[index1][x]==metric[index2][x]:
                    ans+=1
            ans1 = 0
            for x in range(nB):
                if metric[index1][x+mB]==metric[index2][x+mB]:
                    ans1+=1
            if ans1>=1:
                ans = ans+1
            sim.append(-1*ans)
            dis.append(ans)
    similarity.append(dis)
    distance.append(sim)

##step5:- using above definition of similarity between hosts, we apply hierarchical clustering.
## This allows to build a dendrogram, i.e., a tree like graph that encodes the relationships among the bots
##n_clusters = 200 for small thtreshold,7 for large threshold.
model = AgglomerativeClustering(affinity='precomputed', n_clusters=200, linkage='complete').fit(distance)
linkage_matrix = hierarchy.linkage(model.children_)
dn = hierarchy.dendrogram(linkage_matrix)

##step6:- used the Davies-Bouldin (DB) validation index to find the best dendrogram cut
##This produces the most compact and well separated clusters. The obtained clusters group bots in (sub-) botnets
db_index = davies_bouldin_score(similarity, model.labels_)
print(db_index)
plt.axhline(y=db_index,color='r',linestyle='-')
plt.show()
