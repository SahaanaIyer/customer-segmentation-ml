import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def grouping() :
    matrix = df.pivot_table(index='Customer Last Name', columns='Offer #', values='n')
    matrix.fillna(0, inplace=True)
    matrix.reset_index(inplace=True)
    return matrix

def clustering(matrix) :
    cluster = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
    matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[1:]])
    return cluster

def princi_comp_anal(matrix) :
    pca = PCA(n_components=2, random_state=0)
    matrix['x'] = pca.fit_transform(matrix[matrix.columns[1:]])[:, 0]
    matrix['y'] = pca.fit_transform(matrix[matrix.columns[1:]])[:, 1]
    clusters = matrix.iloc[:, [0, 33, 34, 35]]
    return clusters

#Loading the dataset_1 (OfferInformation)
offers = pd.read_excel(path, sheet_name=0)
#Loading the dataset_2 (Transactions)
transactions = pd.read_excel(path, sheet_name=1)
transactions['n'] = 1
#Merging the two datasets into df
df = pd.merge(left=offers, right=transactions, how='inner')
print(df.head())

#Creating an Offer-Transaction pivot table
matrix = grouping()
print(matrix.head())

#Using KMeans to cluster the data
cluster = clustering(matrix)
print(matrix.head())

#Visualizing clusters using PCA
clusters = princi_comp_anal(matrix)
print(clusters)
clusters.plot.scatter(x='x', y='y', c='cluster', colormap='viridis')
plt.show()

#To find which cluster orders the most 'Champagne'
data = pd.merge(left=clusters, right=transactions, how='inner')
data = pd.merge(left=offers, right=data, how='inner')
champagne = {}
for i in range(0,5) :
    new_df = data
    counts = new_df['Varietal'].value_counts(ascending=False)
    if counts.index[0] == 'Champagne':
         champagne[i] = counts[0]
cluster_champagne = max(champagne, key=champagne.get)
print("The cluster that orders the most 'Champagne' is {}.".format(cluster_champagne))

#To find which cluster of customers favours discounts more on an average
discount = {}
for i in data.cluster.unique() :
    new_df = data[data.cluster == i]
    counts = (new_df['Discount (%)'].values.sum()) / len(new_df)
    discount[i] = counts
cluster_discount = max(discount, key=discount.get)
print("The cluster of customers which favour discounts more on an average is {}.".format(cluster_discount))

# OUTPUT :
# The cluster that orders the most 'Champagne' is 0.
# The cluster of customers which favour discounts more on an average is 4.