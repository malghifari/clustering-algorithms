import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
from KMeans import KMeans
from Agglomerative import Agglomerative
from sklearn.cluster import KMeans as KMeans_SKLearn
from sklearn.cluster import AgglomerativeClustering

iris = load_iris()
X = iris['data']
y = pd.DataFrame({'label': iris['target']})

N_CLUSTERS = len(iris['target_names'])

# Bikin sendiri
## Single Linkage
agglo = Agglomerative(n_clusters=N_CLUSTERS, linkage='single')
agglo.fit(X)
y_pred = pd.DataFrame({'label': agglo.predict()})

plt.subplot(2, 4, 1)
plt.title('Single Linkage')
pca_pred = PCA(n_components=2).fit_transform(X)
principal_df_pred = pd.DataFrame(data = pca_pred
             , columns = ['principal component 1', 'principal component 2'])
pca_df_pred = pd.concat([principal_df_pred, y_pred], axis = 1)
sns.scatterplot(x="principal component 1", y="principal component 2", hue="label", data=pca_df_pred, legend=False )

## Complete linkage
agglo = Agglomerative(n_clusters=N_CLUSTERS, linkage='complete')
agglo.fit(X)
y_pred = pd.DataFrame({'label': agglo.predict()})

plt.subplot(2, 4, 2)
plt.title('Complete Linkage')
pca_pred = PCA(n_components=2).fit_transform(X)
principal_df_pred = pd.DataFrame(data = pca_pred
             , columns = ['principal component 1', 'principal component 2'])
pca_df_pred = pd.concat([principal_df_pred, y_pred], axis = 1)
sns.scatterplot(x="principal component 1", y="principal component 2", hue="label", data=pca_df_pred, legend=False )

## Average Linkage
agglo = Agglomerative(n_clusters=N_CLUSTERS, linkage='average')
agglo.fit(X)
y_pred = pd.DataFrame({'label': agglo.predict()})

plt.subplot(2, 4, 3)
plt.title('Average Linkage')
pca_pred = PCA(n_components=2).fit_transform(X)
principal_df_pred = pd.DataFrame(data = pca_pred
             , columns = ['principal component 1', 'principal component 2'])
pca_df_pred = pd.concat([principal_df_pred, y_pred], axis = 1)
sns.scatterplot(x="principal component 1", y="principal component 2", hue="label", data=pca_df_pred, legend=False )

## Average-Group Linkage
agglo = Agglomerative(n_clusters=N_CLUSTERS, linkage='average-group')
agglo.fit(X)
y_pred = pd.DataFrame({'label': agglo.predict()})

plt.subplot(2, 4, 4)
plt.title('Average-Group Linkage')
pca_pred = PCA(n_components=2).fit_transform(X)
principal_df_pred = pd.DataFrame(data = pca_pred
             , columns = ['principal component 1', 'principal component 2'])
pca_df_pred = pd.concat([principal_df_pred, y_pred], axis = 1)
sns.scatterplot(x="principal component 1", y="principal component 2", hue="label", data=pca_df_pred, legend=False )

# Scikit Learn

## Single Linkage
agglo_scikit = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='single')
y_pred_sklearn = pd.DataFrame({'label': agglo_scikit.fit_predict(X)})

plt.subplot(2, 4, 5)
plt.title('Scikit-Learn Single Linkage')
pca_2d = PCA(n_components=2).fit_transform(X)
principal_df = pd.DataFrame(data = pca_2d
             , columns = ['principal component 1', 'principal component 2'])
pca_df = pd.concat([principal_df, y_pred_sklearn], axis = 1)
sns.scatterplot(x="principal component 1", y="principal component 2", hue="label", data=pca_df, legend=False)

## Complete Linkage
agglo_scikit = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='complete')
y_pred_sklearn = pd.DataFrame({'label': agglo_scikit.fit_predict(X)})

plt.subplot(2, 4, 6)
plt.title('Scikit-Learn Complete Linkage')
pca_2d = PCA(n_components=2).fit_transform(X)
principal_df = pd.DataFrame(data = pca_2d
             , columns = ['principal component 1', 'principal component 2'])
pca_df = pd.concat([principal_df, y_pred_sklearn], axis = 1)
sns.scatterplot(x="principal component 1", y="principal component 2", hue="label", data=pca_df, legend=False)

## Average Linkage
agglo_scikit = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage='average')
y_pred_sklearn = pd.DataFrame({'label': agglo_scikit.fit_predict(X)})

plt.subplot(2, 4, 7)
plt.title('Scikit-Learn Average Linkage')
pca_2d = PCA(n_components=2).fit_transform(X)
principal_df = pd.DataFrame(data = pca_2d
             , columns = ['principal component 1', 'principal component 2'])
pca_df = pd.concat([principal_df, y_pred_sklearn], axis = 1)
sns.scatterplot(x="principal component 1", y="principal component 2", hue="label", data=pca_df, legend=False)

# Actual
plt.subplot(2, 4, 8)
plt.title('Actual')
pca_2d = PCA(n_components=2).fit_transform(X)
principal_df = pd.DataFrame(data = pca_2d
             , columns = ['principal component 1', 'principal component 2'])
pca_df = pd.concat([principal_df, y], axis = 1)
sns.scatterplot(x="principal component 1", y="principal component 2", hue="label", data=pca_df, legend=False)


plt.tight_layout(pad=0.4, w_pad=0.3, h_pad=1.0)
plt.show()
