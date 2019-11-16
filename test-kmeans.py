import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
from KMeans import KMeans
from sklearn.cluster import KMeans as KMeans_SKLearn

iris = load_iris()
X = iris['data']
y = pd.DataFrame({'label': iris['target']})

N_CLUSTERS = len(iris['target_names'])

# Bikin sendiri
km = KMeans(n_clusters=N_CLUSTERS)
km.fit(X)
y_pred = pd.DataFrame({'label': km.predict(X)})

# Scikit Learn
km_scikit = KMeans_SKLearn(n_clusters=N_CLUSTERS)
y_pred_sklearn = pd.DataFrame({'label': km_scikit.fit_predict(X)})

plt.subplot(2, 2, 1)
plt.title('Predicted')
pca_pred = PCA(n_components=2).fit_transform(X)
principal_df_pred = pd.DataFrame(data = pca_pred
             , columns = ['principal component 1', 'principal component 2'])
pca_df_pred = pd.concat([principal_df_pred, y_pred], axis = 1)
sns.scatterplot(x="principal component 1", y="principal component 2", hue="label", data=pca_df_pred )

plt.subplot(2, 2, 2)
plt.title('Actual')
pca_2d = PCA(n_components=2).fit_transform(X)
principal_df = pd.DataFrame(data = pca_2d
             , columns = ['principal component 1', 'principal component 2'])
pca_df = pd.concat([principal_df, y], axis = 1)
sns.scatterplot(x="principal component 1", y="principal component 2", hue="label", data=pca_df)

plt.subplot(2, 2, 3)
plt.title('Scikit-Learn')
pca_2d = PCA(n_components=2).fit_transform(X)
principal_df = pd.DataFrame(data = pca_2d
             , columns = ['principal component 1', 'principal component 2'])
pca_df = pd.concat([principal_df, y_pred_sklearn], axis = 1)
sns.scatterplot(x="principal component 1", y="principal component 2", hue="label", data=pca_df)

plt.tight_layout()
plt.show()
