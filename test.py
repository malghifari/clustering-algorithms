import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
from DBSCAN import DBSCAN
from sklearn.cluster import DBSCAN as DBSCAN_Sklearn

MINPTS = 5
EPSILON = 0.9

iris = load_iris()
X = iris['data']
y = pd.DataFrame({'label': iris['target']})

# Bikin sendiri
dbscan = DBSCAN(epsilon=EPSILON, minpts=MINPTS)
dbscan.fit(iris['data'])
y_pred = pd.DataFrame({'label': dbscan.predict()})

# Sklearn
dbscan_sklearn = DBSCAN_Sklearn(eps=EPSILON, min_samples=MINPTS)
dbscan_sklearn.fit(X)
y_pred_sklearn = pd.DataFrame({'label': dbscan_sklearn.labels_})

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
plt.title('Scikit-learn')
pca_2d = PCA(n_components=2).fit_transform(X)
principal_df = pd.DataFrame(data = pca_2d
             , columns = ['principal component 1', 'principal component 2'])
pca_df = pd.concat([principal_df, y_pred_sklearn], axis = 1)
sns.scatterplot(x="principal component 1", y="principal component 2", hue="label", data=pca_df)

plt.tight_layout()
plt.show()

sorted_y = y.copy()
sorted_y.sort()
print(np.unique(y))
print(sorted_y)
