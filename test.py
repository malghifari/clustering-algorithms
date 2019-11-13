import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import pandas as pd
from DBSCAN import DBSCAN

iris = load_iris()
dbscan = DBSCAN(epsilon=0.5, minpts=5)
dbscan.fit(iris['data'])
y = dbscan.predict()
y_pred = pd.DataFrame({'label': y})

pca = PCA(n_components=2).fit(iris['data'])
pca_2d = pca.transform(iris['data'])
principal_df = pd.DataFrame(data = pca_2d
             , columns = ['principal component 1', 'principal component 2'])
pca_df = pd.concat([principal_df, y_pred], axis = 1)
ax = sns.scatterplot(x="principal component 1", y="principal component 2", hue="label", size="label", data=pca_df)
plt.show()

# sorted_y = y
# sorted_y.sort()
# print(np.unique(y))
# print(sorted_y)
