import numpy as np
from utils import euclidean_distance

class Agglomerative:
    def __init__(self, n_clusters, linkage):
        self.n_clusters = n_clusters
        self.linkage = linkage
    
    def fit(self, X):
        # Init clusters dengan tiap data point
        self.X = np.asarray(X)
        self.clusters = np.reshape(X, (X.shape[0], 1, X.shape[1])).tolist()
        self.X = X.tolist()

        # looping hingga jumlah cluster == n_cluster
        while len(self.clusters) > self.n_clusters:
            # cari dua cluster dengan jarak paling kecil
            min_distance = 0
            a = 0
            b = 0
            for i in range(len(self.clusters)):
                for j in range(i+1, len(self.clusters)):
                    distance = self._get_distance(self.clusters[i], self.clusters[j])
                    if (i == 0 and j == 1) or (min_distance > distance):
                        min_distance = distance
                        a = i
                        b = j
            
            # buat cluster baru dengan data yang digabung
            combined_cluster = self._combine_cluster(self.clusters[a], self.clusters[b])
            
            # hapus 2 cluster sebelumnya
            self.clusters.pop(b)
            self.clusters.pop(a)

            # append cluster baru ke clusters
            self.clusters.append(combined_cluster)
        
    def predict(self):
        y = np.full(len(self.X), 0).tolist()
        for i in range(len(self.X)):
            y[i] = self._which_cluster(self.X[i])
        return y

    def _which_cluster(self, item):
        for i in range(len(self.clusters)):
            for j in range(len(self.clusters[i])):
                if item == self.clusters[i][j]:
                    return i

    def _get_distance(self, cluster_a, cluster_b):
        if self.linkage == 'single':
            for i in range(len(cluster_a)):
                for j in range(len(cluster_b)):
                    distance = euclidean_distance(cluster_a[i], cluster_b[j])
                    if (i == 0 and j == 0) or (min_distance > distance):
                        min_distance = distance
            return min_distance

        elif self.linkage == 'complete':
            for i in range(len(cluster_a)):
                for j in range(len(cluster_b)):
                    distance = euclidean_distance(cluster_a[i], cluster_b[j])
                    if (i == 0 and j == 0) or (max_distance < distance):
                        max_distance = distance
            return max_distance

        elif self.linkage == 'average':
            total_distance = 0
            count = 0
            for i in range(len(cluster_a)):
                for j in range(len(cluster_b)):
                    total_distance += euclidean_distance(cluster_a[i], cluster_b[j])
                    count += 1
            return total_distance / count

        elif self.linkage == 'average-group':
            cluster_a = np.asarray(cluster_a)
            avg_cluster_a = np.zeros((cluster_a.shape[1]))
            for item in cluster_a:
                avg_cluster_a = avg_cluster_a + item
            cluster_b = np.asarray(cluster_b)
            avg_cluster_b = np.zeros((cluster_b.shape[1]))
            for item in cluster_b:
                avg_cluster_b = avg_cluster_b + item

            return euclidean_distance(avg_cluster_a, avg_cluster_b)
            
    def _combine_cluster(self, cluster_a, cluster_b):
        return np.concatenate((np.asarray(cluster_a), np.asarray(cluster_b))).tolist()