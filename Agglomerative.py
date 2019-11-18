import numpy as np
from utils import euclidean_distance

class Agglomerative:
    def __init__(self, n_clusters, linkage):
        self.n_clusters = n_clusters
        self.linkage = linkage
    
    def fit(self, X):
        # Init clusters dengan tiap data point
        self.clusters = np.reshape(X, (X.shape[0], 1, X.shape[1]))

        # looping hingga jumlah cluster == n_cluster
        while self.clusters.shape[0] > self.n_clusters:
            # cari dua cluster dengan jarak paling kecil
            
            for i in range(self.clusters.shape[0]):
                for j in range(i+1, self.clusters.shape[0]):
                    distance = self._get_distance(self.clusters[i], self.clusters[j])
                    if (i == 0 and j == 0) or (min_distance > distance):
                        min_distance = distance
                        a = i
                        b = j
            
            # buat cluster baru dengan data yang digabung
            self._combine_cluster(self.clusters[a], self.clusters[b])
            
            # hapus 2 cluster sebelumnya
            self.clusters = np.delete(self.clusters, a, 0)
            self.clusters = np.delete(self.clusters, b, 0)

    def _get_distance(self, cluster_a, cluster_b):
        if self.linkage == 'single':
            for i in range(cluster_a.shape[0]):
                for j in range(cluster_b.shape[0]):
                    distance = euclidean_distance(cluster_a[i], cluster_b[j])
                    if (i == 0 and j == 0) or (min_distance > distance):
                        min_distance = distance
            return min_distance

        elif self.linkage == 'complete':
            for i in range(cluster_a.shape[0]):
                for j in range(cluster_b.shape[0]):
                    distance = euclidean_distance(cluster_a[i], cluster_b[j])
                    if (i == 0 and j == 0) or (max_distance < distance):
                        max_distance = distance
            return max_distance

        elif self.linkage == 'average':
            return 0

        elif self.linkage == 'average-group':
            return 0
            
    def _combine_cluster(self, cluster_a, cluster_b):
        return np.concatenate((cluster_a, cluster_b))