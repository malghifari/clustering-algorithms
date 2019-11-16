import numpy as np
import math
from utils import euclidean_distance

class KMeans:
    def __init__(self, n_clusters):
        self.centroids = []
        self.n_clusters = n_clusters

    def fit(self, X):
        for i in range(self.n_clusters):
            self.centroids.append(X[np.random.choice(X.shape[0])])
        
        while True:
            Y = np.empty(shape=(X.shape[0]), dtype=int)
            for i in range(X.shape[0]):
                # hitung centroid mana yang paling dekat
                # assign y ke i dengan nilai index centroid paling dekat
                nearest_centroid = 0
                min_distance = 0
                for j in range(len(self.centroids)):
                    distance = euclidean_distance(X[i], self.centroids[j])
                    if j == 0 or min_distance > distance:
                        nearest_centroid = j
                        min_distance = distance
                Y[i] = nearest_centroid
        
            old_centroids = np.copy(self.centroids)
            
            for i in range(self.n_clusters):
                new_centroid = np.full(X.shape[1], 0)
                count = 0
                for j in range(X.shape[0]):
                    if Y[j] == i:
                        new_centroid = new_centroid + X[j]
                        count += 1
                new_centroid /= count
                self.centroids[i] = new_centroid
            
            if np.array_equal(self.centroids, old_centroids):
                break

    def predict(self, X):
        Y = np.empty(shape=(X.shape[0]), dtype=int)
        for i in range(X.shape[0]):
            # hitung centroid mana yang paling dekat
            # assign y ke i dengan nilai index centroid paling dekat
            nearest_centroid = 0
            min_distance = 0
            for j in range(len(self.centroids)):
                distance = euclidean_distance(X[i], self.centroids[j])
                if j == 0 or min_distance > distance:
                    nearest_centroid = j
                    min_distance = distance
            Y[i] = nearest_centroid
        return Y.tolist()