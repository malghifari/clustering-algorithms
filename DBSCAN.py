from utils import euclidean_distance
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, minpts, epsilon):
        self.minpts = minpts
        self.epsilon = epsilon
    
    def range_query(self, point):
        direct_reachable_points = np.empty((0, 1), dtype='object')
        for neighbor in self.points:
            if euclidean_distance(neighbor.features, point.features) <= self.epsilon:
                direct_reachable_points = np.append(direct_reachable_points, neighbor)
        return direct_reachable_points

    def fit(self, X):
        self.points = np.empty((0, 1), dtype='object')
        for x in X:
            point = Point(x, None)
            self.points = np.append(self.points, point)

    def predict(self):
        label = -1
        for point in self.points:
            if point.label is not None:
                continue
            direct_reachable_points = self.range_query(point)
            if direct_reachable_points.shape[0] < self.minpts:
                point.label = -1
                continue
            label += 1
            point.label = label
            reachable_points = direct_reachable_points[direct_reachable_points != point]
            idx = 0
            while idx < len(reachable_points):
                reachable_point = reachable_points[idx]
                if reachable_point.label == -1:
                    reachable_point.label = label
                if reachable_point.label is not None:
                    idx += 1
                    continue
                reachable_point.label = label
                next_direct_reachable_points = self.range_query(reachable_point)
                if next_direct_reachable_points.shape[0] >= self.minpts:
                    for p in next_direct_reachable_points:
                        if p not in reachable_points:
                            reachable_points = np.append(reachable_points, p)

                idx += 1

        labels = [point.label for point in self.points]
        return labels

class Point:
    def __init__(self, features, label):
        self.features = features
        self.label = label