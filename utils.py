import math

def euclidean_distance(vec_a, vec_b):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(vec_a, vec_b)]))
