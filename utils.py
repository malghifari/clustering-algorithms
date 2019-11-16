import math

def euclidean_distance(x, y):
    """Compute euclidean distance.
        Parameters
        ----------
        x : Array of numbers
        y : Array of numbers
    """
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
