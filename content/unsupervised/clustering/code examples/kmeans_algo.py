import numpy as np


class KMeans:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters

    def fit(self, X, y = None):
        centroids = ...



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    X = np.array([[0], [1], [2], [3], [10],[12], [13], [51], [52], [48], [55], [57]])
    
    n_clusters = 3
    centroids = np.random.randint(low=X.min(), high=X.max(), size=(n_clusters, X.shape[1]))
    # print(centroids)
    plt.scatter(X, np.zeros(X.shape[0]))
    plt.scatter(centroids, np.zeros(n_clusters), c='blue', marker='*')
    plt.show()