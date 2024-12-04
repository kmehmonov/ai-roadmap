import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

class KMeansCustom:
    def __init__(self, n_clusters: int):
        self.n_clusters = n_clusters
        self.is_fitted = False

    def fit(self, X, y=None):
        # Initial centroids
        self.__cluster_centers = None
        centroids = np.random.uniform(low=X.min(), high=X.max(), 
                                      size=(self.n_clusters, X.shape[1]))

        while not np.all(self.__cluster_centers == centroids):
            print(centroids)
            self.__cluster_centers = centroids
            clusters = self.calc_clusters(X, centroids)
            centroids = self.calc_centroids(X, clusters, self.n_clusters)

        self.is_fitted = True
        return self
    
    def predict(self, X):
        clusters = self.calc_clusters(X, self.cluster_centers_)
        return clusters

    def fit_predict(self, X, y=None):
        return self.fit(X, y).predict(X)

    @property
    def cluster_centers_(self):
        if not self.is_fitted:
            raise Exception("Model must have fit first")
        return self.__cluster_centers

    @staticmethod
    def euc_dist(point: NDArray, centroids: NDArray):
        """
            `point` shape => (n,)
            `centroids` is ndarray with shape (m, n)
            Returns ndarray with shape (m,), which represents
                distances from the point to center points.
        """
        distances = []
        for centroid in centroids:
            diff = point - centroid
            distances.append(np.sqrt(np.dot(diff.T, diff)))
        return np.array(distances)
    
  
    def calc_clusters(self, X: NDArray, centroids: NDArray):
        """
            `X` shape => (m1, n)
            `centroids` shape => (m2, n)
            Returns clusters shape => (m1,)
        """
        clusters = []
        for point in X:
            distances = self.euc_dist(point, centroids)
            clusters.append(distances.argmin())
        return np.array(clusters)
    

    def calc_centroids(self, X: NDArray, clusters: NDArray, n_clusters: int):
        """
            `X` shape => (m1, n)
            `clusters` shape => (m1,)
            `n_clusters` is number of clusters
            Returns centroids shape => (m2,n)
        """
        centroids = []
        for i in range(n_clusters):
            centroid = X[clusters==i]
            if centroid.size == 0:
                centroids.append(self.__cluster_centers[i])
            else:
                centroids.append(centroid.mean(axis=0))
        return np.array(centroids)


if __name__ == "__main__":
    from pathlib import Path

    points = pd.read_csv(Path(__file__).resolve().parent/'3d_points.csv').to_numpy()

    kmeans = KMeansCustom(n_clusters=3)
    clusters = kmeans.fit_predict(points)
    print(clusters)
