import numpy as np
import math
from matplotlib import pyplot as plt

from ipywidgets import interactive, IntSlider
import os
import imageio
import time
from src.kmeans import _kmeans
from src.kmeanspp import _kmeansplusplus
from src.kmedoids import _kmedoids
from src.kmedoidspp import _kmedoidsplusplus
from src.cmeans_naive import _cmeans


class Clustering:
    """
    Clustering is a class that implements flat clustering algorithm from scratch.

    Attributes:
    -----------
    k : int
        The number of clusters to form.

    z : numpy.ndarray, shape (n_samples,)
        The final assignment of each point to a cluster.

    loss : float
        The total loss after clustering, defined as the sum of the squared distances
        between each point and its assigned cluster center.

    C : numpy.ndarray, shape (k, n_features)
        The final cluster centers.

    n_iterations : int
        The number of iterations required to converge.

    C_history : numpy.ndarray, shape (k, n_features, n_iterations+1)
        A record of the cluster centers at each iteration.

    z_history : numpy.ndarray, shape (n_samples, n_iterations+1)
        A record of the assignment of each point to a cluster at each iteration.

    algorithm_variant : str, one of ['kmeans', 'kmeans++', 'kmedoids', 'kmedoids++', 'cmeans']
        The variant of the algorithm to use.

    TIME : float
        The time required to fit (s).

    U : numpy.ndarray, shape (n_samples, k)
        Matrix of 'probabilities'

    Example:
        ```
        import numpy as np
        from sklearn.datasets import make_blobs
        from ClusteringClass import Clustering

        # generate random data
        X, _ = make_blobs(n_samples=1000, centers=5, random_state=42)

        # create an instance of Clustering class
        clustering = Clustering(k=5, algorithm_variant="kmeans")

        # fit the data
        clustering.fit(X)

        # print running time to fit
        clustering.time()

        # plot the final partition of the data
        clustering.plot(X)

        # plot the history of cluster centers
        clustering.plot_history(X)

        # generate a GIF
        clustering.create_gif(X)

        # plot the scree test
        clustering.scree_test(X, k_max=15)
        ```
    """

    def __init__(self, k, algorithm_variant="kmeans"):
        """
        Initializes a new instance of the Clustering class.

        Parameters:
        -----------
        k : int
            The number of clusters to form.

        algorithm_variant : str, one of ['kmeans', 'kmeans++', 'kmedoids', 'kmedoids++', 'cmeans']
            The variant of the algorithm to use.
        """
        self.k = k
        self.z = None
        self.loss = None
        self.C = None
        self.n_iterations = 0
        self.C_history = None
        self.z_history = None
        self.algorithm_variant = algorithm_variant
        self.algorithm_variants = [
            "kmeans",
            "kmeans++",
            "kmedoids",
            "kmedoids++",
            "cmeans",
        ]
        self.TIME = None
        self.U = None

    def _kmeans(self, X, random_choice):
        return _kmeans(self, X, random_choice)

    def _kmeansplusplus(self, X, random_choice):
        return _kmeansplusplus(self, X, random_choice)

    def _kmedoids(self, X, random_choice):
        return _kmedoids(self, X, random_choice)

    def _kmedoidsplusplus(self, X, random_choice):
        return _kmedoidsplusplus(self, X, random_choice)

    def _cmeans(self, X, random_choice, err=0.1, f=2):
        return _cmeans(self, X, random_choice, err=0.1, f=2)

    def fit(self, X, random_choice=0):
        """
        Fits the algorithm to the provided dataset X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to cluster.

        random_choice : integer (default=0)
            Seed value for random initialization of centers/centroids.

        Returns:
        --------
        None

        Example:
        ```
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)
        ```
        """
        if self.algorithm_variant not in self.algorithm_variants:
            raise ValueError("Invalid algorithm variant: %s" % self.algorithm_variant)

        start = time.time()

        if self.algorithm_variant == "kmeans":
            self._kmeans(X, random_choice)
        elif self.algorithm_variant == "kmeans++":
            self._kmeansplusplus(X, random_choice=random_choice)
        elif self.algorithm_variant == "kmedoids":
            self._kmedoids(X, random_choice=random_choice)
        elif self.algorithm_variant == "kmedoids++":
            self._kmedoidsplusplus(X, random_choice=random_choice)
        elif self.algorithm_variant == "cmeans":
            self._cmeans(X, random_choice=random_choice, err=0.1, f=2)
        else:
            raise ValueError("Invalid algorithm variant: %s" % self.algorithm_variant)

        end = time.time()

        self.TIME = end - start

    def scree_test(self, X, max_k, random_choice=0):
        """
        Performs the scree test to determine the optimal number of clusters.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to cluster.

        random_choice : integer (default=0)
            Seed value for random initialization of centroids.

        max_k : int
            The maximum number of clusters to test.

        Returns:
        --------
        The scree test plot for the setted number of clusters

        Example:
        ```
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)
        kmeans.scree_test(X)
        ```
        """
        sse = np.zeros(max_k)

        for k in range(1, max_k + 1):
            kmeans = Clustering(k, algorithm_variant=self.algorithm_variant)
            kmeans.fit(X, random_choice=random_choice)
            sse[k - 1] = kmeans.loss

        plt.plot(np.arange(1, max_k + 1), sse, "o-")
        plt.xticks(np.arange(1, max_k + 1))
        plt.xlabel("Number of clusters (k)")
        plt.ylabel("Loss")
        plt.title(
            f"Scree Test. Algorithm: {self.algorithm_variant}. Number of clusters: {max_k}"
        )
        plt.show()

    ###############################################################
    ############# Plot Methods ####################################
    ###############################################################

    def plot(self, X, cool=False, save=False):
        """
        Plots the input data with cluster centers using scatter plots.

        Parameters:
        -----------
        X : array-like, shape (n_samples, 2)
            The input data to cluster.

        cool : bool, default=False
            If True, plots on a dark background with white markers.

        Raises:
        ------
        ValueError: If the input dataset X has more than 2 features.

        Returns:
        --------
        None

        Example:
        ```
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)
        kmeans.plot(X)
        ```
        """
        if X.shape[1] != 2:
            raise ValueError(".create_gif method not supported if n_features != 2.")

        color = "black"

        if cool == True:
            plt.style.use("dark_background")
            color = "white"

        cmap = plt.get_cmap("jet", self.k)
        plt.scatter(X[:, 0], X[:, 1], c=self.z, cmap=cmap)
        plt.scatter(self.C[:, 0], self.C[:, 1], marker="x", color=color)
        plt.title(
            f"Algorithm: {self.algorithm_variant}. log-Loss: {np.log(self.loss)}. Iterations: {self.n_iterations}"
        )
        if save is True:
            plt.savefig(os.path.join("images", f"{self.algorithm_variant}.png"))

        plt.show()
        plt.style.use("default")

    def plot_history(self, X, cool=False):
        """
        Plots the change in cluster centers and cluster assignments over iterations.

        Parameters:
        -----------
        X : array-like, shape (n_samples, 2)
            The input data to cluster.

        cool : bool, default=False
            If True, plots on a dark background with white markers.

        Raises:
        ------
        ValueError: If the input dataset X has more than 2 features.

        Returns:
        --------
        Interactive plot.

        Example:
        ```
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)
        kmeans.plot_history(X)
        ```
        """
        if X.shape[1] != 2:
            raise ValueError(".create_gif method not supported if n_features != 2.")

        def plot_iteration(i):
            color = "black"
            if cool == True:
                plt.style.use("dark_background")
                color = "white"

            cmap = plt.get_cmap("jet", self.k)
            plt.scatter(X[:, 0], X[:, 1], c=self.z_history[:, i], cmap=cmap)
            plt.title(f"Algorithm: {self.algorithm_variant}")
            plt.scatter(
                self.C_history[:, 0, i],
                self.C_history[:, 1, i],
                marker="x",
                color=color,
            )

        plt.style.use("default")

        slider = IntSlider(min=0, max=self.n_iterations, step=1, value=0)
        return interactive(plot_iteration, i=slider)

    def create_gif(self, X, duration=2, cool=False):
        """
        Create an animated GIF of K-means clustering algorithm iterations on a 2D dataset.

        Parameters:
        -----------
        X : array-like, shape (n_samples, 2)
            The input data to cluster.
        duration : int, default=2
            Duration of each frame in seconds (default=2).
        cool : bool, default=False
            If True, plots on a dark background with white markers.

        Raises:
        ValueError: If the input dataset X has more than 2 features.

        Returns:
        --------
        None

        Saves an animated GIF file named "kmeans.gif" in the current directory.

        Example:
        ```
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)
        kmeans.create_gif(X)
        ```
        """
        if X.shape[1] != 2:
            raise ValueError(".create_gif method not supported if n_features != 2.")

        color = "black"
        if cool == True:
            plt.style.use("dark_background")
            color = "white"

        # Create directory to store the images
        if not os.path.exists("iterations"):
            os.makedirs("iterations")

        # Create list to store each plot as an image
        images = []

        # Create plot for each iteration and save as an image
        for i in range(self.n_iterations):
            plt.scatter(X[:, 0], X[:, 1], c=self.z_history[:, i + 1])
            plt.scatter(
                self.C_history[:, 0, i + 1],
                self.C_history[:, 1, i + 1],
                marker="x",
                color=color,
            )
            plt.title(f"Algorithm: {self.algorithm_variant}. Iteration {i+1}")
            plt.xlabel("Feature 1")
            plt.ylabel("Feature 2")
            plt.tight_layout()
            plt.savefig(f"iterations/iteration_{i+1}.png")
            plt.close()

            # Append image to list
            images.append(imageio.imread(f"iterations/iteration_{i+1}.png"))

        # Use imageio to create GIF from list of images
        imageio.mimsave(
            os.path.join("images", f"{self.algorithm_variant}.gif"),
            images,
            duration=duration,
        )
        plt.style.use("default")

    def set_to_zero(self):
        """
        Reset all instance variables to their initial values.
        """
        self.z = None
        self.loss = None
        self.C = None
        self.n_iterations = 0
        self.C_history = None
        self.z_history = None

    def time(self):
        """
        Print the running time of the algorithm.
        """
        return print(f"Running time: {self.TIME:.4f} seconds")
