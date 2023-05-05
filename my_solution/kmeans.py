import numpy as np
import math
from matplotlib import pyplot as plt

from ipywidgets import interactive, IntSlider
import os
import imageio


class MyKmeans:
    """
    MyKmeans is a class that implements the k-means clustering algorithm.

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

    algorithm_variant : str, one of ['kmeans', 'kmeans++']
        The variant of the k-means algorithm to use.
    """

    def __init__(self, k, algorithm_variant='kmeans'):
        """
        Initializes a new instance of the MyKmeans class.

        Parameters:
        -----------
        k : int
            The number of clusters to form.
        
        algorithm_variant : str, one of ['kmeans', 'kmeans++']
            The variant of the k-means algorithm to use.
        """
        self.k = k
        self.z = None
        self.loss = None
        self.C = None
        self.n_iterations = 0
        self.C_history = None
        self.z_history = None
        self.algorithm_variant = algorithm_variant
        self.algorithm_variants=['kmeans', 'kmeans++']


        
    def _kmeans(self, X, random_choice):
        """
        Fits the standard k-means algorithm to the provided dataset X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to cluster.

        random_choice : integer (default=0)
            Seed value for random initialization of centroids.

        Returns:
        --------
        None
        """
        # first initialization
        np.random.seed(random_choice)
        c_idxs = np.random.choice(X.shape[0], size=self.k)
        self.C = np.array(X[c_idxs])
        self.z = np.zeros(X.shape[0])
        tmp = np.ones(X.shape[0])
        # Initialize C_history and z_history arrays
        self.C_history = np.empty((self.k, X.shape[1], 0))
        self.z_history = np.empty((X.shape[0], 0))

        while not np.array_equal(self.z, tmp):
            self.n_iterations += 1

            # Append current C and z values to C_history and z_history respectively
            self.C_history = np.concatenate(
                (self.C_history, self.C[..., np.newaxis]), axis=-1
            )
            self.z_history = np.concatenate(
                (self.z_history, self.z[..., np.newaxis]), axis=-1
            )

            tmp = np.copy(self.z)

            # assign each point to its neares neighbor
            for i in range(X.shape[0]):
                best_d = math.inf
                nearest_center = None
                for j in range(self.k):
                    d = np.linalg.norm(X[i] - self.C[j, :])
                    if d < best_d:
                        best_d = d
                        nearest_center = j
                self.z[i] = nearest_center

            # recompute the centers
            for i in range(self.k):
                idxs = np.where(self.z == i)[0]  # index of datapoint in cluster i
                self.C[i, :] = np.mean(X[idxs, :], axis=0)  # update centers

        self.loss = 0
        for i in range(self.k):
            idxs = np.where(self.z == i)[0]
            for j in range(len(idxs)):
                self.loss += np.linalg.norm(X[idxs[j]] - self.C[i, :]) ** 2

        # Append final C and z values to C_history and z_history respectively
        self.C_history = np.concatenate(
            (self.C_history, self.C[..., np.newaxis]), axis=-1
        )
        self.z_history = np.concatenate(
            (self.z_history, self.z[..., np.newaxis]), axis=-1
        )

        

    def _kmeansplusplus(self, X, random_choice):
        """
        Fits the standard k-means++ algorithm to the provided dataset X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to cluster.

        random_choice : integer (default=0)
            Seed value for random initialization of centroids.

        Returns:
        --------
        None
        """
        # initialization
        np.random.seed(random_choice)
        first_idx = np.random.choice(X.shape[0], size=1)
        c_idxs = np.array([first_idx[0]])
        D_x = np.zeros(X.shape[0])
        while len(c_idxs) < self.k:
            for i in range(X.shape[0]):
                if i not in c_idxs:
                    distances_to_centers = np.zeros(len(c_idxs))
                    for j in range(len(c_idxs)):
                        distances_to_centers[j] = np.linalg.norm(X[i] - X[c_idxs[j]])
                    minimum_distance_center_index = np.min(distances_to_centers)
                    D_x[i] = minimum_distance_center_index**2
            
            probs = D_x / np.sum(D_x)
            new_c = np.random.choice(X.shape[0], size=1, p=probs)
            while new_c in c_idxs:
                new_c = np.random.choice(X.shape[0], size=1, p=probs)
            c_idxs = np.append(c_idxs, new_c)

        # iteration
        self.C = np.array(X[c_idxs])
        self.z = np.zeros(X.shape[0])
        tmp = np.ones(X.shape[0])
        # Initialize C_history and z_history arrays
        self.C_history = np.empty((self.k, X.shape[1], 0))
        self.z_history = np.empty((X.shape[0], 0))

        while not np.array_equal(self.z, tmp):
            self.n_iterations += 1
            # Append current C and z values to C_history and z_history respectively
            self.C_history = np.concatenate(
                (self.C_history, self.C[..., np.newaxis]), axis=-1
            )
            self.z_history = np.concatenate(
                (self.z_history, self.z[..., np.newaxis]), axis=-1
            )

            tmp = np.copy(self.z)

            # assign each point to its neares neighbor
            for i in range(X.shape[0]):
                best_d = math.inf
                nearest_center = None
                for j in range(self.k):
                    d = np.linalg.norm(X[i] - self.C[j, :])
                    if d < best_d:
                        best_d = d
                        nearest_center = j
                self.z[i] = nearest_center

            # recompute the centers
            for i in range(self.k):
                idxs = np.where(self.z == i)[0]  # index of datapoint in cluster i
                self.C[i, :] = np.mean(X[idxs, :], axis=0)  # update centers

        self.loss = 0
        for i in range(self.k):
            idxs = np.where(self.z == i)[0]
            for j in range(len(idxs)):
                self.loss += np.linalg.norm(X[idxs[j]] - self.C[i, :]) ** 2

        # Append final C and z values to C_history and z_history respectively
        self.C_history = np.concatenate(
            (self.C_history, self.C[..., np.newaxis]), axis=-1
        )
        self.z_history = np.concatenate(
            (self.z_history, self.z[..., np.newaxis]), axis=-1
        )

    
    def fit(self, X, random_choice=0):
        """
        Fits the k-means algorithm to the provided dataset X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to cluster.

        random_choice : integer (default=0)
            Seed value for random initialization of centroids.

        Returns:
        --------
        None
        """
        if self.algorithm_variant not in self.algorithm_variants:
            raise ValueError('Invalid algorithm variant: %s' % self.algorithm_variant)
        
        if self.algorithm_variant == 'kmeans':
            self._kmeans(X, random_choice)
        elif self.algorithm_variant == 'kmeans++':
            self._kmeansplusplus(X, random_choice)
        else:
            raise ValueError('Invalid algorithm variant: %s' % self.algorithm_variant)
    


    def scree_test(self, X, random_choice=0, max_k=15):
        """
        Performs the scree test to determine the optimal number of clusters.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to cluster.

        random_choice : integer (default=0)
            Seed value for random initialization of centroids.

        max_k : int (default=None)
            The maximum number of clusters to test. If None, defaults to the size of X.

        Returns:
        --------
        The scree test plot for the setted number of clusters 
        """
        sse = np.zeros(max_k)

        for k in range(1, max_k+1):
            kmeans = MyKmeans(k, algorithm_variant=self.algorithm_variant)
            kmeans.fit(X, random_choice=random_choice)
            sse[k-1] = kmeans.loss

        plt.plot(np.arange(1, max_k+1), sse, 'o-')
        plt.xticks(np.arange(1, max_k+1))
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Loss')
        plt.title(f'Scree Test. Algorithm: {self.algorithm_variant}. Number of clusters: {max_k}')
        plt.show()



    ###############################################################
    ############# Plot Methods ####################################
    ###############################################################

    
    def plot(self, X):
        cmap = plt.get_cmap("jet", self.k)
        plt.scatter(X[:, 0], X[:, 1], c=self.z, cmap=cmap)
        plt.scatter(self.C[:, 0], self.C[:, 1], marker="x", color="black")
        plt.title(f"Algorithm: {self.algorithm_variant}. log-Loss: {np.log(self.loss)}. Iterations: {self.n_iterations}")
        plt.show()

    def plot_history(self, X):
        """
        Plots the change in cluster centers and cluster assignments over iterations.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            The input data to cluster.

        Returns:
        --------
        None
        """

        def plot_iteration(i):
            cmap = plt.get_cmap("jet", self.k)
            plt.scatter(X[:, 0], X[:, 1], c=self.z_history[:, i], cmap=cmap)
            plt.title(f'Algorithm: {self.algorithm_variant}')
            plt.scatter(self.C_history[:, 0, i], self.C_history[:, 1, i], marker="x")

        slider = IntSlider(min=0, max=self.n_iterations, step=1, value=0)
        return interactive(plot_iteration, i=slider)

    def create_gif(self, X, duration = 2):
        # Create directory to store the images
        if not os.path.exists("iterations"):
            os.makedirs("iterations")

        # Create list to store each plot as an image
        images = []

        # Create plot for each iteration and save as an image
        for i in range(self.n_iterations):
            plt.scatter(X[:, 0], X[:, 1], c=self.z_history[:, i + 1])
            plt.scatter(
                self.C_history[:, 0, i + 1], self.C_history[:, 1, i + 1], marker="x"
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
        imageio.mimsave("kmeans.gif", images, duration=duration)

    def set_to_zero(self):
        self.z = None
        self.loss = None
        self.C = None
        self.n_iterations = 0
        self.C_history = None
        self.z_history = None