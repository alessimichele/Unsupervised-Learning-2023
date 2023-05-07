import numpy as np
import math
from matplotlib import pyplot as plt

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