import numpy as np
import math
from matplotlib import pyplot as plt


def _cmeans(self, X, random_choice, err=0.1, f=2):
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
    self.z = np.empty(X.shape[0])

    # initialize U
    matrix = np.random.rand(X.shape[0], self.k)
    row_sums = np.sum(matrix, axis=1)
    self.U = matrix / row_sums[:, np.newaxis]

    tmp = np.zeros([X.shape[0], self.k])

    # Initialize C_history and z_history arrays
    self.C_history = np.empty((self.k, X.shape[1], 0))
    self.z_history = np.empty((X.shape[0], 0))

    while np.linalg.norm(self.U - tmp) > err:
        self.n_iterations += 1

        # Append current C and z values to C_history and z_history respectively
        self.C_history = np.concatenate(
            (self.C_history, self.C[..., np.newaxis]), axis=-1
        )
        self.z_history = np.concatenate(
            (self.z_history, self.z[..., np.newaxis]), axis=-1
        )

        tmp = np.copy(self.U)

        # recompute the centers
        for i in range(self.k):
            sum = np.sum(
                (self.U[:, i] ** f)[:, np.newaxis] * X, axis=0
            )  # using python broadcasting
            # print(sum)
            self.C[i, :] = sum / np.sum(self.U[:, i] ** f)

        # assign each point to its neares neighbor
        for i in range(X.shape[0]):
            for j in range(self.k):
                sum = 0
                for n in range(self.k):
                    sum += (
                        np.linalg.norm(X[i] - self.C[j])
                        / np.linalg.norm(X[i] - self.C[n])
                    ) ** (2 / (f - 1))

                self.U[i, j] = 1 / sum

        for i in range(X.shape[0]):
            self.z[i] = np.argmax(self.U[i, :])

    for i in range(X.shape[0]):
        self.z[i] = np.argmax(self.U[i, :])

    self.loss = 0
    for i in range(self.k):
        idxs = np.where(self.z == i)[0]
        for j in range(len(idxs)):
            self.loss += np.linalg.norm(X[idxs[j]] - self.C[i, :]) ** 2

    # Append final C and z values to C_history and z_history respectively
    self.C_history = np.concatenate((self.C_history, self.C[..., np.newaxis]), axis=-1)
    self.z_history = np.concatenate((self.z_history, self.z[..., np.newaxis]), axis=-1)
