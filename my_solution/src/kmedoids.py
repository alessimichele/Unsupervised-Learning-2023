import numpy as np
import math
from matplotlib import pyplot as plt


def _kmedoids(self, X, random_choice):
    """
    Fits the standard k-medoids algorithm to the provided dataset X.

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

    # self.indexs = np.copy(X[c_idxs])  # indexes of medoids

    # iteration

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

        # assign each point to its nearest medoid
        for i in range(X.shape[0]):
            # if i not in self.indexs:
            best_d = math.inf
            nearest_center = None
            for j in range(self.k):
                d = np.linalg.norm(X[i] - self.C[j, :])
                if d < best_d:
                    best_d = d
                    nearest_center = j
            self.z[i] = nearest_center

        # print(self.z)

        # recompute the medoids
        for m in range(self.k):  # for each cluster
            idxs = np.where(self.z == m)[0]  # index of datapoint in cluster m

            if (
                len(idxs) < 1
            ):  # meaning "no point assigned to that cluster except the medoid itself"
                raise RuntimeError(
                    f"No assignation possible for cluster {m}. Try another seed."
                )
            pairwise_dist = np.zeros(
                len(idxs)
            )  # variable to store sum of pairwise distances: in k-th position there is the sum between k-th point and any other point in the cluster
            for n in range(len(idxs)):  # for each point in cluster i
                # print("point", n)
                distances = np.zeros(
                    len(idxs)
                )  # variable to store distance between point j and other points in cluster i
                for l in range(
                    len(idxs)
                ):  # compute the distance between this point and any other point in the group
                    distances[l] = np.linalg.norm(X[idxs[n]] - X[idxs[l]])

                pairwise_dist[n] = np.sum(distances)
                # print(pairwise_dist)

            new_medoid = np.argmin(pairwise_dist)

            self.C[m, :] = X[idxs[new_medoid]]  # update centers
            # self.indexs[m] = idxs[new_medoid]

    self.loss = 0
    for i in range(self.k):
        idxs = np.where(self.z == i)[0]
        for j in range(len(idxs)):
            self.loss += np.linalg.norm(X[idxs[j]] - self.C[i, :]) ** 2

    # Append final C and z values to C_history and z_history respectively
    self.C_history = np.concatenate((self.C_history, self.C[..., np.newaxis]), axis=-1)
    self.z_history = np.concatenate((self.z_history, self.z[..., np.newaxis]), axis=-1)
