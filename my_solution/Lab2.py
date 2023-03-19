import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_excel("../Datasets/Dry_Bean_Dataset.xlsx")


X = df. loc[:, df. columns != 'Class']
y = df.loc[:, df. columns == 'Class']

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8)

# 1. Standardize the dataset
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# 2. Calculate the covariance matrix
cov_mat = np.cov(X_std.T)

# 3. Calculate the eigenvectors and eigenvalues of the covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# 4. Sort the eigenvectors and eigenvalues
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# 5. Select the top k eigenvectors based on the explained variance
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
n_components = 2 # choose the number of components
top_k = eig_pairs[:n_components]
W = np.hstack((top_k[0][1].reshape(13,1), top_k[1][1].reshape(13,1)))

# 6. Transform the dataset into the new subspace
X_pca = X_std.dot(W)

