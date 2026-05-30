import numpy as np

class RidgeRegressionScratch:

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.theta = None

    
    def fit(self, X, y):

        n_samples, n_features = X.shape

        bias_col = np.ones((n_samples, 1))

        X_b = np.c_[bias_col, X]
        penalty_matrix = np.eye(n_features+1)
        penalty_matrix[0, 0] = 0

        self.theta = (
            np.linalg.pinv(X_b.T @ X_b + self.alpha * penalty_matrix)
            @ X_b.T
            @ y
        )


    def predict(self, X):
        n_samples = X.shape[0]
        bias_col = np.ones((n_samples, 1))

        X_b = np.c_[bias_col, X]

        return X_b @ self.theta
    
    @property
    def bias(self):
        return self.theta[0]
    
    @property
    def weights(self):
        return self.theta[1:]