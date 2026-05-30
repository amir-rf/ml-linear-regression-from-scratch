import numpy as np

class LinearRegressionNormalEquation:


    def __init__(self):
        self.theta = None


    def fit(self, X, y):
        n_sample = X.shape[0]

        bias_col = np.ones((n_sample, 1))

        X_b = np.c_[bias_col, X]

        self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        n_sample = X.shape[0]

        bias_col = np.ones((n_sample, 1))
        X_b = np.c_[bias_col, X]

        return X_b @ self.theta
    
    @property
    def bias(self):
        return self.theta[0]
    
    @property
    def weights(self):
        return self.theta[1:]
    
    