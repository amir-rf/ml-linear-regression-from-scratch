import numpy as np

class LinearRegressionScartch:
    
    def __init__(self, learning_rate=0.01, n_iteration=1000,):
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration

        self.bias = None
        self.weights = None

        self.loss_history = []

    def fit(self, X, y):

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for iteration in range(self.n_iteration):
            y_predict = np.dot(X, self.weights) + self.bias

            error = y_predict - y

            dw = (1/n_samples) * np.dot(X.T, error)
            db = (1/n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = np.mean(error**2)

            self.loss_history.append(loss)

            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Loss: {loss:.4f}")



def predict(self, X):

    return np.dot(X, self.weights) + self.bias


