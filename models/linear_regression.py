import numpy as np

class myLinearRegression:

    def __init__(self, learning_rate=0.01, epochs=1000, method="gd"):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.method = method
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        if self.method == "gd":
            self._gradient_descent(X, y)
        elif self.method == "normal":
            self._normal_equation(X, y)

    def _gradient_descent(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):

            y_pred = X @ self.weights + self.bias
            error = y - y_pred

            dw = (-2 / n_samples) * (X.T @ error)
            db = (-2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def _normal_equation(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y

        self.bias = theta[0]
        self.weights = theta[1:]

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)