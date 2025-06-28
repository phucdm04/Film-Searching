import numpy as np

class SVDModel:
    """
    SVD class for dimensionality reduction of Bag of Words matrices
    Built from scratch using eigenvalue decomposition
    """
    def __init__(self, n_components=100):
        self.n_components = n_components
        self.U = None
        self.s = None  # singular values
        self.Vt = None
        self.mean_vec = None
        self.is_fitted = False
        self.explained_variance_ratio_ = None

    def _svd_from_scratch(self, X):
        """
        SVD implementation using eigenvalue decomposition.
        X = U * S * Vt
        """
        m, n = X.shape

        # Compute covariance matrix
        XtX = X.T @ X
        eigenvals, V = np.linalg.eigh(XtX)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[sorted_idx]
        V = V[:, sorted_idx]

        # Remove non-positive eigenvalues
        positive_mask = eigenvals > 1e-10
        eigenvals = eigenvals[positive_mask]
        V = V[:, positive_mask]

        # Compute singular values and U
        singular_values = np.sqrt(eigenvals)
        r = len(singular_values)
        U = np.zeros((m, r))
        for i in range(r):
            U[:, i] = (X @ V[:, i]) / singular_values[i]

        # Normalize sign consistency
        for i in range(r):
            if U[0, i] < 0:
                U[:, i] *= -1
                V[:, i] *= -1

        return U, singular_values, V.T

    def fit(self, X):
        """Fit SVD to input matrix X"""
        X = np.asarray(X, dtype=np.float64)
        self.mean_vec = np.mean(X, axis=0)
        X_centered = X - self.mean_vec

        self.U, self.s, self.Vt = self._svd_from_scratch(X_centered)

        # Truncate to top n_components
        if self.n_components < len(self.s):
            self.U = self.U[:, :self.n_components]
            self.s = self.s[:self.n_components]
            self.Vt = self.Vt[:self.n_components, :]

        total_var = np.sum(self.s ** 2)
        self.explained_variance_ratio_ = (self.s ** 2) / total_var if total_var > 0 else np.zeros(len(self.s))
        self.is_fitted = True
        return self

    def transform(self, X):
        """Project input data X to reduced dimension space"""
        if not self.is_fitted:
            raise ValueError("SVD must be fitted before transform.")
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_vec
        return X_centered @ self.Vt.T

    def fit_transform(self, X):
        """Fit and transform input data in one call"""
        self.fit(X)
        return self.U * self.s

    def inverse_transform(self, X_transformed):
        """Reconstruct data from reduced dimension"""
        if not self.is_fitted:
            raise ValueError("SVD must be fitted before inverse_transform.")
        X_transformed = np.asarray(X_transformed)
        return X_transformed @ self.Vt + self.mean_vec