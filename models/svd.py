import numpy as np
from scipy.linalg import svd

class SVDModel:
    """
    SVD class for dimensionality reduction of Bag of Words matrices
    """
    def __init__(self, n_components=100, random_state=42):
        """
        Initialize SVD for BOW    Random state for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.U = None
        self.s = None  # singular values
        self.Vt = None
        self.mean_vec = None
        self.is_fitted = False
        self.explained_variance_ratio_ = None
        
    def fit(self, X):
        """Fit SVD to BOW matrix"""
        X = np.asarray(X, dtype=np.float64)
        
        # Center the data (optional for BOW, but often helpful)
        self.mean_vec = np.mean(X, axis=0)
        X_centered = X - self.mean_vec
        
        # Perform SVD
        # X = U * S * Vt
        self.U, self.s, self.Vt = svd(X_centered, full_matrices=False)
        
        # Keep only n_components
        if self.n_components < len(self.s):
            self.U = self.U[:, :self.n_components]
            self.s = self.s[:self.n_components]
            self.Vt = self.Vt[:self.n_components, :]
        
        # Calculate explained variance ratio
        total_var = np.sum(self.s ** 2)
        self.explained_variance_ratio_ = (self.s ** 2) / total_var
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        """Transform BOW matrix to reduced dimensions"""
        if not self.is_fitted:
            raise ValueError("SVD must be fitted before transform")
            
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_vec
        
        # Transform: X_new = X_centered * Vt.T * S^(-1) * S = X_centered * Vt.T
        # But we want: X_new = U * S (the left singular vectors scaled by singular values)
        # For new data: X_new = X_centered @ Vt.T @ diag(1/s) @ diag(s) = X_centered @ Vt.T
        X_transformed = X_centered @ self.Vt.T
        
        return X_transformed
    
    def fit_transform(self, X):
        """Fit SVD and transform data in one step"""
        self.fit(X)
        # For fitted data, we can directly use U * S
        return self.U * self.s
    
    def inverse_transform(self, X_transformed):
        """Transform data back to original space (approximate reconstruction)"""
        if not self.is_fitted:
            raise ValueError("SVD must be fitted before inverse_transform")
            
        X_transformed = np.asarray(X_transformed)
        
        # Reconstruct: X = U * S * Vt
        # X_transformed = U * S, so we need to multiply by Vt
        X_reconstructed = X_transformed @ self.Vt + self.mean_vec
        
        return X_reconstructed