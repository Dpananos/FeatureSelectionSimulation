import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def linear_spline(x):
    
    xx = x.copy()
    xx[xx<0] = 0
    
    return xx
    
    
class RestrictedCubicSpline(BaseEstimator, TransformerMixin):
    
    def __init__(self, k=3):

        self.k = k
    
    def fit(self, X, y=None):
        
        if self.k<0:
            return self
               
        # From Regression Modelling strategies
        knot_locations = {
            3: [0.1, 0.5, 0.9],
            4: [0.05, 0.365, 0.65, 0.95],
            5: [0.05, 0.275, 0.5, 0.725, 0.95],
            6: [0.05, 0.23, 0.41, 0.59, 0.77, 0.95],
            7: [0.025, 0.1833, 0.3417, 0.5, 0.6583, 0.8167, 0.975]
        }
        
        if self.k>7 or self.k<3:
            # Need a smart way to do more knots than 7.  For now, just throw an error
            raise ValueError('Value of k is not supported.  Set k between 3 and 7.')
            
        self._quantiles = knot_locations[self.k]
        self.t_ = np.quantile(X, q = self._quantiles)
        
        return self
    
    def transform(self, X, y=None):
        
        n_observations, n_features = self._validate_data(X, y).shape
        
        if self.k<0:
            return X
                
        self.basis_expansion = np.zeros((n_observations, self.k-1))
        
        self.basis_expansion[:,0, np.newaxis] = X
        
        for j in range(self.k-2):
            basis_function = 0
            basis_function += linear_spline(X - self.t_[j])**3
            basis_function -= linear_spline(X - self.t_[self.k-2])**3*(self.t_[-1] - self.t_[j])/(self.t_[-1] - self.t_[-2])
            basis_function += linear_spline(X - self.t_[-1])**3 *(self.t_[-2] - self.t_[j])/(self.t_[-1] - self.t_[-2])
            self.basis_expansion[:,j+1, np.newaxis] = basis_function/(self.t_[-1] - self.t_[0])**2
            
        return self.basis_expansion