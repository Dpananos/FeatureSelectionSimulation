import numpy as np
import pandas as pd

from scipy.stats import norm, t

from sklearn.datasets import load_boston, load_diabetes
from sklearn.compose import ColumnTransformer
from rcs import RestrictedCubicSpline

def boston_data():
    data = load_boston()
    X = data['data']
    # Dummy variables cause problems with Boston.  Not going to go through the trouble
    # Of leaving them in the model.  All models will not see this variable
    X=X[:, np.array([0,1,2,4,5,6,7,8,9,10,11,12])]
    y = data['target']
    return X, y

def diabetes_data():
    data = load_diabetes()
    X = data['data']
    y = data['target']
    return X, y

def non_linear_data():
    n_observations, n_features = 2500, 50

    # Simulate some covariates
    X = np.random.normal(size = (n_observations, n_features))
    df = pd.DataFrame(X, columns = [f'x_{j}' for j in range(n_features)])

    # Now want to investigate what happens when we misspecify model because that is always the case.
    # Expand all features into splines with 7 degrees of freedom.
    # Only the first 10 are non-linear.
    list_of_spline_transforms = [(f'spline_{feature}', RestrictedCubicSpline(k=7), [feature]) for feature in df.columns.tolist()[:10]]
    ct = ColumnTransformer(list_of_spline_transforms, remainder = 'passthrough')

    #Expand the basis functions and simulate an outcome
    Xspline = ct.fit_transform(df)

    betas = norm().rvs(size = Xspline.shape[1], random_state = 0)
    # Some features are useless, remainder are linear
    betas[-20:] = 0
    y = Xspline@betas + t(df=10).rvs(size=n_observations, random_state = 0)

    return X, y