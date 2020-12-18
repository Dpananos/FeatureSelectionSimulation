import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RepeatedKFold, cross_val_score


from rcs import RestrictedCubicSpline
from make_data import *

class LassoSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        
        return None
        
    def fit(self, X, y):
        self.mod_ = LassoCV(cv=10).fit(X,y)
        return self
    
    def transform(self, X, y=None):
        self.coef_ = self.mod_.coef_
        return X[:, np.abs(self.coef_)>0]

class CorrelationSelector(BaseEstimator, TransformerMixin):

    def __init__(self, cutoff = 0.2):
        self.cutoff = cutoff
        return None
        
    def fit(self, X, y):

        self.correlations_ = np.array([pearsonr(j, y)[0] for j in X.T])
        return self
    
    def transform(self, X, y=None):
        selection = np.argwhere(np.abs(self.correlations_)>self.cutoff).ravel()
        return X[:, selection]



def experiment(data_loader, experiment_name):

    X,y = data_loader()
    
    # Set up pipelines.  Standard scale everything just in case.
    lasso_selection = make_pipeline(StandardScaler(), LassoSelector(), LinearRegression())
    correlation_selection = make_pipeline(StandardScaler(), CorrelationSelector(), LinearRegression())
    linear_regression = make_pipeline(StandardScaler(), LinearRegression())
    lasso = make_pipeline(StandardScaler(), LassoCV(cv=10))

    # Spline model is finicky.  I have to use column transformer to apply a spline to all columns independently.
    # Turn the data into a datframe, then apply column transformer.
    # I scale everything before applying the spline, not sure if this is necccesary though.
    n_observations, n_features = X.shape
    df = pd.DataFrame(X, columns = [f'x_{j}' for j in range(n_features)])

    # Purposefully mispecify the knots in the non-linear problem
    list_of_spline_transforms = [(f'spline_{feature}', make_pipeline(StandardScaler(),RestrictedCubicSpline(k=3)), [feature]) for feature in df.columns.tolist()]
    ct = ColumnTransformer(list_of_spline_transforms)

    splines = make_pipeline(ct, LinearRegression())

    # loop over models
    models = [linear_regression, lasso_selection, correlation_selection, lasso, splines]

    cv_results = np.array([cross_val_score(model, df, y, cv=RepeatedKFold(n_splits=10, n_repeats=100), scoring = 'neg_mean_squared_error') for model in models]).T

    results_df = pd.DataFrame(cv_results, columns = ['linear_regression','lasso_selection','correlation_selection','lasso','splines'])

    results_df.to_csv(f'{experiment_name}.csv', index = False)


if __name__=='__main__':

    # Execute the experiments
    experiment(diabetes_data, experiment_name='diabetes')
    experiment(boston_data, experiment_name='boston')
    experiment(non_linear_data, experiment_name='non-linear')