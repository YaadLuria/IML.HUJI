from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree
    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate
    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    min_range, max_range = -1.2, 2
    x = np.linspace(min_range, max_range, n_samples)
    x = x
    f = (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    y = f + np.random.normal(loc=0, scale=noise, size=n_samples)

    x_train, y_train, x_test, y_test = split_train_test(pd.DataFrame(x),
                                                        pd.Series(y), 2/3)
    x_train = x_train.to_numpy()[:, 0]
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()[:, 0]
    y_test = y_test.to_numpy()

    go.Figure([go.Scatter(x=x, y=f, mode='lines', name="Model"),
                     go.Scatter(x=x_train, y=y_train, mode='markers',
                                name="Train"),
                     go.Scatter(x=x_test, y=y_test, mode='markers',
                                name="Test")]).update_layout(
        title_text=f"Generate dataset for model of {n_samples} samples with "
                   f"{noise} noise").show()


    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_err = []
    validation_err = []
    for k in range(11):
        train, vali = cross_validate(PolynomialFitting(k), x_train, y_train,
                                     mean_square_error)
        train_err.append(train)
        validation_err.append(vali)

    degrees = [k for k in range(11)]
    go.Figure([go.Scatter(x=degrees, y=train_err, mode="markers+lines",
                          name="training error"),
               go.Scatter(x=degrees, y=validation_err, mode="markers+lines",
                          name="validation err")]).update_layout(
        title_text=f"CV for polynomial fitting of {n_samples} samples with "
                   f"{noise} noise").show()


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k = np.argmin(validation_err)
    poly = PolynomialFitting(k)
    poly.fit(x_train, y_train)
    err = mean_square_error(y_test, poly.predict(x_test))
    print(f"for {n_samples} samples with "
          f"{noise} noise:\n"
          f"the minimal k is {k} with MSE={np.round(err,2)} and validation "
          f"MSE={np.round(validation_err[k],2)}")




def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions
    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate
    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    x, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    x_train, y_train, x_test, y_test = split_train_test(pd.DataFrame(x),
                                                        pd.Series(y),
                                                        n_samples/x.shape[0])
    x_train = x_train.to_numpy()
    y_train = y_train.to_numpy()
    x_test = x_test.to_numpy()
    y_test = y_test.to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lambdas = np.linspace(0.02, 2, num=n_evaluations)
    ridge_train_err = []
    ridge_validation_err = []
    for i in range(n_evaluations):
        ridge = RidgeRegression(lambdas[i])
        train, vali = cross_validate(ridge, x_train, y_train,
                                     mean_square_error)
        ridge_train_err.append(train)
        ridge_validation_err.append(vali)

    lasso_train_err = []
    lasso_validation_err = []
    for i in range(n_evaluations):
        lasso = Lasso(lambdas[i])
        train, vali = cross_validate(lasso, x_train, y_train,
                                     mean_square_error)
        lasso_train_err.append(train)
        lasso_validation_err.append(vali)

    go.Figure([go.Scatter(x=lambdas, y=lasso_train_err, mode="lines",
                          name="training error for lasso"),
               go.Scatter(x=lambdas, y=lasso_validation_err, mode="lines",
                          name="validation error for lasso"),
               go.Scatter(x=lambdas, y=ridge_train_err, mode="lines",
                          name="training error for ridge"),
               go.Scatter(x=lambdas, y=ridge_validation_err, mode="lines",
                          name="validation error for ridge")]).update_layout(
        title_text=f"CV for ridge and lasso").show()


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    ridge_min = lambdas[np.argmin(ridge_validation_err)]
    lasso_min = lambdas[np.argmin(lasso_validation_err)]

    linear = LinearRegression()
    linear.fit(x_train, y_train)
    ridge = RidgeRegression(ridge_min)
    ridge.fit(x_train, y_train)
    lasso = Lasso(lasso_min)
    lasso.fit(x_train, y_train)

    print(f"Best lambda for ridge: {ridge_min} got MSE="
          f"{ridge.loss(x_test, y_test)}\n"
          f"Best lambda for lasso: {lasso_min} got MSE="
          f"{mean_square_error(y_test, lasso.predict(x_test))}\n"
          f"Least Squares got MSE={linear.loss(x_test, y_test)}")




if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()