from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    x_split = np.array_split(X, cv)
    y_split = np.array_split(y, cv)
    avg_train_err = []
    avg_validation_err = []
    for i in range(cv):
        x_folds = np.concatenate(np.delete(x_split, i, axis=0))
        y_folds = np.concatenate(np.delete(y_split, i, axis=0))
        estimator.fit(x_folds, y_folds)
        avg_train_err.append(scoring(y_folds, estimator.predict(x_folds)))
        avg_validation_err.append(scoring(y_split[i], estimator.predict(
            x_split[i])))

    return np.mean(avg_train_err), np.mean(avg_validation_err)

