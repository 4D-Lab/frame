import math
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression


def reg_through_origin(y_true, y_pred):
    """Regression Through Origin (RTO) coefficient of determination

    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels

    Returns:
        float: RTO coefficient of determination
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    true = y_true.reshape(-1, 1)
    pred = y_pred.reshape(-1, 1)

    regression = LinearRegression(fit_intercept=False)
    rto = regression.fit(pred, true)

    rto_r2 = rto.score(pred, true)
    slope = regression.coef_

    return rto_r2, float(slope)


def concordance_correlation(y_true, y_pred):
    """Concordance Correlation Coefficient (CCC)
    https://doi.org/10.2307/2532051

    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels

    Returns:
        float: Coefficient
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mean_true = y_true.mean()
    mean_pred = y_pred.mean()

    vx, cov_xy, cov_xy, vy = np.cov(y_true, y_pred, bias=True).flat
    ccc = 2 * cov_xy / (vx + vy + (mean_true - mean_pred) ** 2)

    return ccc


def roy_criteria(y_true, y_pred, inverse=False):
    """Proposed criteria by Roy based on regression through origin (RTO)
    https://doi.org/10.1016/j.ejps.2014.05.019

    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels

    Returns:
        float: Roy criteria
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if inverse:
        rto, _ = reg_through_origin(y_pred, y_true)
        r2 = metrics.r2_score(y_pred, y_true)

        roy = r2 * (1 - math.sqrt(abs(r2 - rto)))

    else:
        rto, _ = reg_through_origin(y_true, y_pred)
        r2 = metrics.r2_score(y_true, y_pred)

        roy = r2 * (1 - math.sqrt(abs(r2 - rto)))

    return roy


def golbraikh_tropsha(y_true, y_pred):
    """Proposed criteria by Alexander Golbraikh and Alexander Tropsha
    https://doi.org/10.1016/S1093-3263(01)00123-1

    Args:
        y_true (np.array): True labels
        y_pred (np.array): Predicted labels

    Returns:
        float: Golbraikh and Tropsha criteria
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rto_0, _ = reg_through_origin(y_true, y_pred)
    rto_1, _ = reg_through_origin(y_pred, y_true)

    delta = abs(rto_0 - rto_1)

    return delta
