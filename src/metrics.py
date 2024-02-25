import numpy as np

def R2(y, r):
    """
    Coefficient of determination
    r = y - y_pred 
    """
    # r = y - y_pred

    u = r.T @ r
    v = y.T @ y - len(y) * np.mean(y)**2
    return 1 - (u/v)

def rmse(r):
    """
    Root Mean Squared Error
    r = y - y_pred
    """
    # r = y - y_pred
    return np.sqrt( np.mean( r**2 ) )
