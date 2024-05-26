
import numpy as np
from phylokrr.utils import split_data

def permutation_importance(X, y, model, num_test = 100, seed = 12038, iterations = 1000):
    """
    Permutation importance

    X: data
    y: target
    model: model
    num_test: number of test samples
    seed: random seed
    iterations: number of iterations

    returns: Feature importance for all features. 
        Each column represents the change in error
    """

    np.random.seed(seed)

    n,p = X.shape

    X_train, X_test, y_train,y_test = split_data(X, y, num_test, seed = seed)
    model.fit(X_train, y_train)
    error_orig = model.score(X_test, y_test)


    out = np.zeros((iterations, p))

    for i in range(iterations):

        for j in range(X_test.shape[1]):
            # Create a copy of X_test
            X_test_copy = X_test.copy()

            # Scramble the values of the given predictor
            X_test_copy[:,j] = np.random.permutation(X_test_copy[:,j])
                        
            # Calculate the new RMSE
            error_perm = model.score(X_test_copy, y_test)

            out[i,j] = error_perm - error_orig

    return out

def sample_feature(X, feature, quantiles, sample, integer = False):
    """
    sample unweighted column
    """
    # X = X0
    # feature =0
    Xpdp = X[:,feature]

    if integer:
        return np.sort(np.unique(Xpdp))
    
    q = np.linspace(
        start = quantiles[0],
        stop  = quantiles[1],
        num   = sample
    )

    # sample from the original feature distribution
    # values are not necessarily found in the original
    # feature column
    Xq = np.quantile(Xpdp, q = q)

    # np.any(Xpdp == Xq[1])
    return Xq    

def pdp_1d(X, feature, model, quantiles = [0,1], sample = 70, integer = False):
    """
    1D partial dependence plot

    feature: feature index
    model: model
    quantiles: quantiles to sample feature values
    sample: number of samples
    integer: boolean to indicate if feature is integer

    returns: Xq, pdp_values
    """

    Xq = sample_feature(X, feature, quantiles, sample, integer)
    pdp_values = []
    for n in Xq:
        # copy original values
        X_tmp = X.copy()
        # make original values with 
        # modified feature column
        X_tmp[:,feature] = n

        pdp_values.append(
            np.mean(
                model.predict(X_tmp)
            )
        )

    out = np.hstack((
        Xq.reshape(-1,1), 
        np.array(pdp_values).reshape(-1,1)
        ))
    
    return out

def pdp_2d(X, f1_idx, f2_idx, model, quantiles = [0, 1], sample = 10, integer = [False, False]):
    """
    2D partial dependence plot

    f1_idx: feature 1 index
    f2_idx: feature 2 index
    model: model
    quantiles: quantiles to sample feature values
    sample: number of samples
    integer: boolean list to indicate if feature is integer


    """

    Xq1 = sample_feature(X, f1_idx, quantiles, sample, integer[0])
    Xq2 = sample_feature(X, f2_idx, quantiles, sample, integer[1])

    n_xq1 = len(Xq1)
    n_xq2 = len(Xq2)

    X1, X2 = np.meshgrid(Xq1, Xq2)

    Y = np.zeros((n_xq1, n_xq2))
    for i in range(n_xq1):
        for j in range(n_xq2):
            # j,i = 0,0
            X_tmp = X.copy()
            X_tmp[:,f1_idx] = X1[i,j]
            X_tmp[:,f2_idx] = X2[i,j]

            Y[i,j] = np.mean( model.predict(X_tmp) )

    return X1, X2, Y

