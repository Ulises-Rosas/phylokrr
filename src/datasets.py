import numpy as np
from phylokrr.utils import P_mat_simple, myfunc
import pkg_resources


def load_vcv():
    vcv_file = pkg_resources.resource_stream('phylokrr', 'data/test_cov2.csv')
    vcv = np.loadtxt(vcv_file, delimiter=',')
    return vcv


def load_1d_data_example(return_cov=True):
    
    """
    created with the following code:
    
    ```
    import random
    import numpy as np

    from phylokrr.utils import P_inv_simple
    from phylokrr.datasets import load_vcv

    np.random.seed(12037)


    vcv = load_vcv()
    mean_vector = np.zeros(vcv.shape[0])

    X_w_uc = np.random.normal(0, 1, vcv.shape[0])
    # Non-linear response variable (sine curve)
    y_w_uc = np.sin(X_w_uc*1.5).ravel() 

    # Add noise to the response variable
    y_w_uc[::10] += 4 * (0.5 - np.random.rand(X_w_uc.shape[0] // 10))

    # we can attempt to unweight to the original space
    # with the square root of the covariance matrix
    P_inv = P_inv_simple(vcv)
    X_uw_uc, y_uw_uc = P_inv @ X_w_uc, P_inv @ y_w_uc

    Xy = np.stack([X_uw_uc, y_uw_uc], axis=1)
    np.savetxt('../src/data/test_data_unweigthed.csv', Xy, delimiter=',')
    ```

    """

    data_file = pkg_resources.resource_stream('phylokrr', 'data/test_data_unweigthed.csv')
    data = np.loadtxt(data_file, delimiter=',')
    X, y = data[:,0].reshape(-1,1), data[:,1]

    if return_cov:
        return X, y, load_vcv()
    
    else:
        return X, y


def sim_data(vcv, mean_vector, b, 
             on_weighted=True, 
             add_noise = True, 
             noise_var = 1,
             n_var = 3, 
             extra_weights = [], 
             type='pol',
             weight_with_vcv = True,
             vcv2 = None
             ):
    """
    Simulate weighted data for n_var independent variables

    Parameters
    ----------
    vcv: np.array
        covariance matrix where data is simulated from
    
    mean_vector: np.array
        mean vector
    
    b: int
        power of the polynomial function

    on_weighted: bool
        whether to use weighted data or not to obtain the response variable.
        the weights are obtained using the 
        squared root of the inverse of 
        the covariance matrix

    add_noise: bool
        whether to add noise to the response variable

    noise_var: float
        variance of the noise

    extra_weights: list
        weights for the independent variables

    type: str
        type of the function to simulate the data
        if 'sin' then a sine function is used over the
        sum of the independent variables
        if 'pol' then a polynomial function is used over the
        sum of the independent variables
    
    weight_with_vcv: bool
        if true (default) the weighting of the values is 
        done using vcv. Otherwise, vcv2 is used

    vcv2: np.array
        if weight_with_vcv is false, then we use this covariance matrix
        to weight the data
    
    Returns
    -------
    X_w_uc: np.array
        weighted independent variables (uncentered)
    
    y_w_uc: np.array
        weighted response variable (uncentered)
    """
    # n_var = 1
    # mean_vector = np.zeros(vcv.shape[0])
    # b = 3
    n = vcv.shape[0]
    if len(extra_weights):
        assert len(extra_weights) == n_var, "extra weights must have the same length as the number of predictors"
    
    else:
        extra_weights = np.ones(n_var)

    X_uw_uc = np.zeros((n, n_var))
    for j in range(n_var):
        X_uw_uc[:,j] = np.random.multivariate_normal(mean=mean_vector, cov=vcv)*extra_weights[j]
    
    if add_noise:
        noise = np.random.normal(0, noise_var, n)

    if weight_with_vcv:
        P = P_mat_simple(vcv)

    else:
        assert isinstance(vcv2, np.ndarray), 'vcv2 must be a numpy array'
        P = P_mat_simple(vcv2)

    if on_weighted:
        X_w_uc = P @ X_uw_uc
        # the variance and expectation 
        # of the response variable will depend on the moment 
        # that generates b. V(x^b) = E(x^2b) - E(x^b)^2
        # E(x^b) is the expectation in the moment b.
        # Both the expectation and variance are point estimates
        y_w_uc = myfunc(X_w_uc, b=b, type=type, 
                        add_noise=add_noise,
                        noise=noise).ravel()
    else:
        y_uw_uc = myfunc(X_uw_uc, b=b, type=type, 
                         add_noise=add_noise, 
                         noise=noise).ravel()

        X_w_uc = P @ X_uw_uc
        y_w_uc = P @ y_uw_uc

    return X_w_uc, y_w_uc
