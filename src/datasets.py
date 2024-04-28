import numpy as np
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
