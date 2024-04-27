import random
import numpy as np

def split_data(X, y, num_test, seed = 123):
    random.seed(seed)
    n,_ = X.shape

    test_idx = random.sample(range(n), k = num_test)
    train_idx = list( set(range(n)) - set(test_idx) )

    X_train, X_test = X[train_idx,:], X[test_idx,:]
    y_train, y_test = y[train_idx]  , y[test_idx]

    return X_train, X_test, y_train, y_test

def split_data_vcv(X,y,vcv, num_test, seed = 123):
    
    random.seed(seed)
    n,_ = X.shape

    test_idx  = random.sample(range(n), k = num_test)
    train_idx = list( set(range(n)) - set(test_idx) )

    X_train, X_test = X[train_idx,:], X[test_idx,:]
    y_train, y_test = y[train_idx]  , y[test_idx]
    vcv_train, vcv_test = vcv[train_idx,:][:,train_idx], vcv[test_idx,:][:,test_idx]

    return X_train, X_test, y_train, y_test, vcv_train, vcv_test

def k_folds(X, folds = 3, seed = 123):
    """
    test_indx, train_indx
    """
    # X = X_train
    # folds = 4
    random.seed(seed)
    
    n,_ = X.shape
    all_index = list(range(n))
    random.shuffle(all_index)

    window = n/folds

    k_folds = []

    i = 0
    while True:

        init_indx = i
        end_indx  = i + window

        test_indx = all_index[round(init_indx):round(end_indx)]
        train_indx = list(set(all_index) - set(test_indx))
        # print(init_indx, end_indx)
        k_folds.append([test_indx, train_indx])

        i += window
        if i >= n:
            break

    # len(k_folds)
    return k_folds

def evaluate_folds(X, y, myFolds, model, tmp_params):

    # print(kwargs)
    # kwargs = {'c': 0.4, 'lambda': 0.1}
    # params = {'gamma': 0.4, 'lambda': 0.1}
    # params = tmp_params
    # model = phyloKRR(kernel='rbf')

    model.set_params(tmp_params)
    # model.get_params()

    all_errs = []
    for test_indx, train_indx in myFolds:
        # print(len(test_indx), len(train_indx))
        X_train,y_train = X[train_indx,:], y[train_indx]
        X_test,y_test = X[test_indx,:], y[test_indx]

        model.fit(X_train, y_train)

        # print(np.var(X_train))
        # print(np.var(model.X))
        # print(np.var(model.alpha))

        tmp_err = model.score(X_test, y_test, metric = 'rmse')
        all_errs.append(tmp_err)

    # return np.mean(all_errs)
    return np.median(all_errs)

def k_fold_cv(X, y, vcv, model, num_folds):
    """
    k-fold cross-validation with covariance matrix
    """
    n, p = X.shape
    fold_size = n // num_folds
    mse_sum = 0

    for i in range(num_folds):

        test_idx = list(range(i * fold_size, (i + 1) * fold_size))
        train_idx = list(set(range(n)) - set(test_idx))

        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]

        vcv_train = vcv[train_idx,:][:,train_idx]
        vcv_test = vcv[test_idx,:][:,test_idx]

        model.fit(X_train, y_train, vcv = vcv_train)
        mse_sum += model.score(X_test, y_test, vcv_test)

    return mse_sum / num_folds

def k_fold_cv_random(X, y, vcv,
                     model, 
                     params,
                     folds = 3, 
                     sample = 500,
                     verbose = True,
                     seed = 123
                     ):
    """
    Random search for hyperparameter tuning using k-fold cross-validation
    and covariance matrix
    """
    
    np.random.seed(seed=seed)
    # make random choice from the grid of hyperparameters
    all_params = params.keys()
    tested_params = np.zeros((sample, len(all_params)))
    for n,k in enumerate(all_params):
        tested_params[:,n] = np.random.choice(params[k], sample)

    if verbose:
        # check tested_params are unique
        tested_params = np.unique(tested_params, axis=0)
        print("Number of unique hyperparameters: ", tested_params.shape[0])
    
    all_errors = []
    for vec in tested_params:
        tmp_params = dict(zip(all_params, vec))
        model.set_params(**tmp_params)
        tmp_err = k_fold_cv(X, y, vcv, model, folds)
        all_errors.append([tmp_params, tmp_err])

    best_ = sorted(all_errors, key=lambda kv: kv[1], reverse=False)[0]

    if verbose:
        print("CV score: ", best_[1])

    return best_[0]

def PGLS(X, y, vcv):
    """
    Generalized Least Squares with phylogenetic covariance matrix
    as the weight matrix
    """
    n,p = X.shape
    Oinv = np.linalg.inv(vcv)
    X_ = np.hstack((np.ones((n,1)), X))
    beta = np.linalg.inv(X_.T @ Oinv @ X_) @ X_.T @ Oinv @ y
    return beta


def _scaler(ref, dat, use_sd = False):
    """
    Center and scale data
    """

    u = np.mean(ref, axis=0)
    centered = dat - u

    if use_sd:
        sd = np.std(ref, axis=0)
        sted = centered / sd

        return sted
    else:

        return centered

class LS:
    def __init__(self, fit_intercept=False, weighted=False) -> None:
        self.fit_intercept = fit_intercept
        self.weighted = weighted
        self.intercept = 0
        self.beta = np.array([])

    def fit(self, X, y,):
        n, p = X.shape

        if self.weighted:
            tmp_beta = np.linalg.solve(X.T @ X, X.T @ y)

        if self.fit_intercept:
            self.intercept = np.mean(y - X @ tmp_beta)

        self.beta = np.hstack((self.intercept, tmp_beta)) if self.fit_intercept else tmp_beta

    def predict(self, X):
        n, p = X.shape

        if self.fit_intercept:
            X = np.hstack((np.ones((n, 1)), X))

        return X @ self.beta

    def score(self, X_test, y_test, metric='rmse'):
        y_pred = self.predict(X_test)

        if metric == 'rmse':
            return np.sqrt(np.mean((y_pred - y_test) ** 2))
        else:
            u = ((y_test - y_pred) ** 2).sum()
            v = ((y_test - y_test.mean()) ** 2).sum()

            return 1 - (u / v)


def P_mat_simple(vcv):
    """
    get the square root of the inverse of the
    """

    # Kr = np.diag(1/np.sqrt(np.diag(vcv)))
    # vcv = Kr @ vcv @ Kr

    if isinstance(vcv, type(None)):
        return None

    L,Q = np.linalg.eig( vcv )
    P = Q @ np.diag( L**(-1/2) ) @ Q.T

    return P


def myfunc(x, b=0, type='sin', add_noise = False, noise_var = 1, noise_freq = None):

    noise = np.random.normal(1, noise_var, x.shape[0])

    if type == 'sin':
        y =  np.sin( np.sum(x, axis=1) * b).ravel()
        # if add_noise:
            # y += noise
            # if noise_freq:
            #     y[::noise_freq] += 3 * (0.5 - np.random.rand(x.shape[0] // noise_freq))
        # if add_noise:
        #     y[::noise_freq] += 3 * (0.5 - np.random.rand(x.shape[0] // noise_freq))

    else:
        y =  np.sum(x, axis=1) ** b

    if add_noise:
        y += noise

    return y


def sim_data(vcv, mean_vector, b, 
             on_weighted=False, 
             add_noise = False, 
             noise_var = 1,
             noise_freq = 10, # only applied to sine type of function
             n_var = 3, 
             extra_weights = None, 
             type='pol',
             weight_with_vcv = True,
             vcv2 = None
             ):
    """
    Simulate data for n_var independent variables

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
    # n_var = 3

    if extra_weights:
        assert len(extra_weights) == n_var, "extra weights must have the same length as the number of predictors"
    
    else:
        extra_weights = np.ones(n_var)

    X_uw_uc = np.zeros((vcv.shape[0], n_var))
    for j in range(n_var):
        X_uw_uc[:,j] = np.random.multivariate_normal(mean=mean_vector, cov=vcv)*extra_weights[j]


    if weight_with_vcv:
        P = P_mat_simple(vcv)

    else:
        assert isinstance(vcv2, np.ndarray), 'vcv2 must be a numpy array'
        P = P_mat_simple(vcv2)

    if on_weighted:
        X_w_uc = P @ X_uw_uc
        y_w_uc = myfunc(X_w_uc, b=b, type=type, 
                        add_noise=add_noise,
                        noise_freq=noise_freq, 
                        noise_var=noise_var).ravel()

        # plt.scatter(np.sum(X_w_uc, 1), y_w_uc, color='blue', alpha=0.5)
    else:
        y_uw_uc = myfunc(X_uw_uc, b=b, type=type, 
                         add_noise=add_noise, 
                         noise_freq=noise_freq,
                         noise_var=noise_var).ravel()

        X_w_uc = P @ X_uw_uc
        y_w_uc = P @ y_uw_uc

    return X_w_uc, y_w_uc

def get_Xy_w_uc(X_uw_uc, P, b, type, on_weighted, add_noise, noise):    

    if on_weighted:

        X_w_uc = P @ X_uw_uc
        y_w_uc = myfunc(X_w_uc, b=b, type=type).ravel()

        if add_noise:
            y_w_uc += noise

    else:
        y_uw_uc = myfunc(X_uw_uc, b=b, type=type).ravel()

        if add_noise:
            y_uw_uc += noise        

        X_w_uc = P @ X_uw_uc
        y_w_uc = P @ y_uw_uc

    return X_w_uc, y_w_uc

def get_Xy_wc(X_uw_uc, P_mat, b, type, on_weighted, add_noise, noise):    

    if on_weighted:
        
        X_w_uc = P_mat @ X_uw_uc
        y_w_uc = myfunc(X_w_uc, b=b, type=type).ravel()

        if add_noise:
            y_w_uc += noise

    else:
        y_uw_uc = myfunc(X_uw_uc, b=b, type=type).ravel()

        if add_noise:
            y_uw_uc += noise        

        X_w_uc = P_mat @ X_uw_uc
        y_w_uc = P_mat @ y_uw_uc

    X_wc = _scaler(X_w_uc, X_w_uc, use_sd= not False)
    y_wc = _scaler(y_w_uc, y_w_uc, use_sd=  False)

    return X_wc, y_wc

def sim_data_signal(vcv, mean_vector, b, 
             on_weighted=False, 
             add_noise = False, 
             noise_var = 1,
             n_var = 3, 
             extra_weights = [1,1,1], 
             type='pol'):
    """
    Simulate data for n_var independent variables

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
    # n_var = 3

    assert len(extra_weights) == n_var, "extra weights must have the same length as the number of predictors"

    X_uw_uc = np.zeros((vcv.shape[0], n_var))
    for j in range(n_var):
        X_uw_uc[:,j] = np.random.multivariate_normal(mean=mean_vector, cov=vcv)*extra_weights[j]


    noise = np.random.normal(0, noise_var, X_uw_uc.shape[0])


    P1 = P_mat_simple(vcv)
    P2 = P_mat_simple(np.diag(np.diag(vcv)))
    
    (X_w_uc_1, y_w_uc_1) = get_Xy_w_uc(X_uw_uc, P1, b, type,
                                       on_weighted,
                                       add_noise, noise)

    (X_w_uc_2, y_w_uc_2) = get_Xy_w_uc(X_uw_uc, P2, b, type,
                                       on_weighted,
                                       add_noise, noise)

    return (X_w_uc_1, X_w_uc_2, y_w_uc_1, y_w_uc_2)
