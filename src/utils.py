import random
import numpy as np

def split_data(X,y,num_test, seed = 123):

    random.seed(seed)
    n,_ = X.shape

    test_idx  = random.sample(range(n), k = num_test)
    train_idx = list( set(range(n)) - set(test_idx) )

    X_train, X_test = X[train_idx,:], X[test_idx,:]
    y_train, y_test = y[train_idx]  , y[test_idx]

    return X_train,y_train, X_test, y_test

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
