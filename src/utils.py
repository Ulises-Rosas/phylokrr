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


def k_fold_cv_random(X, y, 
                     model, 
                     params,
                     folds = 3, 
                     sample = 500,
                     verbose = False,
                     seed = 123
                     ):
    
    np.random.seed(seed=seed)
    
    all_params = params.keys()
    tested_params = np.ones((sample, len(all_params)))
    for n,k in enumerate(all_params):
        tested_params[:,n] = np.random.choice(params[k], sample)
    
    all_errors = []
    myFolds = k_folds(X, folds)
    for vec in tested_params:
        # vec
        tmp_params = dict(zip(all_params, vec))
        tmp_err = evaluate_folds(X, y, 
                                 myFolds,
                                 model, 
                                 tmp_params)
        all_errors.append([tmp_params, tmp_err])

    best_ = sorted(all_errors, key=lambda kv: kv[1], reverse=False)[0]
    if verbose:
        print("CV score: ", best_[1])

    return best_[0]

