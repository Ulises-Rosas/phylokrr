import numpy as np


def distance_matrix(a, b):
    """
    l2 norm squared matrix
    """
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)**2

def RBF_kernel(a, b, gamma):
    """
    Radial Basis Function
    """        
    tmp_rbf = -gamma * distance_matrix(a, b)
    np.exp(tmp_rbf, tmp_rbf) # RBF kernel. Inplace exponentiation
    return tmp_rbf

def linear_kernel(a, b, c):
    """
    Linear Kernel
    """
    XXt = a.dot(b.T)
    C = c * np.ones(XXt.shape)

    return XXt + C



class KRR:

    def __init__(self, kernel = 'rbf') -> None:

        self.kernel = kernel

        if self.kernel == 'rbf':
            self.params = {'gamma': 0.1, 'lambda': 0.1}

        else:
            self.params = {'c': 0.1, 'lambda': 0.1}

        # internal
        self.alpha = []
        self.X = []
        self.y = []

    def set_params(self, **params):

        if self.kernel == 'rbf':
            self.params['gamma'] = params['gamma']

        else:
            self.params['c'] = params['c']

        self.params['lambda'] = params['lambda']

    def get_params(self):
        return self.params
    
    def fit(self, X, y):
        
        self.X = X
        self.y = y

        if self.kernel == 'rbf':
            K_train = RBF_kernel(self.X, self.X, self.params['gamma'])

        else:
            K_train = linear_kernel(self.X, self.X, self.params['c'])

        self.alpha = self.opt_alpha(K_train, self.y, self.params['lambda'])

    def predict(self, X_test):

        assert len(self.alpha) > 0, "The model needs to be fitted first"

        if self.kernel == 'rbf':
            K_test = RBF_kernel(X_test, self.X, self.params['gamma'])

        else:
            K_test = linear_kernel(X_test, self.X, self.params['c'])
        
        return K_test @ self.alpha
    
    def score(self, X_test, y_test, metric = 'rmse'):

        y_pred = self.predict(X_test)

        if metric == 'rmse':
            return np.sqrt( np.mean( (y_pred - y_test)**2 ) )
    
        else:
            # r2
            u = ((y_test - y_pred)**2).sum()
            v = ((y_test - y_test.mean())** 2).sum()

            return 1 - (u/v)
    
    def rmse(self, K, alpha, Y):

        return np.sqrt( np.mean( (K.dot(alpha) - Y)**2 ) )
    
    def opt_alpha(self, K, y, reg_lam = None):
        # Y = y
        # X = X
        # m = None
        n,_ = self.X.shape
        K_Idx = K + n * reg_lam * np.diag( np.ones( K.shape[0] ) )
        return np.linalg.inv( K_Idx ).dot( y )


class LKRR:

    def __init__(self, kernel = 'rbf') -> None:

        self.kernel = kernel

        if self.kernel == 'rbf':
            self.params = {'gamma': 0.1, 'lambda': 0.1}

        else:
            self.params = {'c': 0.1, 'lambda': 0.1}

        # internal
        self.alpha = []
        self.X = []
        self.y = []

    def p1(self, X, beta):
        return 1/(1 + np.exp(X.dot(beta)))

    def p0(self, X, beta):
        EX = np.exp(X.dot(beta))
        return EX/(1 + EX)

    def set_params(self, params):
        self.params = params

    def get_params(self):
        return self.params
    
    def fit(self, X, y):
        
        self.X = X
        self.y = y

        if self.kernel == 'rbf':
            K_train = RBF_kernel(self.X, self.X, self.params['gamma'])

        else:
            K_train = linear_kernel(self.X, self.X, self.params['c'])

        self.alpha = self.opt_alpha(K_train, self.y, self.params['lambda'])

    def predict(self, X_test):

        assert len(self.alpha) > 0, "The model needs to be fitted first"

        if self.kernel == 'rbf':
            K_test = RBF_kernel(X_test, self.X, self.params['gamma'])

        else:
            K_test = linear_kernel(X_test, self.X, self.params['c'])
        
        return K_test @ self.alpha
    
    def score(self, X_test, y_test, metric = 'rmse'):

        y_pred = self.predict(X_test)

        if metric == 'rmse':
            return np.sqrt( np.mean( (y_pred - y_test)**2 ) )
    
        else:
            # r2
            u = ((y_test - y_pred)**2).sum()
            v = ((y_test - y_test.mean())** 2).sum()

            return 1 - (u/v)
    
    def rmse(self, K, alpha, Y):

        return np.sqrt( np.mean( (K.dot(alpha) - Y)**2 ) )
    
    def opt_alpha(self, K, y, reg_lam = None):
        # Y = y
        # X = X
        # m = None
        K_Idx = K + reg_lam * np.diag( np.ones( K.shape[0] ) )
        return np.linalg.inv( K_Idx ).dot( y )


# def P_mat(vcv, chol = False, corr = False):
    
#     if corr:
#         Kr = np.diag(1/np.sqrt(np.diag(vcv)))
#         vcv = Kr @ vcv @ Kr

#     if chol:
#         P = np.linalg.cholesky( np.linalg.inv( vcv ) )

#     else:
#         Oinv = np.linalg.inv( vcv )
#         L,Q  = np.linalg.eig( Oinv )
#         P  = Q @ np.diag( np.sqrt( 1/L ) ) @ Q.T

#     return P


# ## logistic data
# data = np.loadtxt('../data/test_log_data.csv', delimiter=',')

# # get covariance matrix
# cov = np.loadtxt('../data/test_cov.csv',delimiter=',')


# [n,p] = np.shape(data)
# num_train = int(0.5*n)

# X,y = data[:,:-1], data[:,-1]
# P = P_mat(cov,corr=True)

# X = P @ X
# y = P @ y





# sample_train = X[0:num_train,:]
# sample_test  = X[num_train: ,:]
    
# label_train = y[0:num_train]
# label_test  = y[num_train: ]
    


# # import matplotlib.pyplot as plt
# # plt.scatter(X[:,0], X[:,2], c = y, alpha=0.6)

# import warnings

# def p1(X, beta):
#     return 1/(1 + np.exp(X.dot(beta)))

# def p0(X, beta):
#     EX = np.exp(X.dot(beta))
#     return EX/(1 + EX)

# def error_est(X, y,  beta, norm = False):
#     if norm:
#         return np.linalg.norm(p1(X,beta) - y, ord = 2)
#         # pass
#     else:
#         y_pred = (p1(X, beta) >= 0.5).astype(float)
#         return np.mean(y_pred != y)

# def NewtonRaphsonStep(X, y, beta):

#     Lp = -X.T @ ( y - p1(X, beta) )
#     W  = np.diag( p1(X, beta) * p0(X, beta) )

#     Lpp = -X.T @ W @ X
#     return  beta - np.linalg.inv(Lpp) @ Lp


# n_updts = 20
# beta0 = np.array(np.random.normal(size = p - 1))
# beta = np.copy(beta0)

# epsilon = 1e-6
# for i in range(n_updts):
#     # i = 0
#     beta_old = np.copy(beta)

#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")

#         beta = NewtonRaphsonStep(sample_train, label_train, beta)

#         if np.any( np.isnan(beta) ):
#             print('beta re-redifined')
#             beta = np.random.normal(size = p - 1)

#         error = error_est(sample_test, label_test, beta, norm=True)
#         print("update = %i, error = %s" %(i, round(error, 5))) 

#     if error <= epsilon:
#         print('error convergence')
#         break

#     beta_norm = np.linalg.norm(beta_old - beta, ord = np.inf )
#     if np.any( beta_norm <= epsilon  ):
#         print('coefficient convergence')
#         break

# y_pred = (p1(sample_test, beta) >= 0.5).astype(float)

# from sklearn.metrics import confusion_matrix
# confusion_matrix(label_test,y_pred)
