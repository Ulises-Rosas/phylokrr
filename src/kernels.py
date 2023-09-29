import numpy as np

class KRR:

    def __init__(self, kernel = 'rbf') -> None:

        # self.cov = cov
        self.kernel = kernel

        if self.kernel == 'rbf':
            self.params = {'gamma': 0.1, 'lambda': 0.1}

        else:
            self.params = {'c': 0.1, 'lambda': 0.1}

        # internal
        self.alpha = []
        self.X = []
        self.y = []

    def set_params(self, params):
        self.params = params

    def get_params(self):
        return self.params
    
    def fit(self, X, y):
        
        self.X = X
        self.y = y

        if self.kernel == 'rbf':
            K_train = self.RBF_kernel(self.X, self.X, self.params['gamma'])

        else:
            K_train = self.linear_kernel(self.X, self.X, self.params['c'])

        self.alpha = self.opt_alpha(K_train, self.y, self.params['lambda'])

    def predict(self, X_test):

        assert len(self.alpha) > 0, "The model needs to be fitted first"

        if self.kernel == 'rbf':
            K_test = self.RBF_kernel(X_test, self.X, self.params['gamma'])

        else:
            K_test = self.linear_kernel(X_test, self.X, self.params['c'])
        
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

    def distance_matrix(self,a, b):
        """
        l2 norm squared matrix
        """
        return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)**2

    def RBF_kernel(self, a, b, gamma):
        """
        Radial Basis Function
        """        
        tmp_rbf = -gamma * self.distance_matrix(a, b)
        np.exp(tmp_rbf, tmp_rbf) # RBF kernel. Inplace exponentiation
        return tmp_rbf

    def linear_kernel(self, a, b, c):
        """
        Linear Kernel
        """
        XXt = a.dot(b.T)
        C = c * np.ones(XXt.shape)

        return XXt + C
    
    def rmse(self, K, alpha, Y):

        return np.sqrt( np.mean( (K.dot(alpha) - Y)**2 ) )
    
    def opt_alpha(self, K, y, reg_lam = None):
        # Y = y
        # X = X
        # m = None
        K_Idx = K + reg_lam * np.diag( np.ones( K.shape[0] ) )
        return np.linalg.inv( K_Idx ).dot( y )

