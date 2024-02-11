
import numpy as np

from phylokrr.utils import split_data, k_fold_cv_random
from phylokrr.kernels import KRR


class PhylogeneticRegressor:

    def __init__(self, X, y, cov, kernel='rbf') -> None:

        self.X = X
        self.y = y
        self.vcv = self.read_cov(cov)
        self.model = KRR(kernel=kernel)

        self.check_shapes()

        # weighting data by the cov matrix
        self.P, self.X1, self.y1 = self.P_mat(chol = False)


        self.hyperparamter_space = {}
        self.P_inv = np.array([])
        self.test_size = None

    
    def check_shapes(self):
        assert self.vcv.shape[0] == self.vcv.shape[1], 'not square cov matrix'
        assert self.X.shape[0] == self.y.shape[0] == self.vcv.shape[0], 'dimensions do not match'

        # assert self.check_matrix(self.vcv), 'not symmetric matrix'
    def check_symmetric(self, a, tol=1e-8):
        return np.all(np.abs(a-a.T) < tol)        

    def read_cov(self, cov):

        if isinstance(cov, np.ndarray):
            return cov
        
        else:
            return np.loadtxt(cov, delimiter=',')
        
    def P_mat(self, chol = False):

        if chol:
            P = np.linalg.cholesky( np.linalg.inv( self.vcv ) )

        else:
            Oinv = np.linalg.inv( self.vcv )
            L,Q  = np.linalg.eig( Oinv )
            P  = Q @ np.diag( np.sqrt( 1/L ) ) @ Q.T

        return P, P @ self.X, P @ self.y
    
    def split_data(self, seed = 123):
        """
        split weighted data
        """
        num_test = round( self.X.shape[0] * self.test_size )

        return split_data(self.X1, self.y1, num_test, seed = seed)
    
    def set_hyperparameter_space(self, params):
        self.hyperparamter_space = params

    def fit(self, 
            cv = 3, 
            sample = 100, 
            test_size = 0.4, 
            seed = 123,
            verbose = True
            ):
        
        """
        Train model with weighted data

        Parameters
        ----------
        cv : int, default = 3
            number of cross validation rounds 
            for hyperparameter tuning with the 
            training set
            
        sample: int, default = 100
            sample from the hyperparameter space 
            for hyperparameter tuning with the
            training set
        """
        self.test_size = test_size

        X_train, y_train, X_test, y_test = self.split_data(seed = seed)

        assert len(self.hyperparamter_space), 'set hyperparameter space'

        # X_train.shape
        hyperparameters = k_fold_cv_random(
                            X_train,
                            y_train,
                            self.model,
                            self.hyperparamter_space,
                            folds  = cv,
                            sample = sample,
                            verbose=verbose
                            )
        
        if verbose:
            print('hyperparameters: %s' % hyperparameters)

        self.model.set_params(**hyperparameters)
        self.model.fit(X_train, y_train)

        if verbose:
            testing_error = self.model.score(X_test, y_test, metric='rmse')
            training_error = self.model.score(X_train, y_train, metric='rmse')
            print('Training error: %s' % training_error)
            print('Testing error: %s' % testing_error)
            
    def set_params(self, hyperparameters, fit = False, test_size = 0.4, seed = 123):
        """
        Set new hyperparameters

        Parameters
        ----------
        refit : bool, default = False
            refit model after setting hyperparamters
            
        test_size: float, default = 0.4
            if refit, the test size proportion
        """
        self.model.set_params(**hyperparameters)

        if fit:
            self.test_size = test_size

            X_train, y_train, X_test, y_test = self.split_data(seed = seed)
            self.model.fit(X_train, y_train)

            testing_error  = self.model.score(X_test, y_test, metric='rmse')
            training_error = self.model.score(X_train, y_train, metric='rmse')

            print('Training error: %s' % training_error)
            print('Testing error: %s' % testing_error)


    def sample_feature(self, feature, quantiles, sample, integer = False):
        """
        sample unweighted column
        """
        # X = X0
        # feature =0
        Xpdp = self.X[:,feature]

        if integer:
            return np.sort(np.unique(Xpdp))
        
        q = np.linspace(
            start = quantiles[0],
            stop  = quantiles[1],
            num   = sample
        )

        # sample from the original feature space
        Xq = np.quantile(Xpdp, q = q)

        return Xq    

    def pdp(self, feature, integer = False, quantiles = [0,1], sample = 70):
        """
        partial depedence plot values.

        Parameters
        ----------
        feature : int

        Returns
        -------
        PDP of the transformed data
        """
        if not len(self.P_inv):
            self.P_inv = np.linalg.inv(self.P)

        Xq = self.sample_feature(feature, quantiles, sample, integer)
        
        pdp_values = []
        for n in Xq:
            # copy original values
            X_tmp = self.X.copy()
            # make original values with 
            # modified feature column
            X_tmp[:,feature] = n
            # weight the whole dataset as
            # the model learned from the 
            # weighted data
            X_tmp_test = self.P @ X_tmp

            pdp_values.append(
                np.mean(
                    # make rows unweighted
                    # (i.e., again 'correlated')
                    self.P_inv @ self.model.predict(X_tmp_test)
                )
            )
 
        out = np.hstack((
            Xq.reshape(-1,1), 
            np.array(pdp_values).reshape(-1,1)
        ))
        return out
    
    def FeatureImportance(self, seed = 123):
        """
        permutation feature importance
        """        
        assert len(self.model.alpha), 'model needs to be fitted'

        np.random.seed(seed=seed)
        

        X_train, y_train, X_test, y_test = self.split_data(seed = seed)
        self.model.fit(X_train, y_train)
        error_orig = self.model.score(X_test, y_test)


        out = []
        for i in range(X_test.shape[1]):
            # Create a copy of X_test
            X_test_copy = X_test.copy()

            # Scramble the values of the given predictor
            X_test_copy[:,i] = np.random.permutation(X_test_copy[:,i])
                        
            # Calculate the new RMSE
            error_perm = self.model.score(X_test_copy, y_test)

            out.append(error_perm - error_orig)

        return out
    
class PhylogeneticLogisticRegressor:
    def __init__(self) -> None:
        pass


