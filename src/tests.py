
import numpy as np
from phylokrr.core import PhylogeneticRegressor

def dim_red(X, n_com):
    # extracting averages
    X_centered = (X - np.mean(X, axis=0))

    # U,E,Vt (SVD)
    _,_,Vt = np.linalg.svd(X_centered)
    # PCA components
    W = Vt.T[:, :n_com]
    # projection into n_com dimensions
    return X_centered.dot(W)



# get data
data = np.loadtxt('../data/test_data.csv', delimiter=',')
X,y = data[:,:-1], data[:,-1]


X1D = dim_red(X, 1)

import matplotlib.pyplot as plt
plt.scatter(X1D[:,0], y,)


# get covariance matrix
cov = np.loadtxt('../data/test_cov.csv',delimiter=',')

# set model
model = PhylogeneticRegressor(X, y, cov, kernel='rbf')

# set hyperparamter space
params = {
    'lambda' : np.logspace(-5, 15, 100, dtype = float, base=2),
    'gamma' : np.logspace(-15, 3, 100, dtype = float, base=2),
}
model.set_hyperparameter_space(params=params)
    
# fit the model
model.fit(seed=12)



# also you can set new set of hyperparameters
# and then fit
model.set_params(
    {'lambda': 0.0001, 'gamma': 0.0755},
    fit=True
)

# plotting
import matplotlib.pyplot as plt


feature_names = ['feature 1', 'feature 2', 'feature 3']

FI = model.FeatureImportance(seed=None)
plt.bar(feature_names, FI)
plt.ylabel('RMSE increase')

# len(model.P_inv)
# partial depedence plot

pdp_0 = model.pdp(feature=0)
pdp_1 = model.pdp(feature=1)
pdp_2 = model.pdp(feature=2)






plt.rcParams['figure.figsize'] = [10, 4]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
fig.suptitle('Partial Dependence Plot')

ax1.scatter(pdp_0[:,0], pdp_0[:,1])
ax1.set_ylabel('Predicted y')
ax1.set_xlabel('feature 1')

ax2.scatter(pdp_1[:,0], pdp_1[:,1])
ax2.set_xlabel('feature 2')

ax3.scatter(pdp_2[:,0], pdp_2[:,1])
ax3.set_xlabel('feature 3')


## 
data = np.loadtxt('../data/test_log_data.csv', delimiter=',')
X,y = data[:,:-1], data[:,-1]
import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,2], c = y, alpha=0.6)


# file_path = "/Users/ulises/Desktop/ABL/software/phylokrr/data/test.tree"

# import dendropy
# tree = dendropy.Tree.get( path = 'path/to/tree', schema = 'newick')

# for i in tree.postorder_node_iter():
#     print( i.distance_from_root() )
