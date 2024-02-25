# Non-linear Phylogenetic regression using regularized kernels


# Installation

```
pip install phylokrr
```

# Quick overview

## Data simulation
This simulation is based on a given covariance matrix


```python
import random
import numpy as np

# seed for reproducibility
seed = 12038 
np.random.seed(seed)
random.seed(seed)


# cov. matrix obtained from the phylogenetic tree
vcv = np.loadtxt("./data/test_cov2.csv", delimiter=',') 

# Trait simulation under Brownian motion
n = vcv.shape[0]
mean = np.zeros(n)
X = np.random.multivariate_normal(cov=vcv, mean=mean).reshape(-1,1)
# Non-linear response variable (sine curve)
y = np.sin(X*2.1).ravel() + 5 

# Add noise to the response variable
y[::10] += 4 * (0.5 - np.random.rand(X.shape[0] // 10)) 
```
We then split data into training and testing sets, including their covariances

```python
from phylokrr.utils import split_data_vcv

# split data into training and testing sets 
num_test = round(0.5*n)

(X_train  , X_test,  
 y_train  , y_test,  
 vcv_train, vcv_test) = split_data_vcv(X, y, vcv, num_test, seed = seed) # seed defined above
```

## Simple model fitting without Cross-Validation (CV)

```python
from phylokrr.kernels import KRR

# set model
model = KRR(kernel='rbf', fit_intercept= True)

# arbitrarily proposed hyperparameters
params = {'lambda': 2, 'gamma': 2}

# set hyperparamters
model.set_params(**params)

# fit model with phylogenetic covariance matrix
model.fit(X_train, y_train, vcv = vcv_train)
y_pred1 = model.predict(X_test)
```

Let's compare it with the standard phylogenetic regression (i.e., PGLS)

```python
import matplotlib.pyplot as plt

from phylokrr.utils import PGLS

# fit standard phylogenetic regression
b_wls = PGLS(X_train, y_train, vcv_train)
y_pred3 = np.hstack((np.ones((X_test.shape[0],1)), X_test)) @ b_wls

plt.scatter(X_test, y_test , color = 'blue' , alpha=0.5, label = 'Testing (unseen) data')
plt.scatter(X_test, y_pred1, color = 'green', alpha=0.5, label = 'phyloKRR predictions w\o CV')
plt.scatter(X_test, y_pred3, color = 'red', alpha=0.5, label = 'PGLS predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
```
<p align="center">
<img src="https://github.com/Ulises-Rosas/phylokrr/blob/main/data/imgs/phyloKRR_vs_PGLS.png" alt="drawing" width="600px"/>
</p>

## Hyperparameter tuning with CV

```python
from phylokrr.utils import k_fold_cv_random

params = {
    'lambda' : np.logspace(-5, 5, 200, base=2),
    'gamma' : np.logspace(-5, 5, 200,  base=2),
}

best_params = k_fold_cv_random(X_train, y_train, vcv_train,
                                model, 
                                params,
                                folds = 2, 
                                sample = 50)

model.set_params(**best_params)
model.fit(X_train, y_train, vcv = vcv_train)
y_pred_cv = model.predict(X_test)

plt.scatter(X_test, y_test, color = 'blue' , alpha=0.5, label = 'Testing (unseen) data')
plt.scatter(X_test, y_pred_cv, color = 'green', alpha=0.5, label = 'phyloKRR predictions \w CV')
plt.scatter(X_test, y_pred3, color = 'red', alpha=0.5, label = 'PGLS predictions') # y_pred3 defined above
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
```

<p align="center">
<img src="https://github.com/Ulises-Rosas/phylokrr/blob/main/data/imgs/phyloKRR_vs_PGLS_cv.png" alt="drawing" width="600px"/>
</p>


# Reference

Rosas-Puchuri, U., Santaquiteria, A., Khanmohammadi, S., Solis-Lemus, C., & Betancur-R, R. (2023). [Non-linear phylogenetic regression using regularized kernels](https://www.biorxiv.org/content/10.1101/2023.10.04.560983v1.abstract). bioRxiv, 2023-10.
