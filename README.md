# Non-linear phylogenetic regression using regularized kernels


# Installation

```
pip install phylokrr
```

# Quick overview


### Data

The data used below is obtained from simulation in such a way that the phylogenetically weighted observations follows a sine curve in the response variable. All these files are available at `src/data` folder.


```python
import numpy as np
import matplotlib.pyplot as plt


# load phylokrr functions
from phylokrr.dataio import read_data
from phylokrr.utils import weight_data

# tree file in newick format
tree_file = "./src/data/test_tree.txt"

# data file in csv format, without column names
data_file = "./src/data/test_data3.csv"

# This file contains only a list species names, each 
# corresponding to a row in `data_file`. 
data_file_spps = "./src/data/test_data_spps.txt"

# Read data
X_uw_uc, y_uw_uc, vcv = read_data(tree_file, data_file, data_file_spps,
                                  y_col = 1, # response variable column
                                  delimiter = ',', verbose = True)

# Weight data
X_w, y_w = weight_data(X_uw_uc, y_uw_uc, vcv, use_sd = False)


np.random.seed(12038)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
axs[0].scatter(X_uw_uc, y_uw_uc, color = 'red', alpha=0.5, )
axs[0].set_title('Unweighted data')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[1].scatter(X_w, y_w, color = 'blue', alpha=0.5, )
axs[1].set_title('Phylogenetically weighted data')
axs[1].set_xlabel('$x^*$')
axs[1].set_ylabel('$y^*$')
plt.tight_layout()
```

<p align="center">
<img src="https://github.com/Ulises-Rosas/phylokrr/blob/main/data/imgs/data_plotting.png" alt="drawing" width="800px"/>
</p>

### Simple model fitting without Cross-Validation (CV)


```python
from phylokrr.utils import split_data

n,_ = X_w.shape
# split data into training and testing sets
num_test = round(0.5*n)

(X_train, X_test,
 y_train, y_test,) = split_data(X_w, y_w, num_test, seed = 12038) # seed defined above


from phylokrr.kernels import KRR
# set model
pkrr_model = KRR(kernel='rbf', fit_intercept= True)
# arbitrarily proposed hyperparameters
params = {'lambda': 1, 'gamma': 1}
# set hyperparamters
pkrr_model.set_params(**params)
# fit model
pkrr_model.fit(X_train, y_train,)
# make predictions
y_pred_kernel = pkrr_model.predict(X_test)
```

Let's compare it with the standard phylogenetic regression (i.e., PGLS)

```python
from phylokrr.utils import PGLS

# fit standard phylogenetic regression
pgls = PGLS(fit_intercept=True)
pgls.fit(X_train, y_train)
y_pred_pgls = pgls.predict(X_test)

# plot model fits
plt.scatter(X_test, y_test , color = 'blue' , alpha=0.5, label = 'Testing (unseen) data')
plt.scatter(X_test, y_pred_kernel, color = 'green', alpha=0.5, label = 'phyloKRR predictions w\o CV')
plt.scatter(X_test, y_pred_pgls, color = 'red', alpha=0.5, label = 'PGLS predictions')
plt.xlabel('$x^*$')
plt.ylabel('$y^*$')
plt.legend()
plt.tight_layout()
```


<p align="center">
<img src="https://github.com/Ulises-Rosas/phylokrr/blob/main/data/imgs/fit_wo_cv.png" alt="drawing" width="600px"/>
</p>


### Hyperparameter tuning with CV


```python
from phylokrr.utils import k_fold_cv_random
params = {
    'lambda' : np.logspace(-5, 3, 200, base=2),
    'gamma' : np.logspace(-5, 3, 200,  base=2),
}

# cross validation
best_params = k_fold_cv_random(X_train, y_train,
                                pkrr_model,
                                params,
                                verbose = False,
                                folds = 3,
                                sample = 100)

pkrr_model.set_params(**best_params)
pkrr_model.fit(X_train, y_train)
y_pred_cv = pkrr_model.predict(X_test)

# plot model fits
fs = 10
plt.scatter(X_test, y_test, color = 'blue' ,
            alpha=0.5,
            label = 'Testing (unseen) data',)
plt.scatter(X_test, y_pred_cv,
            color = 'green', alpha=0.5,
            label = 'phyloKRR predictions',)
plt.scatter(X_test, y_pred_pgls,
            color = 'red', alpha=0.5,
            label = 'PGLS predictions',)
plt.xlabel('$x^*$', fontsize = fs)
plt.ylabel('$y^*$', fontsize = fs)
plt.legend(fontsize = fs)
plt.tight_layout()
```

<p align="center">
<img src="https://github.com/Ulises-Rosas/phylokrr/blob/main/data/imgs/fit_w_cv.png" alt="drawing" width="600px"/>
</p>

### Model performance metrics

```python
r2_kernel = pkrr_model.score(X_test, y_test, metric='r2')
r2_pgls = pgls.score(X_test, y_test, metric='r2')

print(f"R2 for phyloKRR: {r2_kernel}")
print(f"R2 for PGLS: {r2_pgls}")
```
```
R2 for phyloKRR: 0.752485346280156
R2 for PGLS: 0.4126035522866096
```

# More information at these notebooks

* [Quick Overview](https://colab.research.google.com/drive/1TrQymi-D6B4KCmWciqneMzMDfTEcTSYX?usp=sharing)
* [Model Inspection](https://colab.research.google.com/drive/1sW67wIf7IH30zpLPe0qo8wlvBOTLYLaU?usp=sharing)
* [Multi-output Regression](https://colab.research.google.com/drive/1wGNtyyl_0taAgUCktLr1tbTv-nDrgTa0?usp=sharing)
* [RBF kernel approximating a linear model](https://colab.research.google.com/drive/1u-KKbQTrhj-GRCyMC7vL_t9XXfD9xGsq?usp=sharing)


# Reference

Rosas‐Puchuri, U., Santaquiteria, A., Khanmohammadi, S., Solís‐Lemus, C., & Betancur‐R, R. (2024). [Non‐linear phylogenetic regression using regularised kernels](https://besjournals.onlinelibrary.wiley.com/doi/abs/10.1111/2041-210X.14385). Methods in Ecology and Evolution.
