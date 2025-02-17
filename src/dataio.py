import numpy as np
from collections import deque
from phylokrr.treeio import get_vcv

def one_hot(cat_feat):
    tmp_uniq = list(set(cat_feat))
    # print(tmp_uniq)
    feat_encod = []
    for tmp_str in cat_feat:
        tmp_encod = [0.]*len(tmp_uniq)
        tmp_encod[tmp_uniq.index(tmp_str)] = 1.0
        feat_encod.append(tmp_encod) 
        
    return feat_encod

def get_Xy(data_file, verbose = False, delimiter = ',', y_col = -1):
    """
    Read data file and return predictors and response variable.
    
    Parameters
    ----------
    data_file : str
        path to the data file

    verbose : bool
        print information about the data

    delimiter : str
        delimiter of the data file

    y_col : int
        column index of the response variable

    Returns
    -------
    X : numpy array
        predictors

    y : numpy array
        response variable
    """

    data = np.loadtxt(data_file, 
                         delimiter=delimiter, 
                         dtype=str)
    
    assert data.shape[0] > 0, "data file must have at least one row"

    try:
        y = data[:,y_col].astype(float)

    except ValueError:
        assert False, "y_col must be a numeric column"

    data = np.delete(data, y_col, axis = 1)
    
    n,p = data.shape
    assert p > 0, "predictors must have at least one column"

    k = 0 
    col_struct = deque([])
    # check if columns are numeric or categorical
    # and store the future column indices
    for j in range(p):
        try:
            data[0,j].astype(float)
            col_struct += [k]
            k += 1

        except ValueError:
            col_span = len(set(data[:,j]))
            col_struct += [ list(range(k, k + col_span)) ]
            k += col_span

    # col_struct
    X = np.zeros((n, k))

    float_cols = deque([])
    cat_cols = deque([])
    j = 0

    for col in col_struct:
        if isinstance(col, int):
            X[:,col] = data[:,j].astype(float)
            float_cols += [col]

        else:
            X[:,col] = one_hot(data[:,j])
            cat_cols += [col]

        j += 1

    # verbose = True
    if verbose:
        print("X shape: ", X.shape)
        print("numeric columns: ", list(float_cols))
        print("categorical columns: ", list(cat_cols))

    return X, y

def linTime_intersection(names_vcv, spps_ff):
    # names_vcv, spps_ff

    # O(n)
    a_dict = {spps_ff[i] : i for i in range(len(spps_ff))}
    # O(n)
    b_dict = {names_vcv[i] : i for i in range(len(names_vcv))}

    out = deque([])
    # O(n)
    for k in b_dict.keys():
        if not k in a_dict:
            continue

        tmp_idx = a_dict[k]
        out.append(tmp_idx)

    return list(out)


def intersect_w_spps(names_vcv, spps_file, X, y, delimiter = ',', verbose = False):
    """
    intersect species names from the data file and the tree file

    Parameters
    ----------
    names_vcv : numpy array
        species names from the tree file

    spps_file : str
        path to the species names file

    X : numpy array
        predictors

    y : numpy array
        response variable

    delimiter : str
        delimiter of the species names file

    verbose : bool
        print information about the species names

    Returns
    -------
    X : numpy array
        ordered predictors

    y : numpy array
        ordered response variable
    """

    # species names from file
    spps_ff = np.loadtxt(spps_file, delimiter=delimiter, dtype=str)

    assert len(spps_ff) == len(set(spps_ff)), "species names must be unique"
    assert len(spps_ff) == X.shape[0], "the number of species names must match the number of rows in the data file"


    # match spps_ff and names
    iter_indx = linTime_intersection(names_vcv, spps_ff)
    assert len(iter_indx) > 0, "species names must match the tree and data file"

    # verbose = True
    if verbose:
        print(f"{len(spps_ff)} names from data, {len(names_vcv)} from tree, {len(iter_indx)} intersect")

    X = X[iter_indx,:]
    y = y[iter_indx]

    return X, y

def read_data(tree_file, data_file, data_file_spps, y_col = -1, delimiter = ',', verbose = False):
    """
    read data file and tree file.
    If there are categorical variables in X, they are one-hot encoded.
    If the response variable is not numeric, an assertion error is raised.
    To see where the categorical variables are, set verbose to True.


    Parameters
    ----------
    tree_file : str
        path to the tree file

    data_file : str
        path to the data file. This data file
        does not contain column names or row names

    data_file_spps : str
        path to the species names file. This data file
        contains only the species names. This is used to order
        the data_file based on the species names in the rows 
        of the tree_file's VCV matrix.

    y_col : int
        column index of the response variable

    delimiter : str
        delimiter of the data file

    verbose : bool
        print information about the data

    Returns
    -------
    X : numpy array
        predictors
    
    y : numpy array
        response variable

    vcv : numpy array
        variance-covariance matrix
    """

    with open(tree_file, 'r') as f:
        tree = f.read().strip()

    vcv, names = get_vcv(tree)

    names_vcv = np.array(names)
    vcv = np.array(vcv)

    X, y = get_Xy(data_file,
                  verbose = verbose, 
                  y_col=y_col, 
                  delimiter=delimiter)
    
    
    X, y = intersect_w_spps(names_vcv, data_file_spps, X, y, 
                            delimiter=delimiter,
                            verbose = verbose)
    return X, y, vcv

# Internal original file location: get_Xy_cov.py from LS code base

# tree_file = "./data/test_tree.txt"
# data_file_spps = "./data/test_data_spps.txt"
# from phylokrr.datasets import load_1d_data_example
# from phylokrr.utils import weight_data, P_inv_simple

# with open(tree_file, 'r') as f:
#     tree = f.read().strip()
#     vcv, names = get_vcv(tree)  

# vcv = np.array(vcv)
# with open(data_file_spps, 'w') as f:
#     f.write('\n'.join(names))


# np.random.seed(12037)
# mean_vector = np.zeros(vcv.shape[0])

# X_w = np.random.normal(0, 1, vcv.shape[0])
# # Non-linear response variable (sine curve)
# y_w = np.sin(X_w*1.5).ravel() 

# # Add noise to the response variable
# y_w[::10] += 4 * (0.5 - np.random.rand(X_w.shape[0] // 10))

# # we can attempt to unweight to the original space
# # with the square root of the covariance matrix
# P_inv = P_inv_simple(vcv)
# X_uw_uc, y_uw_uc = P_inv @ X_w, P_inv @ y_w

# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# axs[0].scatter(X_uw_uc, y_uw_uc, color = 'red', alpha=0.5, )
# axs[0].set_title('Unweighted data')
# axs[0].set_xlabel('x')
# axs[0].set_ylabel('y')
# axs[1].scatter(X_w, y_w, color = 'blue', alpha=0.5, )
# axs[1].set_title('Phylogenetically weighted data')
# axs[1].set_xlabel('$x^*$')
# axs[1].set_ylabel('$y^*$')
# plt.tight_layout()



# # Xy = np.stack([X_uw_uc, y_uw_uc], axis=1)
# # np.savetxt('../src/data/test_data3.csv', Xy, delimiter=',')


# data_file = "./data/test_data3.csv"
# X_uw_uc0, y_uw_uc0, vcv0 = read_data(tree_file, data_file, data_file_spps,
#                                  y_col = 1, delimiter = ',', verbose = True)

# X_w0, y_w0 = weight_data(X_uw_uc0, y_uw_uc0, vcv0, use_sd = False)

# y_uw_uc[-1]


# np.random.seed(12038)

# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# axs[0].scatter(X_uw_uc0, y_uw_uc0, color = 'red', alpha=0.5, )
# axs[0].set_title('Unweighted data')
# axs[0].set_xlabel('x')
# axs[0].set_ylabel('y')
# axs[1].scatter(X_w0, y_w0, color = 'blue', alpha=0.5, )
# axs[1].set_title('Phylogenetically weighted data')
# axs[1].set_xlabel('$x^*$')
# axs[1].set_ylabel('$y^*$')
# plt.tight_layout()
