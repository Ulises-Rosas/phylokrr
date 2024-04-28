import numpy as np
import pkg_resources


def load_vcv():
    vcv_file = pkg_resources.resource_stream('phylokrr', 'data/test_cov2.csv')
    vcv = np.loadtxt(vcv_file, delimiter=',')
    return vcv

