# Packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import torchvision
import imp
import matplotlib as mpl
from matplotlib import cm
import statsmodels.tsa.stattools as stat
from sklearn.model_selection import KFold
import time
from tqdm import tqdm
from sklearn import datasets, preprocessing


def import_covertype(c1=1, c2=2, n_data = 500000):
    dataset = datasets.fetch_covtype()
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(dataset.data)
    x_train = X[0:n_data,:]
    x_test = X[n_data:,:]
    y_train = dataset.target[0:n_data]
    y_test = dataset.target[n_data:]
    return preprocessing_covertype(x_train,x_test,y_train,y_test,c1,c2)

# Preprocessing
def preprocessing_covertype(x_train,x_test,y_train,y_test,c1,c2):

    # Making sure that the values are float so that we can get decimal points 
    # after division
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # Select the classes to classify for binary logistic regression
    i1 = np.where((y_train == c1) | (y_train == c2))
    y1_train = y_train[i1]
    x1_train = x_train[i1]
    i1 = np.where((y_test == c1) | (y_test == c2))
    y1_test = y_test[i1]
    x1_test = x_test[i1]

    # Replace labels of the current class with 1 and other labels with 0
    i1 = np.where(y1_train == c1)
    i2 = np.where(y1_train == c2)
    y1_train[i1] = 1
    y1_train[i2] = 0
    i1 = np.where(y1_test == c1)
    i2 = np.where(y1_test == c2)
    y1_test[i1] = 1
    y1_test[i2] = 0
    
    # Return objects of interest
    return x1_train, x1_test, y1_train, y1_test;
    
# Import data

"""
w, v = np.linalg.eig(np.dot(X.T,X))
w = np.max(w.real)
M = w * (1/4) + tau 
gamma = 0.01/M

# Initialization
d = np.size(X,1)
n = np.size(y)
tau = 1 # regularization parameter
"""
