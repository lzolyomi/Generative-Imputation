"""Utility functions for GAIN.
(3) rounding: Handlecategorical variables after imputation
(6) binary_sampler: sample binary random variables
(7) uniform_sampler: sample uniform random variables
"""
# imports
import numpy as np
import pandas as pd
import tensorflow as tf


def rounding(imputed_data, data_x):
    """Round imputed data for categorical variables.
  
  Args:
    - imputed_data: imputed data
    - data_x: original data with missing values
    
  Returns:
    - rounded_data: rounded imputed data
  """

    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])

    return rounded_data


def binary_sampler(p, rows, cols):
    """Sample binary random variables.
  
  Args:
    - p: probability of 1
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - binary_random_matrix: generated binary random matrix.
  """
    unif_random_matrix = np.random.uniform(0.0, 1.0, size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
    """Sample uniform random variables.
  
  Args:
    - low: low limit
    - high: high limit
    - rows: the number of rows
    - cols: the number of columns
    
  Returns:
    - uniform_random_matrix: generated uniform random matrix.
  """
    return np.random.uniform(low, high, size=[rows, cols])


def deprecated_create_ohe_separation(array, batch_size):
    """Receives the original array as input,
    detects the One Hot Encoded columns and creates C matrix to be used for 
    separating OHE values in loss calculation
    NOT USED ANYMORE.
    Args:
      array (np.array): original array without any modification
  """
    df = pd.DataFrame(array)  # convert array to pd dataframe
    binary_cols = []
    for c in df.columns:
        minimum = df.describe().loc["min"][c]
        maximum = df.describe().loc["max"][c]
        col_range = maximum - minimum
        nr_unique = len(df[c].unique())
        # check if max value=1, only 2 unique values and range=1
        # if all fulfilled column marked as one hot encoded (or binary at least)
        binary_checks = all([maximum == 1, nr_unique == 2, col_range == 1.0])
        binary_cols.append(binary_checks)
    c_temp = np.array(binary_cols).astype(int)  # convert to binary np array
    C = np.repeat(c_temp[None, :], batch_size, axis=0)
    # return col x batch_size matrix that has 1s where OHE columns present
    return C


def create_C_total(ohe_indices, row, col):
    """Creates matrix C: 1s for OHE columns, 0s for continuous
    ONLY used to separate OHE and continuous features

    Args:
      ohe_indices (list): list of pairs for begin:end of a OHE feature's column
      row (int): number of rows in matrix 
      col (int): number of total columns in matrix (as ohe_indices only has OHE feature indices)
    """
    if ohe_indices:  # if there are OHE features present
        C_row = np.zeros((1, col))  # initialize one row of matrix
        for begin, end in ohe_indices:  # begin, end two indices for a OHE feature
            length = end - begin
            assert length > 0  # cannot be negative
            C_row[:, begin:end] = np.ones((1, length))  # replace range of col with 1s
        C_complete = np.repeat(C_row, row, axis=0)
    else:  # if no OHE features in data return zero matrix
        C_complete = np.zeros((row, col))
    return C_complete


def create_C_i_per_feature(range: tuple, row, col):
    """Create matrix C used to separate OHE features
    from each other and continuous features

  Args:
      range (tuple): range of feature's OHE columns eg.:(16,31)
      row (int): rows in returned matrix
      col (int): total nr columns in data
  Return:
    Matrix C
  """
    length = range[1] - range[0]
    C_row = np.zeros((1, col))
    assert length > 0
    C_row[:, range[0] : range[1]] = np.ones((1, length))
    C_complete = np.repeat(C_row, row, axis=0)
    return C_complete

