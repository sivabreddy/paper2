"""
Data loading module for prostate MRI analysis system.
Handles reading of features and labels from numpy arrays.
"""
import csv
import os
import numpy as np


def read_data():
    """
    Loads MRI feature data from numpy array file.
    
    Returns:
        numpy.ndarray: Preprocessed MRI features (64x64x3)
        with NaN values replaced by zeros
    """
    # #Read data from the csv file
    # file_name = "Feat.csv"                  #dataset location
    # datas = []
    # with open(file_name, 'rt')as f:
    #     content = csv.reader(f)                        #read csv content
    #     for rows in content:                           #row of data
    #         tem = []
    #         for cols in rows:                          #attributes in each row
    #             tem.append(float(cols))             #add value to temporary array
    #         datas.append(tem)                         #add 1 row of array value to dataset
    datas = np.load(os.path.join(os.path.dirname(__file__), 'Feat_fin.npy'))
    datas = np.nan_to_num(datas)
    return datas

def read_label():
    """
    Loads ground truth labels from numpy array file.
    
    Returns:
        numpy.ndarray: Binary classification labels
        (0 = benign, 1 = malignant)
    """
    # #Read data from the csv file
    # file_name = "Label.csv"                  #dataset location
    # datas = []
    # with open(file_name, 'rt')as f:
    #     content = csv.reader(f)                        #read csv content
    #     for rows in content:                           #row of data
    #         for cols in rows:                          #attributes in each row
    #             datas.append(int(float(cols)))  # add value to temporary array
    datas = np.load(os.path.join(os.path.dirname(__file__), 'lab_fin.npy'))
    return datas
