print(__doc__)

from time import time
import numpy as np
from numpy import genfromtxt
import collections, numpy
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import Normalizer

from sklearn.decomposition import NMF

#read csv file in Numpy array
#usecols can be used to get specific columns:  usecols = 0, 1, 3,9
def read_csv(filelName):
    data = np.genfromtxt(fileName, delimiter=',', usecols = range(0,16), skip_header=1, dtype ='unicode')
    return data

#get frame number form first column 
def get_frame_number(data, n_rows):
    for i in range(0, n_rows):
        firstCol = data[i,0]
        frameNames=firstCol.split('/')
        frameId = frameNames[2].split('_')
        data[i,0] = frameId[1]
    return data

#sort Numpy array by first column
def sort_by_column(data):
    sortedArr = data[data[:,0].argsort()]
    return sortedArr
    




fileName = 'data/p1.csv'
#read data file
data = read_csv(fileName)

fileName = 'data/p2.csv'
#read data file
data2 = read_csv(fileName)


n_rows, n_columns = data.shape
#get frame id from first column
posedata_frame = get_frame_number(data, n_rows)
#sort posedata by first column
sortedPose = sort_by_column(posedata_frame)

#create datapoints 
j = 1
for i in range (0,n_rows):
    
    for k in range(0, 100):
        #successive elements
        if (sortedPose[j,0] == sortedPose[i,0]+1):
            array[i] = sortedPose[i,0]
            array[j] = sortedPose[j,0]
            i = i+1
            j = j+1
         #not successive elements, go next and reset    
         else:
            i = i+1
            j = j+1
            array.empty()
    
  



















