from numpy import linalg as LA
# import sys

import numpy as np
# import pickle as pickle
# import gudhi as gd  
from pylab import *
import matplotlib.pyplot as plt


class Data:
    def __init__(self, filename):
        data = np.load(filename)
        self.x = data['x_train']
        self.label = data['x_label']
        self.test_x = data['x_test']
        self.test_label = data['label_test']

        self.total_num = self.x.shape[0]

    def next_batch(self, batch_size):
        index = np.random.randint(self.total_num, size = batch_size)
        x_ = self.x[index,:]
        label_ = self.label[index,:]
        
        return x_, label_


# def input_data(filename):
#     data = np.load(filename)
#     x_train = data['x_train']
#     x_label = data['x_label']

#     x_test = data['x_test']
#     label_test = data['label_test']

#     data_ = Data(x_train, x_label, x_test, label_test)

#     return data_

