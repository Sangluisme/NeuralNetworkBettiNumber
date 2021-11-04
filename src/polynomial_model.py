from numpy.core.numeric import Infinity
import tensorflow as tf
from tensorflow.python.ops.gen_math_ops import minimum
#from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.compat.v1 as tf
# import input_data
import help_function

import pickle as pickle
import gudhi as gd  
#from pylab import *
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy import linalg as LA
import plot_persistence_barcode



class Network:
    def __init__(self, node):
        tf.disable_v2_behavior()
        self.learning_rate = 0.0005

        self.x = tf.placeholder(tf.float32, [None, 2])
        self.label = tf.placeholder(tf.float32, [None, 2])
        # layer 1
        self.w_layer1 = tf.Variable(tf.random.normal(
            shape=[2, node], dtype=tf.float32))

        self.b_layer1 = tf.Variable(
            tf.random.normal(shape=[node], dtype=tf.float32))

        self.y_layer1 = tf.matmul(self.x, self.w_layer1) + self.b_layer1
        self.y_layer1 = tf.layers.batch_normalization(
            self.y_layer1, training=True)

        self.y_layer1 = tf.pow(self.y_layer1, [2])

        # self.y_layer1 = tf.nn.relu(
        # self.y_layer1)

        self.y_layer1 = tf.layers.batch_normalization(
            self.y_layer1, training=True)

        # layer 2
        self.w_layer2 = tf.Variable(tf.random.normal(
            shape=[node, node], dtype=tf.float32))
        self.b_layer2 = tf.Variable(
            tf.random.normal(shape=[node], dtype=tf.float32))

        self.y_layer2 = tf.matmul(self.y_layer1, self.w_layer2) + self.b_layer2
        self.y_layer2 = tf.layers.batch_normalization(
            self.y_layer2, training=True)

        self.y_layer2 = tf.pow(self.y_layer2, [2])
        # self.y_layer2 = tf.nn.relu(
        #     self.y_layer2)

        self.y_layer2 = tf.layers.batch_normalization(
            self.y_layer2, training=True)

        # layer 3
        self.w_layer3 = tf.Variable(tf.random.normal(
            shape=[node, node], dtype=tf.float32))
        self.b_layer3 = tf.Variable(
            tf.random.normal(shape=[node], dtype=tf.float32))

        self.y_layer3 = tf.matmul(self.y_layer2, self.w_layer3) + self.b_layer3
        self.y_layer3 = tf.layers.batch_normalization(
            self.y_layer3, training=True)
        self.y_layer3 = tf.pow(self.y_layer3, [2])

        # self.y_layer3 = tf.nn.relu(
        #     self.y_layer3)

        self.y_layer3 = tf.layers.batch_normalization(
            self.y_layer3, training=True)


        # layer output
        self.w_out = tf.Variable(tf.random.normal(
            shape=[node, 2], dtype=tf.float32))
        self.b_out = tf.Variable(
            tf.random.normal(shape=[2], dtype=tf.float32))

        self.y = tf.nn.softmax(
            tf.matmul(self.y_layer3, self.w_out) + self.b_out)

        self.loss = -tf.reduce_sum(self.label * tf.log(self.y + 1e-10))

        self.train = tf.train.GradientDescentOptimizer(
            self.learning_rate).minimize(self.loss)

        predict = tf.equal(tf.argmax(self.label, 1), tf.argmax(self.y, 1))

        self.accuracy = tf.reduce_mean(tf.cast(predict, "float"))

    
    def __del__(self):
        print("delete network.")


class Train:
    def __init__(self, data, node):
        self.net = Network(node)
        self.node = node

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    
        # self.data = input_data.read_data_sets('../data_set', one_hot=True)
        self.data = data

    def __del__(self):
        print("delete train.")

    def train(self):
        batch_size = 64
        train_step = 2000
        for batch in range(30):
            for i in range(train_step):
                # x, label = self.data.train.next_batch(batch_size)
                x, label = self.data.next_batch(batch_size)
                _, loss = self.sess.run([self.net.train, self.net.loss],
                                        feed_dict={self.net.x: x, self.net.label: label})
                # if (i + 1) % 10 == 0:
                #     print('第%5d步，当前loss：%.2f' % (i + 1, loss))
                # if(i == (2000-1) and batch == 0):
                #     layer1, accuracy = self.sess.run([self.net.y_layer1, self.net.accuracy],feed_dict={self.net.x: x, self.net.label: label})
                #     layer2, accuracy = self.sess.run([self.net.y_layer2, self.net.accuracy],feed_dict={self.net.x: x, self.net.label: label})
                #     n = LA.norm(layer1[0,:])
                #     self.calculate_betti_numer(layer1, 1, n)
                #     self.calculate_betti_numer(layer2, 2, n)
            self.calculate_accuracy()

    def calculate_accuracy(self):
        test_x = self.data.test_x
        test_label = self.data.test_label
        accuracy = self.sess.run(self.net.accuracy,
                                 feed_dict={self.net.x: test_x, self.net.label: test_label})
        print("Model: accuracy: %.2f，%d images are tested. " % (accuracy, len(test_label)))

        return accuracy



    def calculate_layer_Betti_number(self, class_num):
        # class_index = np.where(self.data.test.labels[:, class_num] == 1)[0]
        # total_num = len(class_index)
        # # we calculate 10% of one class input
        # input_num = np.floor(total_num/10).astype(int)

        # test_x = self.data.test.images[class_index,:]
        # test_x =  test_x[0:input_num,:]
        # test_label  = self.data.test.labels[class_index,:]
        # test_label = test_label[0:input_num,:]
        test_x, test_label = get_input_data(self.data, class_num)

        layer1, accuracy = self.sess.run([self.net.y_layer1, self.net.accuracy],
                                 feed_dict={self.net.x: test_x, self.net.label: test_label})
        layer2, accuracy = self.sess.run([self.net.y_layer2, self.net.accuracy],
                                 feed_dict={self.net.x: test_x, self.net.label: test_label})
        layer3, accuracy = self.sess.run([self.net.y_layer3, self.net.accuracy],
                                 feed_dict={self.net.x: test_x, self.net.label: test_label})
        n = LA.norm(test_x[0,:])

        # self.calculate_betti_numer(test_x, class_num, n)
        _, Betti1 = self.calculate_betti_numer(layer1, class_num, 1, n)
        _, Betti2 = self.calculate_betti_numer(layer2, class_num, 2, n)
        _, Betti3 = self.calculate_betti_numer(layer3, class_num, 3, n)

        return Betti1, Betti2, Betti3


        
    def calculate_betti_numer(self, data, class_num, layer,  max_length):
        # n = LA.norm(data[0,:])
        color = ['red', 'black','black']

        BarCodes_Rips0, Betti1, Betti0 = get_Betti_number(data, max_length)

        file_name = '../data_2_disk/BarCodes_class%d_layer%d_node%d.png' % (class_num, layer, self.node)
        axs = gd.plot_persistence_barcode(BarCodes_Rips0, alpha =  1.0, legend = True, colormap = color)
        plt.savefig(file_name)

        plt.cla()

        # file_name = '../data/Digram_class%dlayer%d.png' % (class_num, layer)  
        # axs = gd.plot_persistence_diagram(BarCodes_Rips0, legend = True)
        # plt.savefig(file_name)

        # plt.cla()

        return Betti1, Betti0

def get_input_data(data, class_num):
    class_index = np.where(data.test_label[:, class_num] == 1)[0]
    total_num = len(class_index)
    # we calculate 10% of one class input
    input_num = np.floor(total_num/5).astype(int)
    if(input_num > 110):
        input_num = 110

    index = np.random.randint(total_num, size=input_num)

    test_x = data.test_x[class_index,:]
    test_x =  test_x[index,:]
    test_label  = data.test_label[class_index,:]
    test_label = test_label[index,:]
    print("Betti number: %d images from class %d are calculated." % (input_num, class_num))

    return test_x, test_label

def get_Betti_number(data, max_length):
    # color = ['red', 'black','black']

    skeleton_D = gd.RipsComplex(points = data.tolist(), max_edge_length = 2.0*max_length)
    Rips_simplex_tree_D = skeleton_D.create_simplex_tree(max_dimension = 2)
    BarCodes_Rips0 = Rips_simplex_tree_D.persistence()

    Barcode = plot_persistence_barcode._array_handler(BarCodes_Rips0)
    (min_birth, max_death) = plot_persistence_barcode.__min_birth_max_death(BarCodes_Rips0)
    delta = (max_death - min_birth) * 0.1
    # Replace infinity values with max_death + delta for bar code to be more
    # readable
    infinity = max_death + delta
    axis_start = min_birth - delta

    Barcode = sorted(
        Barcode,
        key=lambda life_time: life_time[1][1] - life_time[1][0], reverse = True
    )[:1000]
    Barcode = sorted(Barcode, key=lambda birth: birth[1][0], reverse=True)
    Barcode_array = np.asarray(Barcode)

    index = np.where(Barcode_array[:,0] == 1)[0]
    length = len(index)
    if (length > 0 and length < Barcode_array.shape[0]):
        minimum_Betti1 = Barcode_array[length-1, 1][0]
        for i in range(length, Barcode_array.shape[0]):
            if (Barcode_array[i,1][1] < minimum_Betti1):
                Betti0 = i
                break
    else:
        minimum_Betti1 = np.Infinity
        # Betti0 = Barcode_array.shape[0]
        Betti0 = 0

    return BarCodes_Rips0, Betti0, length


if __name__ == "__main__":

    # data = input_data.read_data_sets('../data_set', one_hot=True)
    data = help_function.Data('../data/D1.npz')
    # Betti_input = []
    # Betti0_input = []
    # for class_num in range(2):
    #     x,_ = get_input_data(data, class_num)
    #     n = LA.norm(x[0,:])
    #     Barcode, Betti_, Betti0_ = get_Betti_number(x, n)
    #     file_name = '../data_2_disk/BarCodes_input_class%d.png' % (class_num)
    #     axs = gd.plot_persistence_barcode(Barcode, alpha =  1.0, legend = True, colormap = ['red','black','blue'])
    #     plt.savefig(file_name)

    #     plt.cla()
    #     print("Data: the input data has Betti 0 %d. " % Betti0_)
    #     print("Data: the input data has Betti 1 %d. " % Betti_)
    #     Betti_input.append(Betti_)
    #     Betti0_input.append(Betti0_)
    # Betti_input = np.asarray(Betti_input)
    # Betti0_input = np.asarray(Betti0_input)
    # Betti_input = np.vstack([Betti_input, Betti0_input])
    
    # np.savetxt("../data_2_disk/Betti_input.txt", Betti_input.T)

    b2 = np.zeros([7,10])
    b1 = np.zeros([7,10])
    b3 = np.zeros([7,10])
    Acc = np.zeros([7])
    count = 0
    for node in range(2, 8, 1):
        app = Train(data, node)
        app.train()
        Acc[count] = app.calculate_accuracy() 
        Betti1 = np.zeros([10])
        Betti2 = np.zeros([10])
        Betti3 = np.zeros([10])
    
        for class_num in range(2):

            b10, b20, b30 = app.calculate_layer_Betti_number(class_num)
            Betti1[class_num] = b10
            Betti2[class_num] = b20
            Betti3[class_num] = b30

        b1[count,:] = Betti1
        b2[count,:] = Betti2
        b3[count,:] = Betti3
        count = count+1

    filename = "../data_ply/Betti_layer1_binary.txt"
    np.savetxt(filename, b1)

    filename = "../data_ply/Betti_layer2_binary.txt"
    np.savetxt(filename, b2)

    filename = "../data_ply/Betti_layer3_binary.txt"
    np.savetxt(filename, b3)

    filename = "../data_ply/Betti_accurarcy_binary.txt"
    np.savetxt(filename, Acc.T)
    # class_num = 7
    # node = 127

    # app = Train(node, data)
    # app.train()
    # acc = app.calculate_accuracy()
    # Betti1_, Betti0 = app.calculate_layer_Betti_number(class_num)
    # print("Data: the input data has Betti 0 %d. " % Betti0)
    # print("Data: the input data has Betti 1 %d. " % Betti1_)
    # print("Data: the input data has accuracy %f. " % acc)