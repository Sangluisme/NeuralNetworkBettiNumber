# NeuralNetwrokBettiNumber
demo code for paper [Dive into Layers: Neural Network Capacity Bounding using Algebraic Geometry](https://arxiv.org/abs/2109.01461)

## Introduction
This code build a simple **3 layers fully connected network** with same node for each layer, **polynomial** as activation function and calculate the Betti number of each layer to demonstrate the conclusion in the paper [Dive into Layers: Neural Network Capacity Bounding using Algebraic Geometry](https://arxiv.org/abs/2109.01461), the code build the network with **different layer sizes (node numbers)** to train, then calculate the Betti number of output data of each layer to demonstrate the albegraic complexity of each layer. 

The code will save calculated presistence barcode graph(png) and Betti number files(txt).

The demo code use **mnist** as input data by defaul. 

## Usage
the main script you should run is in *src/main_polynomial_model_mnist.py*

Input arguments:
```
- --start_node int minimum node number

- --last_node int maximum node number

- --step int step size of node number increasing each time

- --savepath str the directory that save presistent barcodes and Betti number txt files
```

## Independences
-Tensorflow 2.5 (CPU is enough)
-[GUDHI](https://gudhi.inria.fr/)
-matplotlib
-numpy


