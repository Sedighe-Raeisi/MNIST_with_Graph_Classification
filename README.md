# MNIST with Graph Classification
Mnist images classification with graph classification in pytorch
In this repository I show how to convert an image to a graph and then classify images with graph convolutional neural networks of pytorch. 
I try to simlify all parts as this repository is a guide for learning the graph classification task. 
It consists of two parts, 
  1. in the first part I define a function to convert an image to a graph
  2. in the second part I use a pree-defined command of sklearn to convert an image to it's adjacency matrix as a nympy array. 
I make the model for each of these methods and evaluate the models. 
## Method 1: Defining a function to convert an image to a graph. 
Here we consider each pixel as a node. Each node is connected to it's most close neighbors. If the node/ pixel is specified by i and j as it's row and column position, then it's most close neighbors would be :
  * i-1, j-1
  * i-1, j
  * i-1, j+1
  * i, j-1
  * i, j+1
  * i+1, j-1
  * i+1, j
  * i+1, j+1
we should define edge indexes as to connect each node/pixel to these neighbors. 
Note that although we spesify each pixel with two indices (row and column) but we specify each node with one index. To convert two indices two one we can 
node_index=i*28+j


