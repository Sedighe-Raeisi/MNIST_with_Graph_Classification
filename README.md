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
Then we define a function that recieve an image as input then we iterate aver all nodes and at the same time we prepare two list, 
  - list of node features for all nodes, it should be list of lists
  - list of edge_indexes , it consists of two list source nodes and destination nodes of all edges. For more details you can 
  - we didn't consider weight of edges, but if you do, you should have a list of weights too. 
I load my dataset and split it to train and test. 
Then we should make a dataloader for train and a dataloader for test data. 
We iterate on the train data. In this loop we use our function and get node features and edge_indices for the graph of each image. 
As we want to use pytorch for our model we should convert these list to an appropriate torch tensor. 
Node features should be float tensor and edge_index should be torch tensor with torch.long as dtype.  
If you don't convert these, you would recieve error when running the model. 
Then we define the data for each image using "from torch_geometric.data import Data". 
**Note, there is a command that convert a networkx graph to Data, but it is better not to use it. As the edge_indexes and other part of networks graph may have a non-tensor type. When converting the networkx graph to Data, the Data would have wrong type of edge indexes or node features. You wont realize untill using the model.**
Then we make a list of Data of each image. (data_list) when making the Data, we specify label of the graph, y.
Then we use DataLoader to convert this data_list to a data_loader. We make a data loader for test data. 
### Model.
I disscussed about the model in the repository of karate club.  
There is one difference, that we have batch here, we didn't use in the node classification of karate club. 
The batch is defined wehen defining the data. We put two GCN layer at the beginning of the model. Before using the linear dense layers we should convert the size of the tensor.That is why we use "x.view(batch_size, -1)".
In the method of __init__ we should specify the input size of the first linear layer. It should be hidden_channel*784. 
The hidden_channel is number of neurons of GCN layer. The 784 is number of nodes for each graph. Then we would have a bach of tensors to feed in the first linear layer. 
I used the tensorflow for computing the accuracy. You should consider the batch size as the train_size/batch_size or test_size/batch_size have zero devide remaining. Other wise you may encounter error. 
## Method 2: SKlearn image to a graph. 
We use "img_to_graph" to reach the adjecency matrix of the weighted graph of each image. Then we use  networkx ".from_numpy_array" command to get the networkx graph from adjacency matrix. 
The we use "from_networkx" to convert the graph to data. But we wont use this data for dataloader as explained in previous section. We use this data to get list of edge indexes and edge weights. Then we define the Data using the tensor format of these lists. 
The other parts are the same as before but we should consider data.weight as the input of model in the forward method. 


