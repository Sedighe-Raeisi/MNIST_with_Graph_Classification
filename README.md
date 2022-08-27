# MNIST with Graph Classification
Mnist images classification with graph classification in PyTorch
In this repository, I show how to convert an image to a graph and then classify images with graph convolutional neural networks of PyTorch. 
I try to simplify all parts as this repository is a guide for learning the graph classification task. 
It consists of two parts, 
  1. in the first part I define a function to convert an image to a graph
  2. in the second part I use a pre-defined command of sklearn to convert an image to its adjacency matrix as a NumPy array. 
I make the model for each of these methods and evaluate the models. 
## Method 1: Defining a function to convert an image to a graph. ACC=0.97
![image](https://user-images.githubusercontent.com/67642255/187019740-02f683bb-0808-47c8-85df-02a492620c92.png)

Here we consider each pixel as a node. Each node is connected to its most close neighbors. If the node/ pixel is specified by i and j as its row and column position, then its most close neighbors would be :
  * i-1, j-1
  * i-1, j
  * i-1, j+1
  * i, j-1
  * i, j+1
  * i+1, j-1
  * i+1, j
  * i+1, j+1   
we should define edge indexes to connect each node/pixel to these neighbors. 
Note that although we specify each pixel with two indices (row and column) but we specify each node with one index. To convert two indices two one we can    
node_index=i x 28+j   
Note that if you draw the graph, the image pixels are read from the upper left to the lower right. So define the position of nodes to place them in an appropriate position. Otherwise, your graph would exhibit a rotated image. 
Then we define a function that receives an image as input then we iterate over all nodes and at the same time we prepare two lists, 
  - list of node features for all nodes, it should be a list of lists
  - list of edge_indexes , it consists of two list source nodes and destination nodes of all edges. For more details, you can 
  - we didn't consider the weight of edges, but if you do, you should have a list of weights too. 
I load my dataset and split it to train and test. 
Then we should make a data loader for train and a data loader for test data. 
We iterate on the train data. In this loop, we use our function and get node features and edge_indices for the graph of each image. 
As we want to use PyTorch for our model we should convert these lists to an appropriate torch tensor. 
Node features should be float tensor and edge_index should be torch tensor with torch.long as dtype.  
If you don't convert these, you would receive an error when running the model. 
Then we define the data for each image using "from torch_geometric.data import Data". 
**Note, there is a command that converts a networkx graph to Data, but it is better not to use it. As the edge_indexes and other parts of networks graph may have a non-tensor type. When converting the networkx graph to Data, the Data would have the wrong type of edge indexes or node features. You won't realize until using the model.**
Then we make a list of Data for each image. (data_list) when making the Data, we specify the label of the graph, y.
Then we use DataLoader to convert this data_list to a data_loader. We make a data loader for test data. 
### Model.
I discussed the model in the repository of the karate club.  
There is one difference, we have batch here, which we didn't use in the node classification of the karate club. 
The batch is defined when defining the data. We put two GCN layers at the beginning of the model. Before using the linear dense layers we should convert the size of the tensor.That is why we use "x.view(batch_size, -1)".
In the method of __init__ we should specify the input size of the first linear layer. It should be (hidden_channel x 784). 
The hidden_channel is a number of neurons of GCN layer. The 784 is a number of nodes for each graph. Then we would have a batch of tensors to feed in the first linear layer. 
I used the tensorflow for computing the accuracy. You should consider the batch size as the train_size/batch_size or test_size/batch_size has zero divide remaining. Otherwise, you may encounter errors. 
## Method 2: SKlearn image to a graph. ACC=0.97
![image](https://user-images.githubusercontent.com/67642255/187019770-afcfe554-5fa4-475f-848a-48df57e085ec.png)

We use "img_to_graph" to reach the adjacency matrix of the weighted graph of each image. Then we use  networkx ".from_numpy_array" command to get the networkx graph from the adjacency matrix. 
Then we use "from_networkx" to convert the graph to data. But we won't use this data for data loader as explained in the previous section. We use this data to get a list of edge indexes and edge weights. Then we define the Data using the tensor format of these lists. 
The other parts are the same as before but we should consider data.weight as the model's input in the forward method. 


