# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 20:46:56 2020

@author: nitin
"""

# Create non-linearly separable data

import numpy as np
import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, log_loss


from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs


import torch
import warnings
import os
import logging

import torch.nn.functional as F
warnings.filterwarnings('ignore')

# Todo: data loading should be done in another class
# Todo: Make each model into a class and pass this into higher NNTrainer class





class NeuralNetworkCluster:
    """
    Holds several neural networks in a dictionary.
    Each NN corresponds to a node in the distributed algorithm.
    """
    def __init__(self):
        from collections import defaultdict
        self.neuralNetDict = defaultdict(dict)
        
        # This will store feature indices for each node - determined by overlap functionality
        self.featureDict = {1: []}
    
    def appendNNToCluster(self, nn_config):
        node_id = nn_config["node_id"]
        if node_id in self.neuralNetDict.keys():
            logging.info("node_id: {} already exists in dictionary. Overwriting...".format(node_id))
        
        dataset = nn_config["dataset_name"]
        base_dir = "C:/Users/nitin/eclipse-workspace/consensus-deep-learning-version-2.0/data/"
        feature_split = nn_config["feature_split"]
        
        if nn_config["model_type"] == "1-layer-nn":
            model = SingleLayerNeuralNetwork()
            model.load_data(dataset, base_dir, feature_split, node_id)
            model.initialize(50)
        if nn_config["model_type"] == "2-layer-nn":
            model = TwoLayerNeuralNetwork()
            model.load_data(dataset, base_dir, feature_split, node_id)
            model.initialize(50, 25)
        
        self.neuralNetDict[node_id]["model"] = model
        
        # Loss criterion
        criterion = torch.nn.CrossEntropyLoss()
        self.neuralNetDict[node_id]["criterion"] = criterion
        
        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr = nn_config["learning_rate"])
        self.neuralNetDict[node_id]["optimizer"] = optimizer
        
        self.neuralNetDict[node_id]["train_losses"] = []
        self.neuralNetDict[node_id]["test_losses"] = []
        
        
    def gossip(self, node_id, neighbor_node_id):
        """
        Performs gossip on two given node_ids.
        """
        model0 = self.neuralNetDict[node_id]["model"]
        model1 = self.neuralNetDict[neighbor_node_id]["model"]
        
        criterion0 = self.neuralNetDict[node_id]["criterion"]
        criterion1 = self.neuralNetDict[neighbor_node_id]["criterion"]
        
        optimizer0 = self.neuralNetDict[node_id]["optimizer"]
        optimizer1 = self.neuralNetDict[neighbor_node_id]["optimizer"]
        
        # Wipe gradients of both optimizers
        optimizer0.zero_grad()
        optimizer1.zero_grad()
        
        # Forward pass
        y_pred0 = model0(model0.X_train)
        y_pred1 = model1(model1.X_train)
        
        y_pred0_2 = y_pred0.clone()
        y_pred1_2 = y_pred1.clone()
        
        y_pred_mean0 = (y_pred0 + y_pred1)/2
        y_pred_mean1 = (y_pred0_2 + y_pred1_2)/2
        
        
        # Compute Loss
        loss0 = criterion0(y_pred_mean0.squeeze(), model0.y_train)
        loss1 = criterion1(y_pred_mean1.squeeze(), model1.y_train)
       
        # Backward pass
        loss0.backward(retain_graph=True)
        loss1.backward(retain_graph=True)
        optimizer0.step()
        optimizer1.step()
        print("Train Loss @ Node {}: {}, Train Loss @ Node {}: {}".format(node_id, 
              loss0.item(), neighbor_node_id, loss1.item()))
        
    
    
    def compute_losses(self):
        """
        Computes train and test losses for all the nodes.
        """
        for node_id in self.neuralNetDict.keys():
            
            model = self.neuralNetDict[node_id]["model"]
            criterion = self.neuralNetDict[node_id]["criterion"]        
            
            # Compute Train Loss
            y_pred_train = model(model.X_train)
            train_loss = criterion(y_pred_train.squeeze(), model.y_test) 
            self.neuralNetDict[node_id]["train_losses"].append(train_loss.item())
            
            # Compute Test Loss
            y_pred_test = model(model.X_test)
            test_loss = criterion(y_pred_test.squeeze(), model.y_test)        
            self.neuralNetDict[node_id]["test_losses"].append(test_loss.item())

    def load_data(self):
        # dataset - load the entire dataset into memory
        # 
        
        
    def train(self, node_id):
        """
        Used for training on only one node in centralized execution. 
        No gossip is performed here.
        """
        model0 = self.neuralNetDict[node_id]["model"]
        criterion0 = self.neuralNetDict[node_id]["criterion"]        
        optimizer0 = self.neuralNetDict[node_id]["optimizer"]        
        # Wipe gradients of both optimizer
        optimizer0.zero_grad()
        # Forward pass
        y_pred0 = model0(model0.X_train)        
        # Compute Loss
        loss0 = criterion0(y_pred0.squeeze(), model0.y_train)
        # Backward pass
        loss0.backward(retain_graph=True)
        # Update parameters
        optimizer0.step()
        
        

class SingleLayerNeuralNetwork(torch.nn.Module):
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def initialize(self, hidden_size):
        super(SingleLayerNeuralNetwork, self).__init__()
        self.input_size = self.X_train.shape[1]
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 2)
        self.sigmoid = torch.nn.Sigmoid()
    
    def load_data(self, dataset, base_dir, feature_split, node_id=None):
        if node_id is None:
            train_filename = "{}_{}.csv".format(dataset, "train_binary")
            test_filename = "{}_{}.csv".format(dataset, "test_binary")
        
        else:
            train_filename = "{}_{}_{}.csv".format(dataset, "train", node_id)
            test_filename = "{}_{}_{}.csv".format(dataset, "test", node_id)
        
        df_train = pd.read_csv(os.path.join(base_dir, 
                                      dataset, 
                                      "feature_split_" + str(feature_split),
                                     train_filename), header=None).to_numpy()
        
        
        df_test = pd.read_csv(os.path.join(base_dir, 
                                      dataset, 
                                      "feature_split_" + str(feature_split),
                                      test_filename), header=None).to_numpy()
        
        
        # Split into features and labels
        X_train = df_train[:,1:]
        y_train = df_train[:,0]
        
        X_test= df_test[:,1:]
        y_test = df_test[:,0]
        
        
        # Map to tensors so torch can use this data
        X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))
        
        # Convert features to float and labels to long
        self.X_train = X_train.float()
        self.y_train = y_train.long()
        self.X_test = X_test.float()
        self.y_test = y_test.long()
        
        
        print(X_train.shape, X_test.shape, 
              y_train.shape, y_test.shape)
        
        
    
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = output.exp()/output.exp().sum(-1).unsqueeze(-1)
        return output


class TwoLayerNeuralNetwork(torch.nn.Module):
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
    
    def initialize(self, hidden_size1, hidden_size2):
        super(TwoLayerNeuralNetwork, self).__init__()
        self.input_size = self.X_train.shape[1]
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size1, self.hidden_size2)
        self.fc3 = torch.nn.Linear(self.hidden_size2, 2)
        self.sigmoid = torch.nn.Sigmoid()
    
    def load_data(self, dataset, base_dir, feature_split, node_id=None):
        if node_id is None:
            train_filename = "{}_{}.csv".format(dataset, "train_binary")
            test_filename = "{}_{}.csv".format(dataset, "test_binary")
        
        else:
            train_filename = "{}_{}_{}.csv".format(dataset, "train", node_id)
            test_filename = "{}_{}_{}.csv".format(dataset, "test", node_id)
        
        df_train = pd.read_csv(os.path.join(base_dir, 
                                      dataset, 
                                      "feature_split_" + str(feature_split),
                                     train_filename), header=None).to_numpy()
        
        
        df_test = pd.read_csv(os.path.join(base_dir, 
                                      dataset, 
                                      "feature_split_" + str(feature_split),
                                      test_filename), header=None).to_numpy()
        
        
        # Split into features and labels
        X_train = df_train[:,1:]
        y_train = df_train[:,0]
        
        X_test= df_test[:,1:]
        y_test = df_test[:,0]
        
        
        # Map to tensors so torch can use this data
        X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))
        
        # Convert features to float and labels to long
        self.X_train = X_train.float()
        self.y_train = y_train.long()
        self.X_test = X_test.float()
        self.y_test = y_test.long()
        
        
        print(X_train.shape, X_test.shape, 
              y_train.shape, y_test.shape)
        
        
    
    def forward(self, x):
        hidden1 = self.fc1(x)
        relu1 = self.relu(hidden1)
        hidden2 = self.fc2(relu1)
        relu2 = self.relu(hidden2)
        output = self.fc3(relu2)
        output = output.exp()/output.exp().sum(-1).unsqueeze(-1)
        return output


def test_cluster():
    nn_cluster = NeuralNetworkCluster()    
    nn_config_0 = {"dataset_name": "arcene",
                   "node_id": 0,
                   "nn_type": "mlp",
                   "num_layers": 2,
                   "loss_function": "cross_entropy",
                   "activation_function": "relu",
                   "learning_rate": 0.1,
                   "feature_split": 1,
                   "run_type": "distributed",
                   "neighbor": 1}
    
    nn_config_1 = {"dataset_name": "arcene",
                   "node_id": 1,
                   "nn_type": "mlp",
                   "num_layers": 2,
                   "loss_function": "cross_entropy",
                   "activation_function": "relu",
                   "learning_rate": 0.1,
                   "feature_split": 1,
                   "run_type": "distributed",
                   "neighbor": 0}
    
    nn_cluster.appendNNToCluster(nn_config_0)
    nn_cluster.appendNNToCluster(nn_config_1)
    
    # Gossip many times
    for i in range(50):
        nn_cluster.gossip(0, 1)
    
    print(nn_cluster.neuralNetDict[0]["train_losses"])
    print(nn_cluster.neuralNetDict[1]["train_losses"])
        
    
    
        
if __name__ == "__main__":
    test_cluster()
#    node_id = 0
#    neighbor_node_id = 1
#    
#    (X_train0, X_test0, y_train0, y_test0) = load_data("arcene", "C:/Users/nitin/eclipse-workspace/consensus-deep-learning-version-2.0/data/", 1, node_id)
#    (X_train1, X_test1, y_train1, y_test1) = load_data("arcene", "C:/Users/nitin/eclipse-workspace/consensus-deep-learning-version-2.0/data/", 1, neighbor_node_id)
#    
#    
#    
#    model0 = Feedforward(X_train0.shape[1], 50)
#    model1 = Feedforward(X_train1.shape[1], 50)
#    
#    criterion0 = torch.nn.CrossEntropyLoss()
#    criterion1 = torch.nn.CrossEntropyLoss()
#    
#    optimizer0 = torch.optim.SGD(model0.parameters(), lr = 0.01)
#    optimizer1 = torch.optim.SGD(model1.parameters(), lr = 0.01)
#       
#    print(model0.eval())
#    print(model1.eval())
#    
#    y_pred0 = model0(X_train0)
#    y_pred1 = model1(X_train1)
#    
#    before_train0 = criterion0(y_pred0.squeeze(), y_train0)
#    before_train1 = criterion1(y_pred1.squeeze(), y_train1)
#    
#
##    y_pred_mean = (y_pred0 + y_pred1)/2
##    
##    y_pred_mean = (y_pred0 + y_pred1)/2
#    
#
#    print('Train loss 0 before training' , before_train0.item()) 
#    print('Train loss 1 before training' , before_train1.item())
#    
#    
#    model0.train()
#    model1.train()
#    epoch = 2000
#    for epoch in range(epoch):
#        optimizer0.zero_grad()
#        optimizer1.zero_grad()
#        # Forward pass
#        y_pred0 = model0(X_train0)
#        y_pred1 = model1(X_train1)
#        
#        
#        y_pred0_2 = y_pred0.clone()
#        y_pred1_2 = y_pred1.clone()
#        
#        y_pred_mean0 = (y_pred0 + y_pred1)/2
#        y_pred_mean1 = (y_pred0_2 + y_pred1_2)/2
#        
#        
#        # Compute Loss
#        loss0 = criterion0(y_pred_mean0.squeeze(), y_train0)
#        loss1 = criterion1(y_pred_mean1.squeeze(), y_train1)
#       
#        # Backward pass
#        loss0.backward(retain_graph=True)
#        loss1.backward(retain_graph=True)
#        optimizer0.step()
#        optimizer1.step()
#        print(loss0.item(), loss1.item())
#        
#        if epoch % 50 == 0:
#            # Print test loss
#            y_pred0 = model0(X_test0)
#            y_pred1 = model1(X_test1)
#            test_loss0 = criterion0(y_pred0.squeeze(), y_test0)
#            test_loss1 = criterion1(y_pred1.squeeze(), y_test1)
#            
#            print("TEST LOSSES at epoch {}: {}, {}".format(epoch, test_loss0, test_loss1))

    