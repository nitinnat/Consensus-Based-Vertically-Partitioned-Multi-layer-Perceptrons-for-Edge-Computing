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

import torch.nn.functional as F
warnings.filterwarnings('ignore')

# Todo: data loading should be done in another class
# Todo: Make each model into a class and pass this into higher NNTrainer class
class NeuralNetwork:
    
    def __init__(self, nnconfig_dict):
        torch.manual_seed(0)
        self.dataset_name = nnconfig_dict['dataset_name']
        self.node_id = nnconfig_dict['node_id']
        self.feature_split = nnconfig_dict['feature_split']
        self.base_dir = "C:/Users/nitin/eclipse-workspace/consensus-deep-learning-version-2.0/data/"
        
        # Weights and biases
        self.weights1 = None
        self.bias1 = None
        self.weights2 = None 
        self.bias2 = None 
        
        # Training parameters
        # Todo: should be put in higher-level trainer class
        self.learning_rate = 0.2
        self.epochs = 0 # Number of epochs or cycles is controlled by peersim. Here we maintain a count
        self.runtype = nnconfig_dict['runtype']
        self.loss_func = F.cross_entropy
        self.loss_arr = []
        self.acc_arr = []
    
    def load_data(self):
        if self.runtype == "centralized":
            train_filename = "{}_{}.csv".format(self.dataset_name, "train_binary")
            test_filename = "{}_{}.csv".format(self.dataset_name, "test_binary")
        
        else:
            train_filename = "{}_{}_{}.csv".format(self.dataset_name, "train", self.node_id)
            test_filename = "{}_{}_{}.csv".format(self.dataset_name, "test", self.node_id)
        
        df_train = pd.read_csv(os.path.join(self.base_dir, 
                                      self.dataset_name, 
                                      "feature_split_" + str(self.feature_split),
                                     train_filename), header=None).to_numpy()
        
        
        df_test= pd.read_csv(os.path.join(self.base_dir, 
                                      self.dataset_name, 
                                      "feature_split_" + str(self.feature_split),
                                      test_filename), header=None).to_numpy()
        
        
        # Split into features and labels
        self.X_train = df_train[:,1:]
        self.y_train = df_train[:,0]
        
        self.X_test= df_test[:,1:]
        self.y_test = df_test[:,0]
        
        
        # Map to tensors so torch can use this data
        self.X_train, self.y_train, self.X_test, self.y_test = map(torch.tensor, (self.X_train, self.y_train, self.X_test, self.y_test))
        
        # Convert features to float and labels to long
        self.X_train = self.X_train.float()
        self.y_train = self.y_train.long()
        self.X_test = self.X_test.float()
        self.y_test = self.y_test.long()
        
        
        print(self.X_train.shape, self.X_test.shape, 
              self.y_train.shape, self.y_test.shape)
        
    
    def mlp(self):
        # If accessing this function for the first time, then 
        # initialize the weights as per input data size
        if self.weights1 is None:
            self.weights1 = torch.randn(self.X_train.shape[1],100) / math.sqrt(2)
            self.weights1.requires_grad_()
            self.bias1 = torch.zeros(100, requires_grad=True)
            self.weights2 = torch.randn(100,2)/math.sqrt(2)
            self.weights2.requires_grad_()
            self.bias2 = torch.zeros(2, requires_grad=True)
        
        A1 = torch.matmul(self.X_train, self.weights1) + self.bias1
        H1 = A1.sigmoid()
        A2 = torch.matmul(H1, self.weights2) + self.bias2
        H2 = A2.exp()/A2.exp().sum(-1).unsqueeze(-1)
        return H2
    
    @staticmethod
    def accuracy(y_hat, y):
        pred = torch.argmax(y_hat, dim=1)
        return (pred==y).float().mean()
        
    
    def train(self):
        """
        Applies the feedforward function on the network
        """
        self.epochs += 1
        y_hat = self.feedforward()
        loss = self.backpropagate(y_hat)
        self.loss_arr.append(loss.item())
        self.acc_arr.append(self.accuracy(y_hat, self.y_train))
        print("Loss at iteration: {}".format(loss.item()))

            
    # todo: Factory pattern class for NN Model - should implement train, FF and BP
    def feedforward(self):
        y_hat = self.mlp() # feedforward
        return y_hat
    
    # Takes in the output vector from FF function
    def backpropagate(self, y_hat, clear_grad=True):
        loss = self.loss_func(y_hat, self.y_train)
        loss.backward(retain_graph=True)
        
        # Update weights and biases
        
        with torch.no_grad():
            self.weights1 -= self.weights1.grad * self.learning_rate
            self.bias1 -= self.bias1.grad * self.learning_rate
            self.weights2 -= self.weights2.grad * self.learning_rate
            self.bias2-= self.bias2.grad * self.learning_rate
            
            if clear_grad:
                # Have to clear the gradients after each epoch, since loss.backward() adds gradients to 
                # existing gradient values
                self.weights1.grad.zero_()
                self.bias1.grad.zero_()
                self.weights2.grad.zero_()
                self.bias2.grad.zero_()
    
    
    # Updates two nodes and returns updated neighbor node
    def gossip(self, neighbor_nn):
        
        # Feed-forward on current node
        y_hat_cur = self.feedforward()
        
        # Feed-forward on neighbor node
        y_hat_neighbor = neighbor_nn.feedforward()
        
        # Average the output vectors
        y_mean = (y_hat_cur + y_hat_neighbor)/2
        
        # Backpropagate on both
        self.backpropagate(y_mean, clear_grad=True)
        neighbor_nn.backpropagate(y_mean, clear_grad=True)
        print("Loss at node {} in iteration {} is: {}".format(self.node_id, self.epochs, self.loss))
        # Update current node's epochs parameter
        self.epochs += 1
        return neighbor_nn
        
        
        
        
    

    