# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:16:28 2020

@author: nitin
"""

from flask import Flask, jsonify
import sys
import flask
import pandas as pd
import json
import traceback
import matplotlib.pyplot as plt
from flask_socketio import SocketIO
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from neuralnet import NeuralNetworkCluster
import os

app = Flask(__name__)
socketio = SocketIO(app)

# This dictionary will store all the neural networks
base_dir = "C:/Users/nitin/eclipse-workspace/consensus-deep-learning-version-2.0/data"
nn_cluster = NeuralNetworkCluster(base_dir)

def save_results(op_path):
    """
    Stores a pickle file of the NeuralNetworkCluster object
    """
    global nn_cluster, base_dir
    import pickle
    pickle.dump(nn_cluster, open(os.path.join(op_path, "results.pkl"), "wb"))
        
@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/plot.png')
def plot_png():
    fig = create_figure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return 

@app.route("/vpnn/<command>", methods=['GET','POST'])
def updateWPProject(command):
    """project_data = user+++++pw+++++pj_jsondoc.
    This will update an existing project data or insert a non-existant project data item."""
    print("Executing {} ".format(command))
    if flask.request.method == 'POST':
            nnconfig = flask.request.form['nnconfig']
            
            if nnconfig:
                try:
                    nnconfig_dict = json.loads(nnconfig)
                    # doing a mod here because Peersim does not reset pegasosnode counter
                    node_id = nnconfig_dict["node_id"]%nnconfig_dict["num_nodes"]
                    
                    
                    if command == "clear":
                        global nn_cluster
                        nn_cluster = NeuralNetworkCluster(base_dir)
                        
                        
                    if command == "init":
                        if len(nn_cluster.neuralNetDict) == 0:
                            num_nodes = int(nnconfig_dict["num_nodes"])
                            if nnconfig_dict["feature_split_type"] == "overlap":
                                nn_cluster.init_data(nnconfig_dict["dataset_name"], num_nodes,
                                    nnconfig_dict["feature_split_type"], nnconfig_dict["random.seed"], nnconfig_dict["overlap_ratio"])
                            else:
                                nn_cluster.init_data(nnconfig_dict["dataset_name"], num_nodes,
                                    nnconfig_dict["feature_split_type"], nnconfig_dict["random.seed"])
                       
                        nn_cluster.appendNNToCluster(nnconfig_dict)
                    
                    if command == "train":
                        nn_cluster.train(node_id)
                        
                    if command == "calc_losses":
                        nn_cluster.compute_losses_and_accuracies()
                        
                    if command == "plot":
#                        if node_id in neural_network_dict.keys():
#                            plot_x, plot_y = list(range(neural_network_dict[node_id].epochs)), neural_network_dict[node_id].loss_arr 
#                            # Check https://stackoverflow.com/questions/50728328/python-how-to-show-matplotlib-in-flask
#                            fig, ax = plt.subplots()
#                            ax.plot(plot_x, plot_y)
#                            output = io.BytesIO()
#                            FigureCanvas(fig).print_png(output)
#                            return Response(output.getvalue(), mimetype='image/png')
                        
                        loss_df = pd.DataFrame(columns=["Node", "Iter", "TrainLoss", "TestLoss"])
                        for node_id in nn_cluster.neuralNetDict.keys():
                            
                            train_losses = nn_cluster.neuralNetDict[node_id]["train_losses"]
                            test_losses = nn_cluster.neuralNetDict[node_id]["test_losses"]
                            nodes = [node_id]*len(train_losses)
                            iters = list(range(len(train_losses)))
                            df = pd.DataFrame(data={"Node": nodes, "Iter": iters, 
                                                    "TrainLoss": train_losses, "TestLoss":test_losses})
                            loss_df = loss_df.append(df)
                            
                        op_path = os.path.join(base_dir, nnconfig_dict["dataset_name"])
                        
                        if not os.path.exists(op_path):
                            os.makedirs(op_path)
                        loss_df.to_csv(os.path.join(op_path,"{}_TrainLosses_{}.csv".format(nnconfig_dict["run_type"], nnconfig_dict["run"])), index=False)
                        print("Node: {}: {}".format(0, nn_cluster.neuralNetDict[0]["train_losses"]))
                        
                    if command == "gossip":
                        # Gossip only works for runtype == "distributed"
                        neighbor_node_id = nnconfig_dict["neighbor"]%nnconfig_dict["num_nodes"]
                        assert nnconfig_dict["run_type"] == "distributed", "Cannot gossip in centralized setting"

                        # Perform gossip with neighbor's dict as the parameter and update both neural networks
                        print("Node {} is gossipping with node {}.".format(node_id, neighbor_node_id))
                        nn_cluster.gossip(node_id, neighbor_node_id)
                    
                    if command == "save_results":
                        op_path = os.path.join(base_dir, nnconfig_dict["dataset_name"])
                        save_results(op_path, nnconfig_dict)
                        print("saved results")
                        
                    print("Number of neural networks currently existing: {}".format(len(nn_cluster.neuralNetDict.keys())))
                    
                
                except Exception as e:
                    track = traceback.format_exc()
                    print(track)
    
    return jsonify(command)



if __name__ == '__main__':
    socketio.run(app)