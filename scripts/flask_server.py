# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:16:28 2020

@author: nitin
"""

from flask import Flask, jsonify
from neuralnet import NeuralNetwork
import sys
import flask
import json
import traceback
import matplotlib.pyplot as plt
from flask_socketio import SocketIO
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)
socketio = SocketIO(app)
# This dictionary will store all the neural networks
neural_network_dict = {}

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
                    node_id = nnconfig_dict["node_id"]        
                    
                    
                    if command == "init":
                        neural_network_dict[node_id] = NeuralNetwork(nnconfig_dict)
                        neural_network_dict[node_id].load_data()
                    
                    if command == "train":
                        if node_id in neural_network_dict.keys():
                            neural_network_dict[node_id].train()
                    
                    if command == "plot":
                        if node_id in neural_network_dict.keys():
                            plot_x, plot_y = list(range(neural_network_dict[node_id].epochs)), neural_network_dict[node_id].loss_arr 
                            # Check https://stackoverflow.com/questions/50728328/python-how-to-show-matplotlib-in-flask
                            fig, ax = plt.subplots()
                            ax.plot(plot_x, plot_y)
                            output = io.BytesIO()
                            FigureCanvas(fig).print_png(output)
                            return Response(output.getvalue(), mimetype='image/png')
                        
                    if command == "gossip":
                        # Gossip only works for runtype == "distributed"
                        neighbor_node_id = nnconfig_dict["neighbor"]
                        assert nnconfig_dict["runtype"] == "distributed", "Cannot gossip in centralized setting"
                        assert (node_id in neural_network_dict.keys() and neighbor_node_id in neural_network_dict.keys()), \
                        "Either one of {} or {} is not present in created list of nodes".format(node_id, neighbor_node_id) 
                        neighbor_node_id = nnconfig_dict["neighbor"]
                        
                        # Perform gossip with neighbor's dict as the parameter and update both neural networks
                        print("Node {} is gossipping with node {}.".format(node_id, neighbor_node_id))
                        neural_network_dict[neighbor_node_id] = \
                        neural_network_dict[node_id].gossip(neural_network_dict[neighbor_node_id])

                    print("Number of neural networks currently existing: {}".format(len(neural_network_dict)))
                except Exception as e:
                    track = traceback.format_exc()
                    print(track)
    
    return jsonify(command)



if __name__ == '__main__':
    socketio.run(app)