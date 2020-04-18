# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:16:28 2020

@author: nitin
"""

from flask import Flask, jsonify

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route("/update_project/<project_data>", methods=['GET','POST'])
def updateWPProject(project_data):
    """project_data = user+++++pw+++++pj_jsondoc.
    This will update an existing project data or insert a non-existant project data item."""
    print(project_data)
    return jsonify(project_data)
 