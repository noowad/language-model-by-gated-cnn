# coding:utf-8
import tensorflow as tf
import os
from graph import Graph

g = Graph()
params = []
with g.graph.as_default():
    for param in tf.trainable_variables():
        print param
        params.append(param)

