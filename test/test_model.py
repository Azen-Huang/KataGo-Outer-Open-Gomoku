#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
sys.path.append('../src')

import socket
import config
import pickle

from collections import deque
from os import path, mkdir
import threading
import time
import math
import numpy as np
import pickle
import concurrent.futures
import random
from functools import reduce

import sys
sys.path.append('../build')
# from library import MCTS, Gomoku, NeuralNetwork

from neural_network import NeuralNetWorkWrapper, NeuralNetWork
from gomoku_gui import GomokuGUI

config = config.config
n = config['n']
n_in_row = config['n_in_row']
use_gui = config['use_gui']
gomoku_gui = GomokuGUI(config['n'], config['human_color'])
action_size = config['action_size']
is_debug = config['is_debug']

# train
max_buffer_size = 40960
num_iters = config['num_iters']
num_eps = config['num_eps']
num_train_threads = config['num_train_threads']
check_freq = config['check_freq']
num_contest = config['num_contest']
dirichlet_alpha = config['dirichlet_alpha']
temp = config['temp']
update_threshold = config['update_threshold']
num_explore = config['num_explore']
start_idx = config['start_idx']

max_num_explore = config['max_num_explore']

examples_buffer = deque([], maxlen=config['examples_buffer_max_len'])

# mcts
num_mcts_sims = config['num_mcts_sims']
c_puct = config['c_puct']
c_virtual_loss = config['c_virtual_loss']
num_mcts_threads = config['num_mcts_threads']
libtorch_use_gpu = config['libtorch_use_gpu']

# neural network
batch_size = config['batch_size']
epochs = config['epochs']

model = NeuralNetWork(128, 128, 15, 225)
print(model)
print()
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
#nnet.save_model()