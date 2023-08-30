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
from library import MCTS, Gomoku, NeuralNetwork

from neural_network import NeuralNetWorkWrapper
from gomoku_gui import GomokuGUI

config = config.config


PORT = 7000
# model_path = './models/best_checkpoint.pt'
model_path = './models/checkpoint.pt'
# num_mcts_sims = 100000 + 225 * 16
# num_mcts_threads = 32 
num_mcts_sims = config['num_mcts_sims']
num_mcts_threads = config['num_mcts_threads']
 # gomoku
n = config['n']
n_in_row = config['n_in_row']
action_size = config['action_size']
is_debug = True
add_noise = True
# mcts
#num_mcts_sims = config['num_mcts_sims']
#num_mcts_threads = config['num_mcts_threads']
c_puct = config['c_puct']
c_virtual_loss = config['c_virtual_loss']
libtorch_use_gpu = config['libtorch_use_gpu']
temp = config['temp']

def get_probs(indata):
    print(indata)
    indata = indata[1:-2].split(',')
    if (indata[0] == ''):
        indata = []
    print(indata)

    libtorch_best = NeuralNetwork(model_path, libtorch_use_gpu, 12)
    mcts_best = MCTS(libtorch_best, num_mcts_threads, \
            c_puct, num_mcts_sims, c_virtual_loss, action_size, add_noise)

    gomoku = Gomoku(n, n_in_row, 1 if len(indata) % 2 == 1 else -1)
    for move in indata:
        gomoku.execute_move(int(move))
    
    prob = mcts_best.get_action_probs(gomoku, 0, is_debug)
    values = gomoku.get_action_value()
    value = values[prob.index(max(prob))]
    prob = ' '.join(map(str, prob))
    print("value: ", value)
    # print(prob)
    return prob + ' ' + str(value)

HOST = '0.0.0.0'

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind((HOST, PORT))
s.listen(5)

print('server start at: %s:%s' % (HOST, PORT))
print('wait for connection...')

while True:
    conn, addr = s.accept()
    print('connected by ' + str(addr))

    while True:
        indata = conn.recv(1024)
        if len(indata) == 0: # connection closed
            conn.close()
            print('client closed connection.')
            break
        print('recv: ' + indata.decode())
        
        # outdata = '1 ' * 225
        outdata = get_probs(indata.decode())
        outdata = outdata + '\n'
        # print(outdata)
        # outdata = 'echo ' + indata.decode()
        conn.send(outdata.encode())

s.close()