import os
import io
import gym
import time
import torch
import random
# import torch.nn as nn
import pickle
import inspect

import numpy as np
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import gym_unrealcv
import real_lsd

import json
from unrealcv import client
from unrealcv.util import read_npy, read_png
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import glog as log


def files_in_dataset(dataset):
    dir = 'data'
    path = os.path.dirname(real_lsd.__file__)
    path = os.path.split(path)[0]
    assert dir in os.listdir(path)
    path = path + '/' + dir + '/' + dataset
    list_of_files = os.listdir(path)
    return list_of_files

def get_episode_mean_reward(data):
    num_of_episodes = data.shape[1]
    avg_rewards = np.zeros(num_of_episodes)
    for i in range(num_of_episodes):
        avg_rewards[i] = np.asarray(doc['episode_{}'.format(i+1)]['rewards']).mean()
    return avg_rewards


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def load_obj(dataset, filename):
    dir = 'data'
    path = os.path.dirname(real_lsd.__file__)
    path = os.path.split(path)[0]
    assert dir in os.listdir(path)
    path = path + '/' + dir + '/' + dataset + '/'
    with open(path + filename + '.pkl', 'rb') as f:
        return CPU_Unpickler(f).load()


# select dataset and episode
dataset = 'dataset_eighteen'
ep      = 11

# select correct file
list_of_files = files_in_dataset(dataset)
list_of_files.remove('specs.txt')
list_of_files.remove('cpptest.json')
list_of_files.remove('training_data.pkl')
file = list_of_files[0][:-4]
print(file)
doc = load_obj(dataset, file)
training_data = load_obj(dataset, 'training_data')

# load data into panda
data = pd.DataFrame.from_dict(doc)

num_of_episodes = data.shape[1]

# for j in range(num_of_episodes):
steps = len(data['episode_{}'.format(ep)]['trajectory'])
x = np.zeros(steps, dtype=np.float32)
y = np.zeros(steps, dtype=np.float32)
z = np.zeros(steps, dtype=np.float32)

for i in range(steps):
    x[i] = data['episode_{}'.format(ep)]['trajectory'][i][0]
    y[i] = data['episode_{}'.format(ep)]['trajectory'][i][1]
    z[i] = data['episode_{}'.format(ep)]['trajectory'][i][2]

client.connect()
print('should have connected to client')

if not client.isconnected():
    print('isconnected is False')
    print('UnrealCV server is not running. Run the game from http://unrealcv.github.io first.')
else:
    print('unrealcv server is running')

print('about to run steps')
for j in range(steps):
    cmd = 'vset /camera/0/moveto {} {} {}'.format(x[j], y[j], z[j])
    time.sleep(1)
    client.request(cmd)
    img_cam_1 = client.request('vget /camera/0/lit file1%02d.png' % j)
    img_cam_2 = client.request('vget /camera/2/lit file2%02d.png' % j)

print('DONE')
