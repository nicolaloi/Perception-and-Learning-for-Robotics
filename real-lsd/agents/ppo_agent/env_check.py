import os
import gym
import time
import torch
import torch.nn as nn
import pickle
import numpy as np
import glog as log

# from stable_baselines import PPO2
# from stable_baselines.common import make_vec_env
# from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.env_checker import check_env

import gym_unrealcv
import real_lsd

from PPOagent import PPOAgent


env = gym.make('MyUnrealLand-cpptestFloorGood-DiscreteHeightFeatures-v0')
check_env(env)
