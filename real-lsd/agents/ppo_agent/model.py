import math
import random
import os
import glog as log

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions import Categorical

from torchviz import make_dot

def init_weights(m):
    if isinstance(m, nn.Linear):
        # Fills the the weight Tensor with values drawn from the normal distribution
        nn.init.normal_(m.weight, mean=0., std=0.01)
        # Fills the bias Tensor with the constant value
        nn.init.constant_(m.bias, 0.1)

class ActorCritic(nn.Module):
    # nn.Module is a base class for all neural network modules.
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        # create mlp model for critic -> 2 layers with hidden_size neurons each
        # A sequential container. Modules will be added to it in the order they
        # are passed in the constructor. Alternatively, an ordered dict of
        # modules can also be passed in.
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.ReLU(),
            nn.LogSoftmax(dim=0)
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

        self.apply(init_weights)

    def forward(self, x):
        value = self.critic(x)
        discrete_log_probabilitiies = self.actor(x)
        #make_dot(discrete_log_probabilitiies, params=dict(list(self.actor.named_parameters()))).render("actor", format="png")
        #make_dot(value, params=dict(list(self.critic.named_parameters()))).render("critic", format="png")

        assert (torch.sum(torch.isnan(discrete_log_probabilitiies)) == 0), "Actor output contains NANs!"

        discrete_probabilitiies = torch.exp(discrete_log_probabilitiies)
        dist = Categorical(discrete_probabilitiies)
        # continuous action space
        # std   = self.log_std.exp().expand_as(mu)
        # dist  = Normal(mu, std)
        return dist, value
