import os
import math
import random

import gym
import glog as log
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

# from IPython.display import clear_output
import matplotlib.pyplot as plt
# Import network
from model import ActorCritic

class PPOAgent():
    def __init__(self,
                    num_inputs,
                    num_actions, # number of outputs of actor network
                    hidden_size,
                    lr,
                    num_steps,
                    mini_batch_size,
                    ppo_epochs,
                    threshold_reward=(-200),
                    std=0.0,
                    device="cpu"):

        # Class attributes
        self.num_inputs = num_inputs
        self.num_actions = num_actions

        self.model = ActorCritic(num_inputs, num_actions, hidden_size).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, action_probabilities):
        """
        Use the network to predict the next action to take, using the model
        example:
        pred = np.array([0.05, 0.85, 0.1])
        action_size = 3
        np.random.choice(a, p=pred)
        result>>> 1, because it have the highest probability to be taken
        """
        action = np.random.choice(self.num_actions, p=action_probabilities)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        return action, action_onehot

    def minibatch_loss(self, states, actions, old_log_probs, returns, advantages, clip_param=0.2):
        # Distributions of all actions for each given state in minibatch
        log.info("Calculating minibatch loss.")
        dist, value = self.model(states)
        entropy = dist.entropy().mean()
        new_log_probs = dist.log_prob(actions)

        ratio = (new_log_probs - old_log_probs).exp()

        term1 = ratio * advantages
        term2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages

        actor_loss  = - torch.min(term1, term2).mean()
        critic_loss = (returns - value).pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy
        log.info("Minibatch loss: {} SIZE: {}".format(loss, loss.size()))
        return loss

    def ppo_iter(self, mini_batch_size, states, actions, log_probs, returns, advantage):
        """
        Divide batch into mini_batches through generator
        mini_batch set is uniformly sampled from the batch
        """
        log.info("Creating minibatch.")
        batch_size = states.size(0)
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]

    def ppo_update(self, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        log.info("PPO update called.")
        for i in range(ppo_epochs):
            log.info("PPO update epoch: {}".format(i))
            for state, action, old_log_probs, return_, advantage in self.ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                log.info("PPO update epoch: {} Optimizing on minibatches".format(i))

                loss = self.minibatch_loss(state, action, old_log_probs, return_, advantage)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def compute_gae(self, next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        log.info("Computing GAE")

        values = values + [next_value]
        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])

        log.info("Dimension of return: {}".format(returns[0].size()))
        return returns

    def save_model(self):
        import time
        # PATH     = os.getcwd()
        PATH     = '/media/scratch2/plr_project/PLR'
        dir      = 'models'
        filename = time.strftime("%Y%m%d_%H%M%S")
        if dir not in os.listdir(PATH):
            PATH = os.path.join(PATH, dir)
            os.mkdir(PATH)
        else:
            PATH = os.path.join(PATH, dir)

        file_abs_path = PATH + '/' + filename + '.pth'
        torch.save(self.model.state_dict(), file_abs_path)

    def load_model(self, filename):
        PATH = os.getcwd()
        dir  = 'models'
        assert dir in os.listdir(PATH)
        PATH = PATH + '/' + dir + '/' + filename + '.pth'
        self.model = torch.load(PATH)



# if __name__ == '__main__':
