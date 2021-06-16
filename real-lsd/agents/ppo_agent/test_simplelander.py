import os
import sys
import gym

from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2
import gym_unrealcv
import real_lsd
import glog as log

import time
import torch
import torch.nn as nn
import pickle
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# initialise environment
# env = gym.make('MyUnrealLand-cpptestFloorGood-DiscretePoseColor-v0')

# multiprocess environment
# example env name
# UnrealLand-cpptestFloorGood-DiscretePoseColor-v0
env = gym.make('MyUnrealLand-gornerFloorGood-Discrete-v0')
# env = make_vec_env('UnrealSearch-RealisticRoomDoor-DiscreteColor-v0', n_envs=1)

# PP02 with mlp network for both actor and critic, both with two layers and 64
# neurons each

'''--------------------------------------------------'''

#policy = MlpLstmPolicy               # MlpPolicy, MlpLstmPolicy, MlpLnLstmPolicy, CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy 
#max_frames = 45000

'''--------------------------------------------------'''


def save_obj(obj, filename, timestamp):
    dir = timestamp
    # path = os.getcwd()
    PATH = '/media/scratch2/plr_project/PLR/data/eval_test'  #'/home/plr/PLR'    # '/media/scratch2/plr_project/PLR'
    
    if dir not in os.listdir(PATH):
        PATH =  os.path.join(PATH, dir)
        os.mkdir(PATH)
    else:
        PATH =  os.path.join(PATH, dir)
    
    abs_file_path = PATH + '/' + filename + '.pkl'
    print("1")
    with open(abs_file_path, 'wb') as f:
        print("2")

        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        print("3")

    return filename

def save_train(obj, filename, timestamp):
    dir = timestamp
    # path = os.getcwd()
    PATH = '/media/scratch2/plr_project/PLR/data/data_train'  #'/home/plr/PLR'    # '/media/scratch2/plr_project/PLR'
    
    if dir not in os.listdir(PATH):
        PATH =  os.path.join(PATH, dir)
        os.mkdir(PATH)
    else:
        PATH =  os.path.join(PATH, dir)
    
    abs_file_path = PATH + '/' + filename + '.pkl'
    print("1")
    with open(abs_file_path, 'wb') as f:
        print("2")

        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

        print("3")

    return filename

def load_obj(filename):
    dir = 'data'
    path = os.getcwd()
    assert dir in os.listdir(path)
    path = path + '/' + dir + '/'
    with open(path + filename + '.pkl', 'rb') as f:
        return pickle.load(f)

# get activation of layers in model
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def plot_roughness(train_timestamp,frame_idx, rewards):
#     clear_output(True)
     PATH = "/media/scratch2/plr_project/PLR/data/test_roughness"
     if train_timestamp not in os.listdir(PATH):
         PATH =  os.path.join(PATH, train_timestamp)
         os.mkdir(PATH)
     else:
         PATH =  os.path.join(PATH, train_timestamp)
     plt.figure(figsize=(20,5))
     plt.subplot(131)
     plt.title('frame %s. Roughness: %s' % (frame_idx, rewards[-1]))
     plt.plot(rewards)
     out_plot = PATH + '/' + "{}_training_roughness_{}.png".format(train_timestamp,frame_idx)
     plt.savefig(out_plot, format='png')
     #plt.show()

def plot_slope(train_timestamp,frame_idx, rewards):
#     clear_output(True)
     PATH = "/media/scratch2/plr_project/PLR/data/test_slope"
     if train_timestamp not in os.listdir(PATH):
         PATH =  os.path.join(PATH, train_timestamp)
         os.mkdir(PATH)
     else:
         PATH =  os.path.join(PATH, train_timestamp)
     plt.figure(figsize=(20,5))
     plt.subplot(131)
     plt.title('frame %s. slope: %s' % (frame_idx, rewards[-1]))
     plt.plot(rewards)
     out_plot = PATH + '/' + "{}_training_slope_{}.png".format(train_timestamp,frame_idx)
     plt.savefig(out_plot, format='png')

def plot_trajectory(poses_x, poses_y, poses_z, test, episode_count, info, timestamp,total_reward):
    PATH = "/media/scratch2/plr_project/PLR/data/test_traj"
    if timestamp not in os.listdir(PATH):
        PATH =  os.path.join(PATH, timestamp)
        os.mkdir(PATH)
    else:
        PATH =  os.path.join(PATH, timestamp)
    plt.figure(figsize=(20,5))
    plt.subplot(111, projection='3d')
    plt.title('Trajectory, reward: '+str(total_reward))
    plt.plot(poses_x, poses_y, poses_z)
    if info['Success']:
        out_plottraj= PATH + '/' + "{}_{}_{}_succ_{}.png".format(timestamp,test,episode_count, total_reward)
    elif info['Max Step']:
        out_plottraj=PATH + '/' + "{}_{}_{}_max_{}.png".format(timestamp,test,episode_count, total_reward)
    elif info['Collision']:
        out_plottraj=PATH + '/' + "{}_{}_{}_coll_{}.png".format(timestamp,test,episode_count, total_reward)
    elif info['Out_of_boundaries']:
        out_plottraj=PATH + '/' + "{}_{}_{}_out_{}.png".format(timestamp,test,episode_count, total_reward)
    else:
        out_plottraj=PATH + '/' + "{}_{}_{}_fail_{}.png".format(timestamp,test,episode_count, total_reward)
    plt.savefig(out_plottraj, format='png')


def test_env():
    state = env.reset()
    done = False
    steps = 0
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        action, _states = model.predict(state)
        next_state, reward, done, info = env.step(action.cpu().numpy())
         
        state = next_state
        steps +=1
        total_reward += reward

    mean_reward = total_reward / steps
    return [total_reward, mean_reward]

'''------------------- Testing the policy after training --------------------'''

log.setLevel("WARN")

# load the model file if needed

train_timestamp = "20210609_162828_PPO2"
model = PPO2.load("/media/scratch2/plr_project/PLR/data/data_train/" + train_timestamp + "_training_data")


# Testing parameters
num_tests = 4
episodes_per_test = np.array([50]*4)
tot_successful_episodes = 0
successful_episodes = 0

test_timestamp = train_timestamp #time.strftime("%Y%m%d_%H%M%S")

state = env.reset()
with torch.no_grad():
    for test in range(num_tests):
        episode_count = 0
        num_test_episodes = episodes_per_test[test]

        episodes = {}
        state = env.reset()

        while episode_count < num_test_episodes:

            done = False
            episode = {}
            total_reward = 0
            poses_x, poses_y, poses_z     = [], [], []
            #states    = [state]
            #dists     = []
            #values    = []
            actions   = []
            rewards   = []
            #log_probs = []
            traj      = []
            mesh_dists = []
            slope = []
            roughness = []

            while not done:
                action = 0
                #dist = 0
                #value = 0
                #log_prob = 0
                #states.append(state)

                #state = torch.FloatTensor(state).to(device)
                
                action, _states = model.predict(state)
                next_state, reward, done, info = env.step(action)
                #env.render()
                total_reward += reward
                log.warn("Step REWARD: {} DONE: {} TOTAL: {}".format(reward, done,total_reward))
                log.warn("Distance to mesh: {}".format(info['Mesh_dists']))

                poses_x.append(info['Pose'][0])
                poses_y.append(info['Pose'][1])
                poses_z.append(info['Pose'][2])
                slope.append(info['slope'])
                roughness.append(info['roughness'])

                #poses = np.concatenate((poses,info['Pose'].np), axis=1)
                #dists.append(dist)
                #values.append(value)
                actions.append(action)
                #log_probs.append(log_prob)
                rewards.append(reward)
                mesh_dists.append(info['Mesh_dists'])

                # next state logic
                if done:
                    if info['Success']:
                        successful_episodes += 1
                        tot_successful_episodes += 1
                    traj = info['Trajectory']
                    #log.warn(info)
                    state = env.reset()
                    episode_count += 1
                else:
                    state = next_state

            #episode['poses']    = poses
            #episode['states']   = states
            #episode['dists']    = dists
            #episode['values']   = values
            episode['actions']  = actions
            episode['rewards']  = rewards
            #episode['log_probs'] = log_probs
            episode['trajectory']= traj
            episode['mesh_dists'] = mesh_dists

            key = 'episode_{}'.format(episode_count)
            episodes[key] = episode

            log.warn("Successes out of {}: {}".format(episode_count, successful_episodes))
            log.warn("Timestamp: {}".format(train_timestamp))
            plot_trajectory(poses_x, poses_y, poses_z, test, episode_count, info, train_timestamp, total_reward)
            plot_roughness(train_timestamp, test, roughness)
            plot_slope(train_timestamp, test, slope)

        filename = test_timestamp + str(test) +'_'+ str(num_test_episodes)+'-' +str(successful_episodes)
        log.warn("Successes out of {}: {}".format(num_test_episodes, successful_episodes))
        log.warn("Total Successes out of {}: {}".format(episodes_per_test[0:test+1].sum(), tot_successful_episodes))
        log.warn("Timestamp: {}".format(train_timestamp))
        log.warn("About to save the test data.")
        file = save_obj(episodes, filename, train_timestamp)
        del episodes
        # print(load_obj(file))
        successful_episodes = 0


log.warn("Total Successes out of {}: {}".format(episodes_per_test[0:test+1].sum(), tot_successful_episodes))

log.warn("Done Testing.")

env.close()

sys.exit('Training and testing completed.')

