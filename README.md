# Perception and Learning for Robotics

This exam project was developed during the semester to take advantage of deep reinforcement learning to train a drone how to land in a realistic and challenging environment (a glacier) using only a mounted RGB camera. The project is based on the work done by Nasib Naimi for his semester project at the Autonomous Systems Lab ETH Zurich.  
  
For the deep reinforcement learning framework, the OpenAI algorithm [Proximal Policy Optimization 2 (PPO2)](https://openai.com/blog/openai-baselines-ppo/) is employed, while Unreal Engine 4.16 is used for the simulation. [UnrealCV](https://unrealcv.org/) and [Gym-UnrealCV](https://github.com/zfw1226/gym-unrealcv) are exploited to connect the RL framework with the simulation of the environment. 

## Setup

The repository is composed of two folder: 
* [gym-unrealcv](gym-unrealcv) is a fork of the original [Gym-UnrealCV](https://github.com/zfw1226/gym-unrealcv) repository. Please read the associated updated [README](https://github.com/zfw1226/gym-unrealcv/blob/v1.0/README.md);
* [real-lsd](real-lsd) is the main container of the RL framework. Note that it is provided only the RL framework, but not the UE4 simulation environment; the user has to add its own simulation. Please follow the associated [README](real-lsd/README.md).

