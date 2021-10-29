# Perception and Learning for Robotics

## Deep-RL-based safety landing using RGB camera on rough terrains

This exam project was developed during one semester to take advantage of deep reinforcement learning to train a drone how to land in a realistic and challenging environment (a glacier) using only a mounted RGB camera.  
  
The paper of the project can be read in [Report.pdf](Report.pdf).  
  
The project is based on the work done by Nasib Naimi in his semester thesis at the Autonomous Systems Lab ETH Zurich (his thesis is available in [real-lsd/Nasib report.pdf](real-lsd/Nasib_report.pdf)).

<p align="center">
 <img height="200" src="https://user-images.githubusercontent.com/79461707/139410971-46d09793-d9d4-4b47-8ed0-f097390972e1.png"/>
 <img height="200" src="https://user-images.githubusercontent.com/79461707/139410959-4b105031-8f48-40ce-8f44-9c0b5c51d61b.png"/>
</p>

<p align="center">
 <img height="150" src="https://user-images.githubusercontent.com/79461707/139410341-b74b38ec-8bea-4efe-860e-a524153fb171.png"/>
</p>

### Framework
For the deep reinforcement learning framework, the OpenAI algorithm [Proximal Policy Optimization 2 (PPO2)](https://openai.com/blog/openai-baselines-ppo/) is employed, while Unreal Engine 4.16 is used for the simulation. The Multi-Layer Perceptron (MLP) policy and the Convolutional Neural Network (CNN) policy were tested feeding as input the RGB images captured from the camera mounted under the drone, looking downwards towards the ground (pitch of -90◦).  
  
[UnrealCV](https://unrealcv.org/) and [Gym-UnrealCV](https://github.com/zfw1226/gym-unrealcv) are exploited to connect the RL framework with the simulation of the environment. 

![ppt_framework](https://user-images.githubusercontent.com/79461707/139410041-56a97cf6-096b-4a98-a7eb-b38367b148d4.png)

### Setup

The repository is composed of two folders: 
* [gym-unrealcv](gym-unrealcv) is a fork of the original [Gym-UnrealCV](https://github.com/zfw1226/gym-unrealcv) repository. Please read the associated updated [gym-unrealcv/README](https://github.com/zfw1226/gym-unrealcv/blob/v1.0/README.md);
* [real-lsd](real-lsd) is the main container of the RL framework. Note that it is provided only the RL framework, but not the UE4 simulation environment. Please follow the instructions on the associated [real-lsd/README](real-lsd/README.md).

